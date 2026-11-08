"""Music ratings: ordinal/NB models, three-fold comparison, and error costs.

Run: python music_cost_cv.py
  MODE = 'cv'      : train folds, tune on inner calibration rows, compare folds.
  MODE = 'retune'  : repeat the decision-rule search using cached probabilities.
  MODE = 'submit'  : refit the selected approach on ALL labeled rows; save a CSV.

No neural networks, no boosting. The NB transformation is inspired by NB-SVM,
not an exact implementation of that paper. Costs change the final DECISION;
they do not change model training or make probabilities calibrated.

References to explain/cite in your README (also describe AI assistance):
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
https://scikit-learn.org/stable/common_pitfalls.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
https://aclanthology.org/P12-2018/
"""
from pathlib import Path
from time import perf_counter
from itertools import product
import gc
import hashlib
import inspect
import json
import sys

import matplotlib
matplotlib.use("Agg")  # Save figures without opening windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import vstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==================== SETTINGS TO CHANGE ====================
MODE = "submit"                     # "cv", "retune", or "submit"
RUN_NAME = "ordinal_cost_cv_5fold"     # New name when changing model/features/folds.
SUBMISSION_NAME = "ordinal_cost_cv_v1.csv"

N_FOLDS = 5
CALIBRATION_FRACTION = 0.20      # Of each outer TRAINING partition, not all data.
SEED = 42
ALPHA = 0.00001                 # Same regularization as the successful ordinal model.
NB_BLEND = 0.40                 # Change this in "retune" without retraining models.

# Keep this small. These are decision-rule parameters, NOT training parameters.
WEIGHT_OPTIONS = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5]
PAIR_OPTIONS = [0.75, 1.0, 1.25, 1.5]
# Two extra multipliers: one for 2<->3 mistakes and one for 3<->4 mistakes.
# 1.0 = unchanged cost; values below/above 1 reduce/increase that cost.
# ============================================================

BASE = Path(__file__).resolve().parent
LABELS = [1, 2, 3, 4, 5]
ID_COLS = ["reviewerID", "album_mbid", "artist_mbid"]
NUM_COLS = ["helpful_share", "log_votes", "log_length", "year"]
APPROACHES = ["ordinal_weights", "blend_weights", "blend_costs"]


def say(text):
    print(text, flush=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ==================== 1. DATA AND FEATURES ====================
def read_reviews():
    say("Reading train.csv (IDs/text are strings; numerical fields are converted next)...")
    raw = pd.read_csv(BASE / "train.csv", dtype=str, low_memory=False)
    required = ["id", "Score", "reviewText", "summary", "genres",
                "VotedHelpful", "TotalVotes", "unixReviewTime"] + ID_COLS
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"train.csv is missing columns: {missing}")
    if raw["id"].isna().any() or raw["id"].str.strip().eq("").any() or raw["id"].duplicated().any():
        raise ValueError("Review IDs must be nonblank and unique.")
    scores = pd.to_numeric(raw["Score"], errors="coerce")
    nonblank = raw["Score"].fillna("").str.strip().ne("")
    if (nonblank & scores.isna()).any() or not scores.dropna().isin(LABELS).all():
        raise ValueError("Score must be blank or an integer from 1 to 5.")
    raw["Score"] = scores
    return prepare(raw)


def prepare(df):
    """Row-by-row features only; no ratings or dataset-wide estimates are used."""
    df = df.copy()
    for col in ["reviewText", "summary", "genres"] + ID_COLS:
        df[col] = df[col].fillna("").astype(str)
    df["genres"] = df["genres"].str.replace(r"\s*,\s*", ",", regex=True)
    votes = pd.to_numeric(df["TotalVotes"], errors="coerce").fillna(0).clip(lower=0)
    helpful = pd.to_numeric(df["VotedHelpful"], errors="coerce").fillna(0).clip(lower=0)
    df["helpful_share"] = (helpful / votes.replace(0, np.nan)).fillna(0).clip(0, 1)
    df["log_votes"] = np.log1p(votes)
    df["log_length"] = np.log1p(df["reviewText"].str.len())
    df["year"] = pd.to_datetime(pd.to_numeric(df["unixReviewTime"], errors="coerce"),
                                unit="s", errors="coerce").dt.year.fillna(2010)
    df[NUM_COLS] = df[NUM_COLS].astype(np.float32)
    return df


def make_features():
    words = dict(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
                 token_pattern=r"(?u)\b\w+\b")
    return ColumnTransformer([
        ("review", TfidfVectorizer(min_df=3, max_features=80000, **words), "reviewText"),
        ("summary", TfidfVectorizer(min_df=2, max_features=15000, **words), "summary"),
        ("genres", TfidfVectorizer(token_pattern=r"[^,]+", dtype=np.float32), "genres"),
        ("ids", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ID_COLS),
        ("numbers", StandardScaler(), NUM_COLS),
    ], sparse_threshold=1.0)


def transform_batches(features, frame):
    return vstack([features.transform(frame.iloc[i:i+25000])
                   for i in range(0, len(frame), 25000)], format="csr")


# ==================== 2. THE SAME TWO ORDINAL MODELS ====================
def ordinal_probs(q):
    """Convert the four P(Score > cutoff) columns into five rating probabilities."""
    # Independent threshold classifiers can disagree about ordering. Repair it.
    q = np.minimum.accumulate(np.clip(q, 0, 1), axis=1)
    return np.column_stack([1-q[:, 0], q[:, 0]-q[:, 1], q[:, 1]-q[:, 2],
                            q[:, 2]-q[:, 3], q[:, 3]])


def fit_and_predict(a, y, b, text_end, include_nb=True):
    """a = fitting features; b = query features. No query labels are accepted."""
    plain_q = np.empty((b.shape[0], 4), dtype=np.float64)
    nb_q = np.empty_like(plain_q) if include_nb else None
    text = a[:, :text_end] if include_nb else None
    for cutoff in (1, 2, 3, 4):
        high = y > cutoff
        model = SGDClassifier(loss="log_loss", alpha=ALPHA, average=True,
                              max_iter=30, tol=1e-3, random_state=SEED)
        model.fit(a, high.astype(int))
        plain_q[:, cutoff-1] = model.predict_proba(b)[:, 1]
        if include_nb:
            positive = np.asarray(text[high].sum(axis=0)).ravel().astype(np.float64)
            negative = np.asarray(text[~high].sum(axis=0)).ravel().astype(np.float64)
            n = len(positive)
            ratio = np.log((positive+1)/(positive.sum()+n)) - np.log((negative+1)/(negative.sum()+n))
            scale = np.ones(a.shape[1], dtype=np.float32)
            scale[:text_end] = np.clip(ratio, -3, 3).astype(np.float32)
            weighted_a = a.multiply(scale).tocsr()
            model = SGDClassifier(loss="log_loss", alpha=ALPHA, average=True,
                                  max_iter=30, tol=1e-3, random_state=SEED)
            model.fit(weighted_a, high.astype(int))
            del weighted_a
            nb_q[:, cutoff-1] = model.predict_proba(b.multiply(scale).tocsr())[:, 1]
        say(f"    Fitted boundary Score > {cutoff}")
    return ordinal_probs(plain_q), ordinal_probs(nb_q) if include_nb else None


def blend(plain, nb):
    return (1-NB_BLEND) * plain + NB_BLEND * nb


# ==================== 3. DECISION WEIGHTS AND ERROR COSTS ====================
def macro_f1(y, pred):
    """Same 5-class macro-F1 as sklearn; fast enough for many cheap rule tests."""
    counts = np.bincount((np.asarray(y, dtype=int)-1)*5 + np.asarray(pred, dtype=int)-1,
                         minlength=25).reshape(5, 5)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    class_f1 = np.divide(2*np.diag(counts), denominator,
                         out=np.zeros(5, dtype=float), where=denominator != 0)
    return float(class_f1.mean())


def cost_matrix(weights, pairs=(1.0, 1.0)):
    """Rows = ACTUAL rating; columns = PREDICTED rating. Diagonal is zero."""
    cost = np.repeat(np.asarray(weights, dtype=float)[:, None], 5, axis=1)
    np.fill_diagonal(cost, 0)
    for (i, j), multiplier in zip([(1, 2), (2, 3)], pairs):
        cost[i, j] *= multiplier  # Zero-based indices: (1,2) means ratings 2 and 3.
        cost[j, i] *= multiplier
    return cost


def choose_rating(probs, cost):
    expected_cost = probs @ np.asarray(cost, dtype=float)
    return expected_cost.argmin(axis=1) + 1


def tune_weights(probs, y):
    weights = np.ones(5)
    # Fix rating 5's weight to 1: multiplying EVERY cost by a constant changes nothing.
    for _ in range(3):
        changed = False
        for k in range(4):
            best = macro_f1(y, (probs * weights).argmax(axis=1) + 1)
            chosen = weights[k]
            for candidate in WEIGHT_OPTIONS:
                trial = weights.copy()
                trial[k] = candidate
                score = macro_f1(y, (probs * trial).argmax(axis=1) + 1)
                if score > best + 1e-12:
                    best, chosen = score, candidate
            changed |= chosen != weights[k]
            weights[k] = chosen
        if not changed:
            break
    return weights


def tune_costs(probs, y, weights):
    # Only TWO extra parameters, not 20 independently tuned error costs.
    best_pairs = (1.0, 1.0)
    best = macro_f1(y, choose_rating(probs, cost_matrix(weights)))
    for pairs in product(PAIR_OPTIONS, repeat=2):
        score = macro_f1(y, choose_rating(probs, cost_matrix(weights, pairs)))
        if score > best + 1e-12:
            best, best_pairs = score, pairs
    return best_pairs


def fit_rules(plain, nb, y):
    """Called on INNER calibration predictions only during fold evaluation."""
    mixed = blend(plain, nb)
    w_plain = tune_weights(plain, y)
    w_blend = tune_weights(mixed, y)
    pairs = tune_costs(mixed, y, w_blend)
    return {
        "ordinal_weights": cost_matrix(w_plain),
        "blend_weights": cost_matrix(w_blend),
        "blend_costs": cost_matrix(w_blend, pairs),
    }


# ==================== 4. SAFE PROBABILITY CACHING ====================
def run_signature():
    """Reject stale predictions after data/model/fold/code or package changes."""
    digest = hashlib.sha256()
    with (BASE / "train.csv").open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    functions = [read_reviews, prepare, make_features, transform_batches,
                 ordinal_probs, fit_and_predict, make_fold]
    code = "\n".join(inspect.getsource(function) for function in functions)
    return {
        "data_sha256": digest.hexdigest(),
        "training_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "alpha": ALPHA, "folds": N_FOLDS, "calibration_fraction": CALIBRATION_FRACTION,
        "seed": SEED, "id_columns": ID_COLS, "number_columns": NUM_COLS,
        "python": sys.version.split()[0], "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    }


def decision_settings():
    return {"nb_blend": NB_BLEND, "weight_options": WEIGHT_OPTIONS, "pair_options": PAIR_OPTIONS}


def make_fold(X, y, train_idx, eval_idx, fold):
    # The outer held-out fold is excluded from BOTH fitting and calibration.
    fit_idx, cal_idx = train_test_split(train_idx, test_size=CALIBRATION_FRACTION,
                                       stratify=y[train_idx], random_state=SEED+fold)
    say(f"  Fitting: {len(fit_idx):,}; calibration: {len(cal_idx):,}; evaluation: {len(eval_idx):,}")
    if len(np.unique(y[fit_idx])) != 5 or len(np.unique(y[cal_idx])) != 5:
        raise ValueError("Each fitting/calibration partition needs all five ratings.")
    features = make_features()
    started = perf_counter()
    a = features.fit_transform(X.iloc[fit_idx], y[fit_idx]).tocsr()
    query_idx = np.r_[cal_idx, eval_idx]
    b = transform_batches(features, X.iloc[query_idx])
    say(f"  Features: {a.shape[1]:,}; preparation: {perf_counter()-started:.1f}s")
    plain, nb = fit_and_predict(a, y[fit_idx], b, features.output_indices_["ids"].start)
    n_cal = len(cal_idx)
    return dict(fit_idx=fit_idx, cal_idx=cal_idx, eval_idx=eval_idx,
                y_cal=y[cal_idx], y_eval=y[eval_idx],
                plain_cal=plain[:n_cal], nb_cal=nb[:n_cal],
                plain_eval=plain[n_cal:], nb_eval=nb[n_cal:])


def check_fold(data, n):
    # Also detects incomplete/corrupt caches rather than silently trusting them.
    indices = np.r_[data["fit_idx"], data["cal_idx"], data["eval_idx"]]
    if not np.array_equal(np.sort(indices), np.arange(n)):
        raise ValueError("Fold cache has overlapping or missing row indices. Use a fresh RUN_NAME.")
    for part in ["cal", "eval"]:
        if len(data[f"y_{part}"]) != len(data[f"{part}_idx"]):
            raise ValueError("Invalid cached labels.")
        for name in ["plain", "nb"]:
            p = data[f"{name}_{part}"]
            if (p.shape != (len(data[f"y_{part}"]), 5) or not np.isfinite(p).all()
                    or (p < 0).any() or not np.allclose(p.sum(axis=1), 1)):
                raise ValueError("Invalid cached probabilities. Use a fresh RUN_NAME.")


# ==================== 5. COMPARE ALL METHODS ON ALL FOLDS ====================
def run_comparison(out, signature):
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("Data, model, folds, code or package versions changed. Choose a new RUN_NAME.")
    else:
        if list(cache.glob("fold_*.npz")):
            raise ValueError("Cache files have no manifest. Choose a new RUN_NAME for a safe rerun.")
        if MODE == "retune":
            raise FileNotFoundError("No probability cache. Run MODE='cv' first.")
        manifest = None

    paths = [cache / f"fold_{fold}.npz" for fold in range(1, N_FOLDS+1)]
    needs_training = any(not path.exists() for path in paths)
    if MODE == "retune" and needs_training:
        raise FileNotFoundError("Not all folds are cached. Finish MODE='cv' first.")
    X = y = splits = None
    if needs_training or manifest is None:
        raw = read_reviews()
        known = raw.loc[raw["Score"].notna()].reset_index(drop=True)
        X, y = known.drop(columns="Score"), known["Score"].astype(int).to_numpy()
        if np.bincount(y, minlength=6)[1:].min() < N_FOLDS * 5:
            raise ValueError("Too few reviews of at least one rating for these folds.")
        del raw, known
        splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(np.zeros(len(y)), y))
        if manifest is None:
            manifest = {"signature": signature, "n_labeled": len(y)}
            write_json(manifest_path, manifest)
    n = manifest["n_labeled"]
    y_oof = np.zeros(n, dtype=int)
    plain_oof = np.empty((n, 5), dtype=np.float64)
    nb_oof = np.empty_like(plain_oof)
    predictions = {name: np.zeros(n, dtype=int) for name in APPROACHES}
    seen = np.zeros(n, dtype=bool)
    rows, rules_by_fold = [], {}

    for fold, path in enumerate(paths, start=1):
        say(f"\nFOLD {fold}/{N_FOLDS}")
        if path.exists():
            say("  Loading saved probabilities; no models are retrained.")
            with np.load(path, allow_pickle=False) as stored:
                data = {key: stored[key] for key in stored.files}
        else:
            data = make_fold(X, y, *splits[fold-1], fold)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **data)
            temporary.replace(path)
            gc.collect()
        check_fold(data, n)
        idx = data["eval_idx"]
        if seen[idx].any():
            raise ValueError("Evaluation rows overlap between folds.")
        seen[idx] = True
        y_oof[idx] = data["y_eval"]
        plain_oof[idx], nb_oof[idx] = data["plain_eval"], data["nb_eval"]
        say("  Choosing weights/costs on calibration rows only...")
        rules = fit_rules(data["plain_cal"], data["nb_cal"], data["y_cal"])
        rules_by_fold[str(fold)] = {key: value.tolist() for key, value in rules.items()}
        mixed = blend(data["plain_eval"], data["nb_eval"])
        for name in APPROACHES:
            probs = data["plain_eval"] if name == "ordinal_weights" else mixed
            pred = choose_rating(probs, rules[name])
            predictions[name][idx] = pred
            score = macro_f1(data["y_eval"], pred)
            rows.append({"approach": name, "fold": fold, "macro_f1": score})
            say(f"  {name:18s} evaluation Macro-F1: {score:.5f}")
    if not seen.all():
        raise ValueError("Not every labeled row received one out-of-fold prediction.")
    del X, y, data
    gc.collect()

    summary_rows = []
    for name in APPROACHES:
        scores = [row["macro_f1"] for row in rows if row["approach"] == name]
        class_f1 = f1_score(y_oof, predictions[name], labels=LABELS, average=None, zero_division=0)
        summary_rows.append({"approach": name,
                             **{f"fold_{i+1}": score for i, score in enumerate(scores)},
                             "mean_F1": np.mean(scores), "fold_SD": np.std(scores, ddof=1),
                             "pooled_F1": macro_f1(y_oof, predictions[name]),
                             "F1_2": class_f1[1], "F1_3": class_f1[2]})
    table = pd.DataFrame(summary_rows).sort_values("mean_F1", ascending=False, kind="stable")
    winner = table.iloc[0]["approach"]  # Best mean, NOT best individual fold.
    report = classification_report(y_oof, predictions[winner], labels=LABELS, digits=4, zero_division=0)
    warning = ("These are model-selection scores, not a fresh final test. Fold SD is descriptive, "
               "not a confidence interval. Do not compare directly with the old single-split score.")
    say("\nCROSS-VALIDATION COMPARISON\n" + table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    say(f"\nSelected approach: {winner}\n\nPooled held-out classification report:\n{report}")
    say(warning)
    table.to_csv(out / "cv_summary.csv", index=False)
    write_json(out / "fold_cost_matrices.json", rules_by_fold)
    (out / "results.txt").write_text(table.to_string(index=False) + "\n\n" + report + "\n" + warning, encoding="utf-8")
    save_plots(out, y_oof, predictions, winner)

    # After evaluation is complete, use ALL raw OOF predictions to fit ONE final
    # decision rule. Each row's base-model probabilities excluded that row.
    # This is final tuning, not another evaluation; do NOT report its fit score.
    say("\nFitting the final decision rule on pooled out-of-fold probabilities...")
    probs = plain_oof if winner == "ordinal_weights" else blend(plain_oof, nb_oof)
    weights = tune_weights(probs, y_oof)
    pairs = tune_costs(probs, y_oof, weights) if winner == "blend_costs" else (1.0, 1.0)
    selected = {"signature": signature, "decision_settings": decision_settings(),
                "approach": winner, "mean_cv_f1": float(table.iloc[0]["mean_F1"]),
                "weights": weights.tolist(), "pair_multipliers": list(pairs),
                "cost_matrix": cost_matrix(weights, pairs).tolist(),
                "note": "Final rule fitted on pooled OOF labels; its training score is NOT a test score."}
    write_json(out / "selected.json", selected)
    say(f"Final weights: {np.round(weights, 3).tolist()}; pair multipliers: {list(pairs)}")
    say(f"Results saved in {out}\nNo Kaggle submission was created.")


def save_plots(out, y, predictions, winner):
    plt.figure(figsize=(7, 4))
    for name in APPROACHES:
        values = f1_score(y, predictions[name], labels=LABELS, average=None, zero_division=0)
        plt.plot(LABELS, values, "o-", label=name)
    plt.xticks(LABELS)
    plt.ylim(0, 1)
    plt.xlabel("Rating")
    plt.ylabel("Pooled held-out F1")
    plt.title("Cross-validation: F1 by rating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "class_f1.png", dpi=150)
    plt.close()
    display = ConfusionMatrixDisplay.from_predictions(y, predictions[winner], labels=LABELS,
                                                      normalize="true", values_format=".2f")
    display.ax_.set_title("Selected approach: held-out predictions")
    display.figure_.tight_layout()
    display.figure_.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)


# ==================== 6. FINAL MODEL AND NAMED SUBMISSION ====================
def submit(out, signature):
    selected_path = out / "selected.json"
    if not selected_path.exists():
        raise FileNotFoundError("Run MODE='cv' first to select an approach and save its rule.")
    chosen = json.loads(selected_path.read_text(encoding="utf-8"))
    if chosen["signature"] != signature:
        raise ValueError("Data/model settings changed. Run CV with a new RUN_NAME first.")
    if chosen["decision_settings"] != decision_settings():
        raise ValueError("Decision settings changed. Run MODE='retune' first.")
    if (Path(SUBMISSION_NAME).name != SUBMISSION_NAME or not SUBMISSION_NAME.lower().endswith(".csv")
            or SUBMISSION_NAME.lower() in {"train.csv", "test.csv", "sample_submission.csv"}):
        raise ValueError("Choose a .csv filename that is not an input filename.")
    target = BASE / SUBMISSION_NAME
    if target.exists():
        raise FileExistsError(f"{SUBMISSION_NAME} exists. Choose a new submission name.")

    raw = read_reviews()
    known = raw.loc[raw["Score"].notna()]
    unknown = raw.loc[raw["Score"].isna()].set_index("id", drop=False)
    test_ids = pd.read_csv(BASE / "test.csv", dtype={"id": str})["id"]
    if test_ids.empty or test_ids.isna().any() or test_ids.duplicated().any():
        raise ValueError("test.csv IDs must be nonmissing, unique and nonempty.")
    if set(test_ids) != set(unknown.index):
        raise ValueError("test.csv IDs must exactly match missing-score rows in train.csv.")
    query = unknown.loc[test_ids].drop(columns="Score")  # Preserve test.csv order.
    X, y = known.drop(columns="Score"), known["Score"].astype(int).to_numpy()
    say(f"Selected approach: {chosen['approach']}; training on ALL {len(y):,} known ratings.")
    say("Refitting the vocabulary, scaling and classifiers on all labeled reviews...")
    features = make_features()
    a = features.fit_transform(X, y).tocsr()
    b = transform_batches(features, query)
    include_nb = chosen["approach"] != "ordinal_weights"
    plain, nb = fit_and_predict(a, y, b, features.output_indices_["ids"].start, include_nb)
    probs = blend(plain, nb) if include_nb else plain
    predictions = choose_rating(probs, chosen["cost_matrix"])
    pd.DataFrame({"id": test_ids.to_numpy(), "Score": predictions}).to_csv(target, index=False)
    write_json(target.with_suffix(".settings.json"), chosen)
    say(f"Saved {len(predictions):,} predictions in test.csv order: {target}")
    say("Upload the CSV manually; commit the corresponding code, not data/CSV/cache files.")


def main():
    start = perf_counter()
    if MODE not in {"cv", "retune", "submit"}:
        raise ValueError("MODE must be 'cv', 'retune', or 'submit'.")
    if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
        raise ValueError("RUN_NAME must be a single folder name.")
    if N_FOLDS < 2 or not 0 < CALIBRATION_FRACTION < 1 or ALPHA <= 0 or not 0 <= NB_BLEND <= 1:
        raise ValueError("Invalid folds, calibration fraction, alpha or blend fraction.")
    for options in [WEIGHT_OPTIONS, PAIR_OPTIONS]:
        if not options or any(not np.isfinite(x) or x <= 0 for x in options):
            raise ValueError("Cost/weight options must contain positive finite numbers.")
    out = BASE / "runs" / RUN_NAME
    out.mkdir(parents=True, exist_ok=True)
    say(f"Mode={MODE}; folds={N_FOLDS}; NB blend={NB_BLEND:.2f}")
    say("Checking the data fingerprint before using any saved probabilities...")
    signature = run_signature()
    if MODE == "submit":
        submit(out, signature)
    else:
        run_comparison(out, signature)
    say(f"Finished in {perf_counter()-start:.1f}s")


if __name__ == "__main__":
    main()
