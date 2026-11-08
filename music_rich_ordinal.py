"""Music ratings: richer text + ordinal/NB models + jointly tuned error costs.

python music_rich_ordinal.py
MODE='validate': compare baseline/candidate on the same fitting/tuning/audit split.
MODE='submit': refit the tuning-selected configuration on all known ratings.
No neural networks, boosting, external data, or automatic Kaggle uploads.

References (explain the methods and disclose AI assistance in your README):
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
https://aclanthology.org/P12-2018/  (NB-weighting inspiration, not exact NB-SVM)
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
"""
from pathlib import Path
from itertools import product
from time import perf_counter
import gc
import hashlib
import inspect
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import differential_evolution
from scipy.special import expit
from scipy.sparse import vstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.sparsefuncs import inplace_csr_column_scale

# ------------------------- SETTINGS -------------------------
MODE = "validate"                 # "validate" or "submit"
RUN_NAME = "rich_ordinal_v1"       # Keeps the earlier model's files untouched.
SUBMISSION_NAME = "rich_ordinal_v1.csv"
FEATURE_SET = "rich"              # "rich" or "standard" for an ablation.
ALPHA = 3e-6                      # Candidate regularization; baseline uses 1e-5.
NB_BLEND = 0.40
SEED = 42
# ------------------------------------------------------------
BASE = Path(__file__).resolve().parent
OUT = BASE / "runs" / RUN_NAME
LABELS = [1, 2, 3, 4, 5]
IDS = ["reviewerID", "album_mbid", "artist_mbid"]
NUMBERS = ["helpful_share", "log_votes", "log_length", "year"]
WEIGHTS = [0.4, 0.55, 0.7, 0.85, 1., 1.2, 1.5, 1.8, 2.2, 2.8, 3.5]
PAIRS = [0.75, 1., 1.25, 1.5]


def say(message):
    print(message, flush=True)


def read_data():
    say("Reading train.csv...")
    d = pd.read_csv(BASE / "train.csv", dtype=str, low_memory=False)
    required = ["id", "Score", "summary", "reviewText", "genres", "TotalVotes",
                "VotedHelpful", "unixReviewTime"] + IDS
    missing = sorted(set(required) - set(d.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if d.id.isna().any() or d.id.str.strip().eq("").any() or d.id.duplicated().any():
        raise ValueError("Review IDs must be unique and nonblank.")
    score = pd.to_numeric(d.Score, errors="coerce")
    nonblank = d.Score.fillna("").str.strip().ne("")
    if (nonblank & score.isna()).any() or not score.dropna().isin(LABELS).all():
        raise ValueError("Scores must be blank or integers 1 through 5.")
    d["Score"] = score
    for col in ["summary", "reviewText", "genres"] + IDS:
        d[col] = d[col].fillna("").astype(str)
    d["genres"] = d.genres.str.replace(r"\s*,\s*", ",", regex=True)
    votes = pd.to_numeric(d.TotalVotes, errors="coerce").fillna(0).clip(lower=0)
    helpful = pd.to_numeric(d.VotedHelpful, errors="coerce").fillna(0).clip(lower=0)
    d["helpful_share"] = (helpful / votes.replace(0, np.nan)).fillna(0).clip(0, 1)
    d["log_votes"] = np.log1p(votes)
    d["log_length"] = np.log1p(d.reviewText.str.len())
    d["year"] = pd.to_datetime(pd.to_numeric(d.unixReviewTime, errors="coerce"),
                                unit="s", errors="coerce").dt.year.fillna(2010)
    d[NUMBERS] = d[NUMBERS].astype(np.float32)
    return d


def make_features(kind):
    words = dict(sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
    rich = kind == "rich"
    blocks = [
        ("review", TfidfVectorizer(min_df=3, ngram_range=(1, 2),
                                  max_features=250000 if rich else 80000, **words), "reviewText"),
        ("summary", TfidfVectorizer(min_df=2, ngram_range=(1, 3) if rich else (1, 2),
                                   max_features=40000 if rich else 15000, **words), "summary"),
    ]
    if rich:
        blocks.append(("chars", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                        min_df=3, max_features=40000, sublinear_tf=True,
                        dtype=np.float32), "summary"))
    blocks += [
        ("genres", TfidfVectorizer(token_pattern=r"[^,]+", dtype=np.float32), "genres"),
        ("ids", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), IDS),
        ("numbers", StandardScaler(), NUMBERS),
    ]
    return ColumnTransformer(blocks, sparse_threshold=1.,
                             transformer_weights={"chars": 0.5} if rich else None)


def transform_batches(features, frame):
    return vstack([features.transform(frame.iloc[i:i+20000])
                   for i in range(0, len(frame), 20000)], format="csr")


def ordinal_probs(q):
    # Repair crossing cumulative probabilities before taking differences.
    q = np.minimum.accumulate(np.clip(q, 0, 1), axis=1)
    return np.column_stack([1-q[:, 0], q[:, 0]-q[:, 1], q[:, 1]-q[:, 2],
                            q[:, 2]-q[:, 3], q[:, 3]])


def fit_predict(a, y, b, text_end, alpha, max_iter, tol):
    """Four ordinary and four NB-weighted logistic models. Query labels are absent."""
    plain, nb = np.zeros((b.shape[0], 4)), np.zeros((b.shape[0], 4))
    # Matrix-vector sums avoid copying a large text matrix for each class.
    total = np.asarray(a.sum(axis=0)).ravel()[:text_end].astype(float)
    for k in range(1, 5):
        target = (np.asarray(y) > k).astype(int)
        args = dict(loss="log_loss", alpha=alpha, average=True,
                    max_iter=max_iter, tol=tol, random_state=SEED)
        model = SGDClassifier(**args).fit(a, target)
        plain[:, k-1] = model.predict_proba(b)[:, 1]
        pos = np.asarray(a.T @ target.astype(np.float32)).ravel()[:text_end].astype(float)
        neg = np.maximum(total-pos, 0)
        ratio = np.log((pos+1)/(pos.sum()+text_end)) - np.log((neg+1)/(neg.sum()+text_end))
        scale = np.ones(a.shape[1], dtype=np.float32)
        scale[:text_end] = np.clip(ratio, -3, 3)
        weighted = a.copy()
        inplace_csr_column_scale(weighted, scale)
        weighted_model = SGDClassifier(**args).fit(weighted, target)
        # Scaling coefficients gives the same query scores without copying b.
        nb[:, k-1] = expit(b @ (weighted_model.coef_.ravel()*scale) + weighted_model.intercept_[0])
        say(f"  Boundary > {k}; epochs {model.n_iter_}/{weighted_model.n_iter_}")
        del weighted, model, weighted_model
        gc.collect()
    return (1-NB_BLEND)*ordinal_probs(plain) + NB_BLEND*ordinal_probs(nb)


def macro(y, pred):
    cm = np.bincount((np.asarray(y, int)-1)*5 + np.asarray(pred, int)-1,
                     minlength=25).reshape(5, 5)
    den = cm.sum(0) + cm.sum(1)
    return np.divide(2*np.diag(cm), den, out=np.zeros(5), where=den != 0).mean()


def cost_matrix(weights, pairs=(1., 1.)):
    cost = np.repeat(np.asarray(weights)[:, None], 5, axis=1).astype(float)
    np.fill_diagonal(cost, 0)
    for (i, j), factor in zip([(1, 2), (2, 3)], pairs):
        cost[i, j] *= factor
        cost[j, i] *= factor
    return cost


def decide(p, cost):
    return (p @ cost).argmin(1) + 1


def old_rule(p, y):
    """Previous search, retained so the comparison includes a real control."""
    w = np.ones(5)
    for _ in range(3):
        changed = False
        for k in range(4):
            best, chosen = macro(y, (p*w).argmax(1)+1), w[k]
            for option in WEIGHTS:
                trial = w.copy(); trial[k] = option
                score = macro(y, (p*trial).argmax(1)+1)
                if score > best + 1e-12:
                    best, chosen = score, option
            changed |= chosen != w[k]
            w[k] = chosen
        if not changed:
            break
    pairs = (1., 1.)
    best = macro(y, decide(p, cost_matrix(w)))
    for trial in product(PAIRS, repeat=2):
        score = macro(y, decide(p, cost_matrix(w, trial)))
        if score > best + 1e-12:
            best, pairs = score, trial
    return cost_matrix(w, pairs), w, np.array(pairs)


def joint_rule(p, y):
    """Search the SAME six cost parameters together; no new training examples/models."""
    old_cost, weights, pairs = old_rule(p, y)
    def unpack(z):
        return np.r_[np.exp(z[:4]), 1.], np.exp(z[4:])
    def objective(z):
        w, pair = unpack(z)
        return -macro(y, decide(p, cost_matrix(w, pair)))
    initial = np.log(np.r_[weights[:4], pairs])
    result = differential_evolution(
        objective, [(np.log(.3), np.log(4.))]*4 + [(np.log(.25), np.log(1.5))]*2,
        x0=initial, seed=23, maxiter=35, popsize=8, tol=0., atol=0.,
        polish=False, workers=1)
    w, pair = unpack(result.x)
    new_cost = cost_matrix(w, pair)
    if macro(y, decide(p, new_cost)) < macro(y, decide(p, old_cost)):
        return old_cost, weights, pairs
    return new_cost, w, pair


def signature():
    digest = hashlib.sha256()
    with (BASE / "train.csv").open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    source = "\n".join(inspect.getsource(f) for f in
                        [read_data, make_features, transform_batches, ordinal_probs, fit_predict])
    return {"data_sha256": digest.hexdigest(), "training_code": hashlib.sha256(source.encode()).hexdigest(),
            "feature_set": FEATURE_SET, "alpha": ALPHA, "nb_blend": NB_BLEND, "seed": SEED,
            "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


def validate(sig):
    cache, manifest = OUT / "probabilities.npz", OUT / "signature.json"
    if cache.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != sig:
            raise ValueError("Data/model changed. Choose a new RUN_NAME for this experiment.")
        with np.load(cache, allow_pickle=False) as z:
            yv, tuning, audit = z["y"], z["tuning"], z["audit"]
            base_p, new_p = z["baseline"], z["candidate"]
        say("Loaded saved probabilities; no classifiers are retrained.")
    else:
        raw = read_data()
        d = raw.loc[raw.Score.notna()].reset_index(drop=True)
        del raw; gc.collect()
        y = d.Score.astype(int).to_numpy()
        fit, valid = train_test_split(np.arange(len(y)), test_size=.2, random_state=SEED, stratify=y)
        yv = y[valid]
        tuning, audit = train_test_split(np.arange(len(valid)), test_size=.5, random_state=17, stratify=yv)
        say(f"Fit: {len(fit):,}; tuning: {len(tuning):,}; audit: {len(audit):,}")
        probabilities = []
        for label, kind, alpha, epochs, tol in [
                ("Previous baseline", "standard", 1e-5, 30, 1e-3),
                ("Candidate", FEATURE_SET, ALPHA, 50, 1e-4)]:
            say(f"\n{label}: building {kind} features...")
            features = make_features(kind)
            a = features.fit_transform(d.iloc[fit], y[fit]).tocsr()
            b = transform_batches(features, d.iloc[valid])
            say(f"Features: {a.shape[1]:,}")
            end = features.output_indices_["ids"].start
            probabilities.append(fit_predict(a, y[fit], b, end, alpha, epochs, tol))
            del a, b, features; gc.collect()
        base_p, new_p = probabilities
        np.savez_compressed(cache, y=yv, tuning=tuning, audit=audit, baseline=base_p, candidate=new_p)
        manifest.write_text(json.dumps(sig, indent=2))
    # These choices use tuning rows ONLY. All choices are frozen before audit reporting.
    candidates = []
    for name, p, kind, search in [
            ("baseline", base_p, "baseline", old_rule),
            ("candidate_grid", new_p, "candidate", old_rule),
            ("candidate_joint", new_p, "candidate", joint_rule)]:
        cost, w, pairs = search(p[tuning], yv[tuning])
        score = float(macro(yv[tuning], decide(p[tuning], cost)))
        candidates.append(dict(name=name, model=kind, tuning_F1=score, cost_matrix=cost.tolist(),
                               weights=w.tolist(), pair_multipliers=pairs.tolist()))
    selected = max(candidates, key=lambda r: r["tuning_F1"])
    say(f"\nSelected on tuning rows: {selected['name']}")
    table = []
    for row in candidates:
        p = base_p if row["model"] == "baseline" else new_p
        row["audit_F1"] = float(macro(yv[audit], decide(p[audit], np.array(row["cost_matrix"]))))
        table.append({k: row[k] for k in ["name", "tuning_F1", "audit_F1"]})
    table = pd.DataFrame(table)
    say("\nMATCHED COMPARISON\n" + table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    p = base_p if selected["model"] == "baseline" else new_p
    pred = decide(p[audit], np.array(selected["cost_matrix"]))
    report = classification_report(yv[audit], pred, labels=LABELS, digits=4, zero_division=0)
    say("\n" + report)
    selected["signature"] = sig
    (OUT / "selected.json").write_text(json.dumps(selected, indent=2))
    note = ("Audit labels were not used to choose settings in this run. This is not a pristine "
            "project-level test: earlier experiments have used these reviews. Kaggle may differ.")
    (OUT / "results.txt").write_text(table.to_string(index=False) + "\n\n" + report + "\n" + note)
    table.to_csv(OUT / "comparison.csv", index=False)
    plt.figure(figsize=(7, 4))
    base_pred = decide(base_p[audit], np.array(candidates[0]["cost_matrix"]))
    for label, predictions in [("Previous baseline", base_pred), (selected["name"], pred)]:
        plt.plot(LABELS, f1_score(yv[audit], predictions, labels=LABELS, average=None,
                                zero_division=0), "o-", label=label)
    plt.xticks(LABELS); plt.ylim(0, 1); plt.xlabel("Rating"); plt.ylabel("Audit F1")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "class_f1.png", dpi=150); plt.close()
    display = ConfusionMatrixDisplay.from_predictions(yv[audit], pred, labels=LABELS,
                                                       normalize="true", values_format=".2f")
    display.figure_.tight_layout(); display.figure_.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)
    say(note + f"\nResults: {OUT}\nNo submission created.")


def submit(sig):
    rule_file = OUT / "selected.json"
    if not rule_file.exists():
        raise FileNotFoundError("Run MODE='validate' first, using the same RUN_NAME.")
    chosen = json.loads(rule_file.read_text())
    if chosen["signature"] != sig:
        raise ValueError("Model/data settings differ from validation. Validate the new configuration first.")
    name = Path(SUBMISSION_NAME)
    if name.name != SUBMISSION_NAME or name.suffix.lower() != ".csv" or name.name.lower() in {
            "train.csv", "test.csv", "sample_submission.csv"}:
        raise ValueError("Choose a plain .csv output filename, not an input filename.")
    output = BASE / name
    if output.exists():
        raise FileExistsError("Submission already exists. Change SUBMISSION_NAME.")
    raw = read_data()
    known = raw.loc[raw.Score.notna()]
    unknown = raw.loc[raw.Score.isna()].set_index("id", drop=False)
    ids = pd.read_csv(BASE / "test.csv", dtype=str)["id"]
    if ids.empty or ids.isna().any() or ids.duplicated().any() or set(ids) != set(unknown.index):
        raise ValueError("test.csv must contain exactly the unique missing-score review IDs.")
    is_base = chosen["model"] == "baseline"
    features = make_features("standard" if is_base else FEATURE_SET)
    y = known.Score.astype(int).to_numpy()
    say(f"Refitting {chosen['name']} on all {len(y):,} labeled reviews...")
    a = features.fit_transform(known, y).tocsr()
    b = transform_batches(features, unknown.loc[ids])
    p = fit_predict(a, y, b, features.output_indices_["ids"].start,
                    1e-5 if is_base else ALPHA, 30 if is_base else 50, 1e-3 if is_base else 1e-4)
    pred = decide(p, np.array(chosen["cost_matrix"]))
    pd.DataFrame({"id": ids.to_numpy(), "Score": pred}).to_csv(output, index=False)
    output.with_suffix(".settings.json").write_text(json.dumps(chosen, indent=2))
    say(f"Saved {len(pred):,} predictions in test.csv order: {output}")
    say("Upload manually. Commit source code and settings, not CSV/data/probability caches.")


def main():
    started = perf_counter()
    if MODE not in {"validate", "submit"} or FEATURE_SET not in {"standard", "rich"}:
        raise ValueError("Invalid MODE or FEATURE_SET.")
    if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
        raise ValueError("RUN_NAME must be a single folder name.")
    if ALPHA <= 0 or not 0 <= NB_BLEND <= 1:
        raise ValueError("ALPHA must be positive and NB_BLEND between 0 and 1.")
    OUT.mkdir(parents=True, exist_ok=True)
    say(f"Mode={MODE}; features={FEATURE_SET}; alpha={ALPHA}; NB blend={NB_BLEND}")
    sig = signature()
    validate(sig) if MODE == "validate" else submit(sig)
    say(f"Finished in {perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
