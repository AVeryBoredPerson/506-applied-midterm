"""Five-fold check for music_rich_ordinal.py (classical ML only).

Put this file NEXT TO music_rich_ordinal.py, train.csv and test.csv.

  MODE='cv'       : five held-out folds, each with its own INNER tuning split.
  MODE='submit'   : use selected mean-CV winner, fit on all known scores,
                    predict unknown scores in the exact order of test.csv.

No neural nets, no boosting; no automatic Kaggle upload.
The three approaches are the SAME as in music_rich_ordinal.py:
  baseline        : standard features, old costs
  candidate_grid  : rich features, old costs
  candidate_joint : rich features, jointly tuned costs

Each fold saves its baseline/rich probabilities separately. Re-running an
unchanged CV uses cached predictions; an interrupted CV resumes missing work.
The audit F1 values (outer folds) NEVER influence each fold's tuned costs.

References: see the docstring of music_rich_ordinal.py. Explain fold splitting,
model selection, joint cost tuning, and AI assistance in the final README.
"""

from pathlib import Path
import gc
import hashlib
import inspect
import json
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import music_rich_ordinal as rich

# ========================= SETTINGS =========================
MODE = "cv"                         # "cv" or "submit"
RUN_NAME = "rich_ordinal_5fold_alpha2_blend3"     # New RUN_NAME if model, folds or data changes
N_FOLDS = 5
INNER_TUNE_FRACTION = 0.125       # Of each outer-training partition
ALPHA = 2e-6
NB_BLEND = 0.30
SEED = 42
SUBMISSION_NAME = "rich_ordinal_5fold_alpha2.csv"
# ============================================================

BASE = Path(__file__).resolve().parent
OUT = BASE / "runs" / RUN_NAME
CACHE = OUT / "cache"
NAMES = ("baseline", "candidate_grid", "candidate_joint")

# Preserve one source of truth: the original file owns feature engineering
# and the classifiers. This file only adds careful cross-validation.
rich.ALPHA = ALPHA
rich.NB_BLEND = NB_BLEND
rich.SEED = SEED


def tell(message):
    print(message, flush=True)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def signature():
    # Include relevant model and fold logic, but not MODE/SUBMISSION_NAME.
    sha = hashlib.sha256()
    with (BASE / "train.csv").open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    sources = [rich.read_data, rich.make_features, rich.transform_batches,
               rich.ordinal_probs, rich.fit_predict, rich.old_rule,
               rich.joint_rule, rich.cost_matrix, rich.decide,
               fit_fold_model, split_fold, tune_fold]
    source_hash = hashlib.sha256(
        "\n".join(inspect.getsource(func) for func in sources).encode()
    ).hexdigest()
    return dict(data_sha256=sha.hexdigest(), code_sha256=source_hash,
                folds=N_FOLDS, inner_tune=INNER_TUNE_FRACTION,
                alpha=ALPHA, nb_blend=NB_BLEND, seed=SEED,
                sklearn=sklearn.__version__, numpy=np.__version__,
                pandas=pd.__version__, python=sys.version.split()[0])


def split_fold(y, outer_train, outer_eval, fold_number):
    fit_idx, tune_idx = train_test_split(
        outer_train, test_size=INNER_TUNE_FRACTION,
        stratify=y[outer_train], random_state=SEED + fold_number)
    return fit_idx, tune_idx, outer_eval


def fit_fold_model(data, y, fit_idx, tune_idx, eval_idx, model_name):
    """Fit preprocessing and model on fit_idx only. No tune/eval labels seen."""
    is_baseline = model_name == "baseline"
    feature_kind = "standard" if is_baseline else "rich"
    alpha = 1e-5 if is_baseline else ALPHA
    epochs = 30 if is_baseline else 50
    tol = 1e-3 if is_baseline else 1e-4
    tell(f"  Building {feature_kind} features...")
    features = rich.make_features(feature_kind)
    a = features.fit_transform(data.iloc[fit_idx], y[fit_idx]).tocsr()
    query_idx = np.r_[tune_idx, eval_idx]
    b = rich.transform_batches(features, data.iloc[query_idx]).tocsr()
    tell(f"  Features: {a.shape[1]:,}; fitting ordinal and NB classifiers")
    p = rich.fit_predict(a, y[fit_idx], b,
                         features.output_indices_["ids"].start,
                         alpha, epochs, tol)
    n_tune = len(tune_idx)
    result = dict(tune_p=p[:n_tune], eval_p=p[n_tune:],
                  tune_y=y[tune_idx], eval_y=y[eval_idx],
                  eval_idx=eval_idx)
    del features, a, b, p
    gc.collect()
    return result


def tune_fold(base_p, rich_p, tune_y):
    """Tune independently for this fold. Evaluation labels are NOT arguments."""
    results = {}
    for name, p, tuner in [
        ("baseline", base_p, rich.old_rule),
        ("candidate_grid", rich_p, rich.old_rule),
        ("candidate_joint", rich_p, rich.joint_rule),
    ]:
        cost, w, pairs = tuner(p, tune_y)
        results[name] = dict(cost_matrix=cost, weights=w, pairs=pairs)
    return results


def save_npz(path, result):
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def validate_cache(obj, eval_idx, expected_model):
    if not np.array_equal(obj["eval_idx"], eval_idx):
        raise ValueError(f"Cached {expected_model} fold indices differ; use a new RUN_NAME")
    if obj["eval_p"].shape != (len(eval_idx), 5):
        raise ValueError(f"Corrupt {expected_model} evaluation probabilities")
    if (not np.isfinite(obj["eval_p"]).all() or
            not np.allclose(obj["eval_p"].sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError(f"Invalid {expected_model} evaluation probabilities")


def cross_validate(sig):
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_file = CACHE / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest != sig:
            raise ValueError("Cached files are from another experiment. Change RUN_NAME.")
    else:
        if list(CACHE.glob("fold_*.npz")):
            raise ValueError("Found caches without a manifest. Change RUN_NAME.")
        save_json(manifest_file, sig)

    # We need IDs/labels to reproduce the folds. Each model sees only its fit rows.
    raw = rich.read_data()
    data = raw.loc[raw.Score.notna()].reset_index(drop=True)
    del raw
    y = data.Score.astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = {name: np.zeros(len(y), dtype=np.int8) for name in NAMES}
    evaluated = np.zeros(len(y), dtype=bool)
    fold_scores = {name: [] for name in NAMES}
    # Inner-tuning probabilities are pooled ONLY to choose final submission costs.
    # They are not scored as held-out validation results.
    all_tune = {"baseline": [], "rich": [], "y": []}

    for fold, (outer_train, outer_eval) in enumerate(cv.split(np.zeros(len(y)), y), 1):
        fit_idx, tune_idx, eval_idx = split_fold(y, outer_train, outer_eval, fold)
        tell(f"\nFOLD {fold}/{N_FOLDS}: fit {len(fit_idx):,}, "
             f"tune {len(tune_idx):,}, held-out {len(eval_idx):,}")
        model_results = {}
        for model_name in ("baseline", "rich"):
            path = CACHE / f"fold_{fold:02d}_{model_name}.npz"
            if path.exists():
                tell(f"  Reusing cached {model_name} probabilities")
                result = load_npz(path)
            else:
                result = fit_fold_model(data, y, fit_idx, tune_idx, eval_idx, model_name)
                save_npz(path, result)
                tell(f"  Saved {model_name} probabilities")
            validate_cache(result, eval_idx, model_name)
            if (not np.array_equal(result["tune_y"], y[tune_idx]) or
                    not np.array_equal(result["eval_y"], y[eval_idx]) or
                    len(result["tune_p"]) != len(tune_idx)):
                raise ValueError("Cached labels/split differ. Use a new RUN_NAME.")
            model_results[model_name] = result

        base = model_results["baseline"]
        candidate = model_results["rich"]
        costs = tune_fold(base["tune_p"], candidate["tune_p"], y[tune_idx])
        for name in NAMES:
            p = base["eval_p"] if name == "baseline" else candidate["eval_p"]
            preds = rich.decide(p, costs[name]["cost_matrix"])
            score = float(rich.macro(y[eval_idx], preds))
            fold_scores[name].append(score)
            oof[name][eval_idx] = preds
            tell(f"  {name:17s} held-out Macro-F1: {score:.5f}")
        if evaluated[eval_idx].any():
            raise ValueError("Overlapping outer evaluation folds")
        evaluated[eval_idx] = True
        all_tune["baseline"].append(base["tune_p"])
        all_tune["rich"].append(candidate["tune_p"])
        all_tune["y"].append(y[tune_idx])
        del model_results, costs, base, candidate
        gc.collect()

    if not evaluated.all():
        raise ValueError("Some labeled reviews were never evaluated")
    rows = []
    for name in NAMES:
        scores = np.array(fold_scores[name])
        by_class = f1_score(y, oof[name], labels=rich.LABELS,
                            average=None, zero_division=0)
        rows.append({"model": name,
                     **{f"fold_{i}": float(s) for i, s in enumerate(scores, 1)},
                     "mean_F1": float(scores.mean()),
                     "fold_SD": float(scores.std(ddof=1)),
                     "pooled_F1": float(rich.macro(y, oof[name])),
                     **{f"F1_{i}": float(s) for i, s in enumerate(by_class, 1)}})
    comparison = pd.DataFrame(rows).sort_values("mean_F1", ascending=False)
    tell("\nFIVE-FOLD COMPARISON\n" + comparison.to_string(
        index=False, float_format=lambda value: f"{value:.5f}"))
    winner = comparison.iloc[0]["model"]
    tell(f"\nWinner by MEAN held-out F1: {winner}")
    report = classification_report(y, oof[winner], labels=rich.LABELS,
                                   digits=4, zero_division=0)
    tell("\nPooled held-out classification report:\n" + report)
    comparison.to_csv(OUT / "cv_comparison.csv", index=False)
    (OUT / "cv_report.txt").write_text(report, encoding="utf-8")

    # Tune final submission costs on all INNER tuning predictions only.
    # Every row represented here was excluded from that fold's model fitting.
    tune_p = np.vstack(all_tune["baseline" if winner == "baseline" else "rich"])
    tune_y = np.concatenate(all_tune["y"])
    tuner = rich.joint_rule if winner == "candidate_joint" else rich.old_rule
    final_cost, final_weights, final_pairs = tuner(tune_p, tune_y)
    selected = dict(signature=sig, winner=winner, mean_cv_f1=float(comparison.iloc[0]["mean_F1"]),
                    final_cost_matrix=final_cost.tolist(),
                    final_weights=np.asarray(final_weights).tolist(),
                    final_pair_multipliers=np.asarray(final_pairs).tolist(),
                    note="Costs fitted on pooled inner-tuning predictions; fold scores use separate outer held-out reviews")
    save_json(OUT / "selected.json", selected)
    tell(f"\nSaved winner and costs in {OUT / 'selected.json'}")
    tell("No Kaggle submission created. CV scores are estimates, not guarantees.")


def submit(sig):
    selected_path = OUT / "selected.json"
    if not selected_path.exists():
        raise FileNotFoundError("Run MODE='cv' first with the same RUN_NAME.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["signature"] != sig:
        raise ValueError("Data/model/fold settings changed. Use CV with a new RUN_NAME.")
    filename = Path(SUBMISSION_NAME)
    if (filename.name != SUBMISSION_NAME or filename.suffix.lower() != ".csv" or
        filename.name.lower() in {"train.csv", "test.csv", "sample_submission.csv"}):
        raise ValueError("Use a safe submission filename ending in .csv")
    dest = BASE / filename
    if dest.exists():
        raise FileExistsError(f"File exists: {dest}. Choose a new SUBMISSION_NAME")

    raw = rich.read_data()
    known = raw.loc[raw.Score.notna()]
    unknown = raw.loc[raw.Score.isna()].set_index("id", drop=False)
    test_ids = pd.read_csv(BASE / "test.csv", dtype=str)["id"]
    if (test_ids.empty or test_ids.isna().any() or test_ids.duplicated().any() or
            set(test_ids) != set(unknown.index)):
        raise ValueError("test.csv IDs must match exactly the missing-Score IDs in train.csv")
    winner = selected["winner"]
    is_baseline = winner == "baseline"
    y = known.Score.astype(int).to_numpy()
    tell(f"Refitting {winner} on ALL {len(y):,} known ratings...")
    transformer = rich.make_features("standard" if is_baseline else "rich")
    a = transformer.fit_transform(known, y).tocsr()
    b = rich.transform_batches(transformer, unknown.loc[test_ids]).tocsr()
    p = rich.fit_predict(a, y, b, transformer.output_indices_["ids"].start,
                         1e-5 if is_baseline else ALPHA,
                         30 if is_baseline else 50,
                         1e-3 if is_baseline else 1e-4)
    pred = rich.decide(p, np.asarray(selected["final_cost_matrix"]))
    pd.DataFrame({"id": test_ids.to_numpy(), "Score": pred}).to_csv(dest, index=False)
    save_json(dest.with_suffix(".settings.json"), selected)
    tell(f"Saved {len(pred):,} predictions in test.csv order: {dest}")
    tell("Commit source code; do not commit CSV/data/cache files.")


def main():
    start = perf_counter()
    if MODE not in {"cv", "submit"}:
        raise ValueError("MODE must be 'cv' or 'submit'")
    if N_FOLDS < 2 or not 0 < INNER_TUNE_FRACTION < 0.5:
        raise ValueError("Invalid N_FOLDS or INNER_TUNE_FRACTION")
    if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
        raise ValueError("RUN_NAME must be a simple folder name")
    if not 0 <= NB_BLEND <= 1 or ALPHA <= 0:
        raise ValueError("Invalid NB_BLEND or ALPHA")
    OUT.mkdir(parents=True, exist_ok=True)
    tell(f"MODE={MODE} | folds={N_FOLDS} | NB_BLEND={NB_BLEND} | ALPHA={ALPHA}")
    sig = signature()
    if MODE == "cv":
        cross_validate(sig)
    else:
        submit(sig)
    tell(f"Finished in {perf_counter() - start:.1f} seconds")


if __name__ == "__main__":
    main()
