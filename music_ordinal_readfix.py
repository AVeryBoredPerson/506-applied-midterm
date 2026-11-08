"""Ordinal (ordered-rating) music predictor: classical ML only.

Run with this file beside train.csv and test.csv:
    python music_ordinal.py

1. MODE='validate': split known ratings, tune class preferences on HALF the
   validation reviews, and report performance on the OTHER untouched half.
   Saves plots/ordinal/weights.json and two diagnostic graphs.
2. MODE='submit': train the classifiers on ALL known ratings using the
   vocabulary saved during validation, apply saved weights, and write a
   named Kaggle submission. No labels from test.csv are used.

No neural nets, boosting, or rating-history leakage.
For your assignment README, cite and explain these methods and your AI use:
https://scikit-learn.org/stable/modules/sgd.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
"""
from pathlib import Path
from time import perf_counter
import json
import joblib
from scipy.sparse import vstack

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------- YOUR SETTINGS -----------------------
MODE = "validate"                 # "validate" OR "submit"
ALPHA = 0.00001                   # Regularization of each binary classifier
TRAIN_START = 0                   # For validation only: slice shuffled training rows
TRAIN_END = None                  # None = all local training rows
SUBMISSION_NAME = "ordinal_v1.csv"  # Used only for MODE='submit'
# -------------------------------------------------------------

BASE = Path(__file__).resolve().parent
OUT = BASE / "plots" / "ordinal"
WEIGHTS_FILE = OUT / "weights.json"
FEATURES_FILE = OUT / "fitted_features.joblib"
RATINGS = [1, 2, 3, 4, 5]
ID_COLS = ["reviewerID", "album_mbid", "artist_mbid"]


def prepare(df):
    """Simple row-level features; no ratings are used to build them."""
    df = df.copy()
    for col in ["reviewText", "summary", "genres"] + ID_COLS:
        df[col] = df[col].fillna("").astype(str)
    df["genres"] = df["genres"].str.replace(r"\s*,\s*", ",", regex=True)

    votes = pd.to_numeric(df["TotalVotes"], errors="coerce").fillna(0).clip(lower=0)
    helpful = pd.to_numeric(df["VotedHelpful"], errors="coerce").fillna(0).clip(lower=0)
    df["helpful_share"] = (helpful / votes.replace(0, np.nan)).fillna(0).clip(0, 1)
    df["log_votes"] = np.log1p(votes)
    df["log_length"] = np.log1p(df["reviewText"].str.len())
    df["year"] = pd.to_datetime(
        pd.to_numeric(df["unixReviewTime"], errors="coerce"),
        unit="s", errors="coerce"
    ).dt.year.fillna(2010)
    for name in ["helpful_share", "log_votes", "log_length", "year"]:
        df[name] = df[name].astype(np.float32)
    return df


def make_features():
    """Exactly the same features as our 0.5270 full-validation experiment."""
    words = dict(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
                 token_pattern=r"(?u)\b\w+\b")
    return ColumnTransformer([
        ("review", TfidfVectorizer(min_df=3, max_features=80000, **words), "reviewText"),
        ("summary", TfidfVectorizer(min_df=2, max_features=15000, **words), "summary"),
        ("genres", TfidfVectorizer(token_pattern=r"[^,]+", dtype=np.float32), "genres"),
        ("ids", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ID_COLS),
        ("numbers", StandardScaler(), ["helpful_share", "log_votes", "log_length", "year"]),
    ], sparse_threshold=1.0)


def fit_ordinal(train_matrix, y):
    """Train four binary classifiers for P(Score > 1), ..., P(Score > 4)."""
    models = []
    for cutoff in [1, 2, 3, 4]:
        classifier = SGDClassifier(
            loss="log_loss", alpha=ALPHA, average=True,
            max_iter=30, tol=1e-3, random_state=42,
        )
        classifier.fit(train_matrix, (y > cutoff).astype(int))
        models.append(classifier)
        print(f"  Fitted threshold: Score > {cutoff}", flush=True)
    return models


def rating_probabilities(models, matrix):
    """Convert four cumulative predictions into five rating probabilities."""
    q = np.column_stack([m.predict_proba(matrix)[:, 1] for m in models])
    # An ordered rating must satisfy P(>1) >= P(>2) >= P(>3) >= P(>4).
    q = np.minimum.accumulate(q, axis=1)
    return np.column_stack([
        1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1] - q[:, 2],
        q[:, 2] - q[:, 3], q[:, 3],
    ])


def predict(probabilities, weights):
    """Multiply each class's probability by its preference weight."""
    return np.argmax(probabilities * weights, axis=1) + 1


def macro_f1(actual, predicted):
    return f1_score(actual, predicted, labels=RATINGS, average="macro", zero_division=0)


def tune_weights(probs, labels):
    """Search four class weights on a calibration set; keep weight 5 fixed at 1."""
    options = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5]
    weights = np.ones(5)
    for _ in range(3):
        for rating_idx in range(4):
            best_score, best_weight = -1.0, weights[rating_idx]
            for candidate in options:
                trial = weights.copy()
                trial[rating_idx] = candidate
                score = macro_f1(labels, predict(probs, trial))
                if score > best_score:
                    best_score, best_weight = score, candidate
            weights[rating_idx] = best_weight
    return weights


def make_plots(y_train, p_train, y_audit, p_audit, weights):
    OUT.mkdir(parents=True, exist_ok=True)
    train_f1 = f1_score(y_train, predict(p_train, weights), labels=RATINGS,
                        average=None, zero_division=0)
    audit_f1 = f1_score(y_audit, predict(p_audit, weights), labels=RATINGS,
                        average=None, zero_division=0)
    plt.figure(figsize=(7, 4))
    plt.plot(RATINGS, train_f1, "o-", label="Training")
    plt.plot(RATINGS, audit_f1, "o-", label="Untouched audit")
    plt.xticks(RATINGS)
    plt.ylim(0, 1)
    plt.xlabel("Rating")
    plt.ylabel("F1")
    plt.title("F1 by rating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "class_f1.png", dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_audit, predict(p_audit, weights), labels=RATINGS,
        normalize="true", values_format=".2f", cmap="Blues",
    )
    plt.title("Untouched audit confusion matrix")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    start = perf_counter()
    raw = pd.read_csv(
        BASE / "train.csv",
        dtype={"id": str, "Score": str, "unixReviewTime": str,
               "VotedHelpful": str, "TotalVotes": str},
        low_memory=False,
    )
    raw["Score"] = pd.to_numeric(raw["Score"], errors="coerce")
    raw = prepare(raw)
    known = raw.loc[raw["Score"].notna()]
    X = known.drop(columns="Score")
    y = known["Score"].astype(int)
    if not y.isin(RATINGS).all():
        raise ValueError("Scores must be between 1 and 5.")

    if MODE == "validate":
        # Fixed split: same seed and 20% holdout as the earlier SVM.
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = X_train.iloc[TRAIN_START:TRAIN_END]
        y_train = y_train.iloc[TRAIN_START:TRAIN_END]
        if y_train.nunique() != 5:
            raise ValueError("Selected training rows must contain all five scores.")

        print(f"Training: {len(X_train):,}; validation: {len(X_valid):,}", flush=True)
        features = make_features()
        train_matrix = features.fit_transform(X_train, y_train)
        valid_matrix = features.transform(X_valid)
        print(f"Features ready: {train_matrix.shape[1]:,}; {perf_counter()-start:.1f}s", flush=True)
        models = fit_ordinal(train_matrix, y_train)
        p_valid = rating_probabilities(models, valid_matrix)
        raw_pred = predict(p_valid, np.ones(5))
        print(f"\nRaw ordinal validation Macro-F1: {macro_f1(y_valid, raw_pred):.4f}")

        # Half the validation labels tune preferences; the other half only scores them.
        cal_idx, audit_idx = train_test_split(
            np.arange(len(y_valid)), train_size=0.5,
            random_state=17, stratify=y_valid,
        )
        weights = tune_weights(p_valid[cal_idx], y_valid.iloc[cal_idx])
        audit_pred = predict(p_valid[audit_idx], weights)
        print(f"Raw audit Macro-F1: {macro_f1(y_valid.iloc[audit_idx], raw_pred[audit_idx]):.4f}")
        print(f"Tuned audit Macro-F1: {macro_f1(y_valid.iloc[audit_idx], audit_pred):.4f}")
        print("Rating weights:", np.round(weights, 2).tolist())
        report = classification_report(y_valid.iloc[audit_idx], audit_pred,
                                       labels=RATINGS, digits=4, zero_division=0)
        print("\n" + report)
        make_plots(y_train, rating_probabilities(models, train_matrix),
                   y_valid.iloc[audit_idx], p_valid[audit_idx], weights)

        OUT.mkdir(parents=True, exist_ok=True)
        # Save the fitted vocabulary/encoders so submission can reuse them
        # without re-learning millions of n-grams from the larger full dataset.
        joblib.dump(features, FEATURES_FILE)
        WEIGHTS_FILE.write_text(json.dumps({
            "weights": weights.tolist(), "alpha": ALPHA, "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "note": "Tuned on calibration half only; evaluated on untouched audit half.",
        }, indent=2), encoding="utf-8")
        (OUT / "results.txt").write_text(
            f"Raw full validation Macro-F1: {macro_f1(y_valid, raw_pred):.6f}\n"
            f"Raw audit Macro-F1: {macro_f1(y_valid.iloc[audit_idx], raw_pred[audit_idx]):.6f}\n"
            f"Tuned audit Macro-F1: {macro_f1(y_valid.iloc[audit_idx], audit_pred):.6f}\n"
            f"Weights: {weights.tolist()}\n\n{report}", encoding="utf-8",
        )
        print(f"Plots and calibrated weights saved in {OUT}")

    elif MODE == "submit":
        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError("First run MODE='validate' to create plots/ordinal/weights.json")
        saved = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
        if saved["alpha"] != ALPHA or saved["train_start"] != TRAIN_START or saved["train_end"] != TRAIN_END:
            raise ValueError("Settings changed after validation; recalibrate first.")
        weights = np.asarray(saved["weights"])
        submission_path = BASE / SUBMISSION_NAME
        if (Path(SUBMISSION_NAME).name != SUBMISSION_NAME or not SUBMISSION_NAME.endswith(".csv")
                or SUBMISSION_NAME in {"train.csv", "test.csv", "sample_submission.csv"}):
            raise ValueError("Choose a new filename ending in .csv.")
        if submission_path.exists():
            raise FileExistsError(f"{SUBMISSION_NAME} already exists. Choose a new filename.")

        unknown = raw.loc[raw["Score"].isna()].drop(columns="Score")
        test_ids = pd.read_csv(BASE / "test.csv", dtype={"id": str})[["id"]]
        print(f"Final training: {len(X):,}; unknown ratings: {len(unknown):,}", flush=True)
        if not FEATURES_FILE.exists():
            raise FileNotFoundError("First run MODE='validate' to save the fitted vocabulary.")
        features = joblib.load(FEATURES_FILE)
        # Transform in small batches to stay within memory. The classifiers
        # still train on ALL known ratings. Vocabulary is frozen from validation.
        def batched_transform(data, chunk_size=25000):
            return vstack([
                features.transform(data.iloc[i:i + chunk_size])
                for i in range(0, len(data), chunk_size)
            ], format="csr")
        train_matrix = batched_transform(X)
        test_matrix = batched_transform(unknown)
        print(f"Features ready: {train_matrix.shape[1]:,}; {perf_counter()-start:.1f}s", flush=True)
        models = fit_ordinal(train_matrix, y)
        predicted = predict(rating_probabilities(models, test_matrix), weights)
        submission = test_ids.merge(pd.DataFrame({"id": unknown["id"], "Score": predicted}),
                                    on="id", how="left", validate="one_to_one", sort=False)
        if submission["Score"].isna().any() or len(submission) != len(test_ids):
            raise ValueError("Predictions did not match every test ID.")
        submission["Score"] = submission["Score"].astype(int)
        submission.to_csv(submission_path, index=False)
        print(f"Kaggle submission: {submission_path} ({len(submission):,} rows)")
    else:
        raise ValueError("MODE must be 'validate' or 'submit'.")
    print(f"Total time: {perf_counter()-start:.1f}s")


if __name__ == "__main__":
    main()
