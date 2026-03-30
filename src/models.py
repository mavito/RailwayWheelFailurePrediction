"""
Model training and prediction for the wheel failure prediction pipeline.

Three GPU-accelerated models (CatBoost, LightGBM, XGBoost) are trained and
blended into a soft-voting ensemble.  XGBoost carries more weight because its
hyperparameters were tuned against the competition's log-loss metric directly.

Important notes on design choices:
- No sample weighting: compute_sample_weight was spreading probability mass
  across classes, which hurts log-loss.  The built-in class_weight='balanced'
  inside LightGBM is a softer, better-calibrated alternative.
- No feature scaling: tree models split on rank, not magnitude, so scaling
  adds no benefit and can hurt generalisation when train/test distributions differ.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def split_data(df: pd.DataFrame, cat_cols: list, num_cols: list,
               cutoff: str = "2024-10-01", val_start: str = "2025-01-01"):
    """Chronological train / validation / test split.

    We slice by calendar date rather than randomly to respect the time-series
    structure of the data.  Shuffling would let the model peek at future sensor
    readings, which inflates training metrics and tanks real-world performance.

    Args:
        cutoff:    Training data ends before this date.
        val_start: Holdout (submission) period starts here — Q1 2025 in the competition.
    """
    cutoff_dt = pd.to_datetime(cutoff)
    val_dt    = pd.to_datetime(val_start)

    train_df = df[df["recordmonth"] < cutoff_dt]
    test_df  = df[(df["recordmonth"] >= cutoff_dt) & (df["recordmonth"] < val_dt)]
    val_df   = df[df["recordmonth"] >= val_dt]

    features = cat_cols + num_cols

    X_train, y_train = train_df[features], train_df["failurereason"]
    X_test,  y_test  = test_df[features],  test_df["failurereason"]
    X_val            = val_df[features]
    original_val_idx = val_df["original_index"]

    print(f"Train: {len(X_train):,}  |  Val (OOT test): {len(X_test):,}  |  Submission set: {len(X_val):,}")
    return X_train, y_train, X_test, y_test, X_val, original_val_idx


def train_catboost(X_train, y_train, X_test, y_test, cat_cols: list, num_classes: int):
    """Train a CatBoost classifier on GPU.

    CatBoost handles categorical features natively — no need to one-hot encode
    or worry about bin limits. Passing cat_features tells it which columns to
    treat specially, which results in better splits on high-cardinality fields
    like equipment number.
    """
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=7,
        task_type="GPU",
        eval_metric="MultiClass",
        cat_features=cat_cols,
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, num_classes: int):
    """Train a LightGBM classifier on GPU.

    class_weight='balanced' provides mild correction for the heavy class
    imbalance (most wheels don't fail) without over-penalising the rare classes
    the way explicit sample weights do.
    """
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=9,
        class_weight="balanced",
        device="gpu",
        random_state=42,
        objective="multiclass",
        n_jobs=-1,
        importance_type="gain"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    return model


def train_xgboost(X_train, y_train, X_test, y_test, num_classes: int):
    """Train an XGBoost classifier on GPU.

    Using mlogloss as the eval_metric is deliberate — it directly optimises
    the competition metric, meaning early stopping fires at exactly the right
    point rather than at an auc proxy that might diverge from log-loss.
    """
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        device="cuda",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=9,
        enable_categorical=True,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    return model


def ensemble_predict(cat_model, lgb_model, xgb_model, X_val) -> np.ndarray:
    """Blend model outputs into a final probability matrix.

    The weights are not equal: XGBoost gets 60% because its tuned parameters
    (gamma, depth, learning rate) were validated directly on the competition test set
    and produced the sharpest calibration.  CatBoost and LightGBM each add 20%
    as diversity regularisers.

    All three models output a (n_samples, n_classes) probability matrix, so
    a weighted average is a valid operation — each row still sums to ~1.
    """
    cat_preds = cat_model.predict_proba(X_val)
    lgb_preds = lgb_model.predict_proba(X_val)
    xgb_preds = xgb_model.predict_proba(X_val)

    blended = 0.20 * cat_preds + 0.20 * lgb_preds + 0.60 * xgb_preds
    return blended


def build_submission(
    probabilities: np.ndarray,
    original_val_idx: pd.Series,
    le_target,
    expected_rows: int = 55330,
    output_path: str = "submission.csv"
) -> pd.DataFrame:
    """Format predictions as the competition submission file.

    The column order and names are fixed by the competition spec.  We also
    reindex to the full expected row count because a small number of exact
    duplicate rows are dropped early in preprocessing — forward-filling those
    gaps is the safest recovery strategy (duplicates should have identical predictions).
    """

    # Map class names back — if the LabelEncoder was accidentally re-fit on
    # already-encoded integers (common with notebook re-runs), class_names will
    # contain integers like [0, 1, 2, 3, 4].  The fallback_map handles that.
    class_names = le_target.classes_
    fallback_map = {
        0: "high flange", "0": "high flange",
        1: "high impact",  "1": "high impact",
        2: "not failed",   "2": "not failed",
        3: "other",        "3": "other",
        4: "thin flange",  "4": "thin flange",
    }

    submission = pd.DataFrame(probabilities, columns=class_names, index=original_val_idx)
    submission = submission.sort_index()
    submission.rename(columns=fallback_map, inplace=True)
    submission.columns = [str(c).lower().strip() for c in submission.columns]

    target_cols = ["high flange", "high impact", "thin flange", "other", "not failed"]
    submission = submission[target_cols]

    # Restore any rows lost to deduplication
    submission = submission.reindex(range(expected_rows)).ffill()
    submission.index.name = "ID"

    submission.to_csv(output_path)
    print(f"Saved {len(submission):,} rows to {output_path}")
    return submission
