"""
Main entry point for the wheel failure prediction pipeline.

Run this from the same directory as your CSV data files:

    python main.py

Any intermediate DuckDB file (rail_informs.duckdb) is created automatically
and can be safely deleted between runs — it gets rebuilt from the CSVs.
"""

import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from src.data_loader import (
    connect_duckdb,
    aggregate_wpd,
    aggregate_thd,
    aggregate_wild,
    load_and_join,
)
from src.features import run_feature_engineering
from src.models import split_data, train_catboost, train_lightgbm, train_xgboost, ensemble_predict, build_submission


def main():
    # -------------------------------------------------------------------------
    # Step 1 — Load raw CSVs into DuckDB and build aggregated detector tables
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading CSVs and building detector aggregations...")
    con = connect_duckdb("rail_informs.duckdb", threads=8)
    aggregate_wpd(con)
    aggregate_thd(con)
    aggregate_wild(con)
    df = load_and_join(con)

    # -------------------------------------------------------------------------
    # Step 2 — Feature engineering
    # -------------------------------------------------------------------------
    print("\n[2/5] Running feature engineering...")
    df, cat_cols, num_cols, le_target = run_feature_engineering(df)
    num_classes = len(le_target.classes_)

    # -------------------------------------------------------------------------
    # Step 3 — Chronological split
    # -------------------------------------------------------------------------
    print("\n[3/5] Splitting data (train < Oct 2024, OOT test Oct-Dec 2024, submission >= Jan 2025)...")
    X_train, y_train, X_test, y_test, X_val, original_val_idx = split_data(
        df, cat_cols, num_cols,
        cutoff="2024-10-01",
        val_start="2025-01-01"
    )

    # -------------------------------------------------------------------------
    # Step 4 — Train the three models
    # -------------------------------------------------------------------------
    print("\n[4/5] Training ensemble models (GPU required)...")

    print("  → CatBoost")
    cat_model = train_catboost(X_train, y_train, X_test, y_test, cat_cols, num_classes)

    print("  → LightGBM")
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test, num_classes)

    print("  → XGBoost")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test, num_classes)

    # -------------------------------------------------------------------------
    # Step 5 — Blend predictions and write submission file
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating ensemble predictions and writing submission...")
    probabilities = ensemble_predict(cat_model, lgb_model, xgb_model, X_val)

    submission = build_submission(
        probabilities=probabilities,
        original_val_idx=original_val_idx,
        le_target=le_target,
        expected_rows=55330,
        output_path="submission.csv"
    )

    print("\nPreview of submission file:")
    print(submission.head())
    print(f"\nAll done. Submit 'submission.csv' to Kaggle.")


if __name__ == "__main__":
    main()
