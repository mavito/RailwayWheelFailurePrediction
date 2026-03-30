"""
Feature engineering for the wheel failure prediction pipeline.

All transformations here are vectorised — no row-wise apply() loops.
That matters because pandas 2.2+ drops groupby keys from apply results,
which caused silent KeyErrors in earlier versions of this code.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# The combination of columns that uniquely identifies one physical wheel over time
WHEEL_GROUP = ["equipmentnumber", "truck", "axle", "side", "applieddate"]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and mileage-based features that capture wear progression.

    The core insight: a wheel that travels 5,000 miles in one month right after
    installation is under very different stress than one that has been on the rail
    for three years at the same mileage rate.  These features express that stress.
    """
    df = df.copy()

    # How long has this wheel been in service?
    df["days_since_applied"] = (df["recordmonth"] - df["applieddate"]).dt.days

    # Miles per day — a wheel doing the same total mileage over fewer days is burning faster
    df["mileage_intensity"] = df["partmileage"] / (df["days_since_applied"] + 1)

    # Month-over-month mileage change (requires sorted order within each wheel group)
    df_g = df.groupby(WHEEL_GROUP)
    df["lag_partmileage_1"] = df_g["partmileage"].shift(1)
    df["travelled_since_lastmonth"] = df["partmileage"] - df["lag_partmileage_1"]

    # 3-month rolling average of monthly mileage — smooths out one-off spikes
    df["partmileage_3monthavg"] = (
        df.groupby(WHEEL_GROUP)["travelled_since_lastmonth"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    return df


def add_detector_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute month-over-month changes in key detector readings.

    A flange that dropped 2 mm last month is a much stronger signal than
    a flange that has been consistently 2 mm below average for years.
    Delta features capture that rate-of-change information.
    """
    df = df.copy()
    df_g = df.groupby(WHEEL_GROUP)

    df["delta_flangeheight"]    = df_g["min_flangeheight"].diff(1)
    df["delta_flangethickness"] = df_g["min_flangethickness"].diff(1)
    df["delta_maxvertical"]     = df_g["max_maxvertical"].diff(1)
    df["delta_dynamicratio"]    = df_g["max_dynamicratio"].diff(1)

    return df


def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the month of the year as sine and cosine components.

    A plain integer month column (1–12) would make the model see January and
    December as far apart when they are actually adjacent in winter conditions.
    Cyclical encoding fixes that by projecting month onto a circle.
    """
    df = df.copy()
    df["recordmonth_month_sin"] = np.sin(2 * np.pi * df["recordmonth"].dt.month / 12)
    df["recordmonth_month_cos"] = np.cos(2 * np.pi * df["recordmonth"].dt.month / 12)
    return df


def forward_fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps in detector readings by carrying the last known value forward.

    A wheel that didn't pass a WPD site in March probably has very similar
    dimensions to what was recorded in February.  We propagate up to 3 months
    forward — beyond that we'd rather use 0 (no signal) than stale data.
    """
    df = df.copy()

    fill_cols = [
        c for c in df.columns
        if c not in WHEEL_GROUP and c not in ["failurereason", "recordmonth", "applieddate"]
    ]

    df[fill_cols] = df.groupby(WHEEL_GROUP)[fill_cols].ffill(limit=3)
    df.fillna(0, inplace=True)

    return df


def encode_and_prepare(df: pd.DataFrame, min_date: str = "2020-01-01"):
    """Label-encode categoricals, drop non-features, encode the target, and return the final frame.

    Pre-2020 data has sparse sensor coverage and different baseline distributions,
    so we drop it.  It would add noise more than signal.

    Returns:
        df: The prepared DataFrame.
        cat_cols: List of categorical column names (now integer-encoded).
        num_cols: List of numerical column names.
        le_target: Fitted LabelEncoder for the target (needed to map prediction
                   integers back to class names when building the submission file).
    """
    df = df[df["recordmonth"] >= min_date].copy()

    cat_cols = ["equipmentnumber", "vendornumbersuppliercode", "truck", "material", "axle", "side"]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    num_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns.tolist()

    # Strip out columns that are identifiers or targets, not features
    non_feature = ["original_index", "failurereason", "recordmonth", "applieddate"] + cat_cols
    num_cols = [c for c in num_cols if c not in non_feature]

    le_target = LabelEncoder()
    df["failurereason"] = le_target.fit_transform(df["failurereason"])

    return df, cat_cols, num_cols, le_target


def run_feature_engineering(df: pd.DataFrame):
    """Run the full feature engineering pipeline end to end.

    Calling this single function is enough — it applies all the steps
    in the right order and returns everything you need for training.
    """
    df = add_temporal_features(df)
    df = add_detector_deltas(df)
    df = add_cyclic_month(df)
    df = forward_fill_gaps(df)
    df, cat_cols, num_cols, le_target = encode_and_prepare(df)

    print(f"Feature engineering done. Dataset: {df.shape}, Features: {len(cat_cols + num_cols)}")
    return df, cat_cols, num_cols, le_target
