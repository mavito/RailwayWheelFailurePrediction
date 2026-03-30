"""
Data loading utilities for the wheel failure prediction pipeline.

Handles reading raw CSVs into DuckDB and running the SQL aggregations that
collapse per-pass detector readings (WPD, THD, WILD) down to one row per
wheel per month.
"""

import gc
import numpy as np
import pandas as pd
import duckdb


def connect_duckdb(db_path: str = "rail_informs.duckdb", threads: int = 8) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and ingest all raw CSVs.

    We wipe existing tables first so re-runs are always clean — no stale
    data left over from a previous session.
    """
    con = duckdb.connect(db_path, config={"threads": threads})

    # Drop anything sitting around from a previous run
    existing = con.execute("SHOW ALL TABLES;").df()["name"].tolist()
    for tbl in existing:
        con.execute(f"DROP TABLE IF EXISTS {tbl};")

    con.execute("""
        CREATE TABLE failure      AS SELECT * FROM read_csv_auto('failure_data_masked.csv');
        CREATE TABLE failure_test AS SELECT * FROM read_csv_auto('failure_data_masked_test.csv');
        CREATE TABLE mileage      AS SELECT * FROM read_csv_auto('mileage_data_masked.csv');
        CREATE TABLE thd          AS SELECT * FROM read_csv_auto('thd_data_masked.csv');
        CREATE TABLE wild         AS SELECT * FROM read_csv_auto('wild_data_masked.csv');
        CREATE TABLE wpd          AS SELECT * FROM read_csv_auto('wpd_data_masked.csv');
    """)

    return con


def aggregate_wpd(con: duckdb.DuckDBPyConnection) -> None:
    """Roll up Wheel Profile Detector (WPD) readings to monthly min/max per wheel.

    Each wheel passes the same sensor point multiple times per month, so we
    collapse those passes to a single row carrying the monthly extremes.
    """
    con.execute("""
        CREATE OR REPLACE TABLE transformed_wpd AS (
            WITH InitialTransform AS (
                SELECT
                    strftime(traindate, '%Y-%m-01') AS recordmonth,
                    CONCAT(equipmentnumber, '_', truck, '_', axle, '_', side) AS wheel_id,
                    trainspeed, wheeldiameter, flangeheight,
                    flangethickness, flangeslope, rimthickness
                FROM wpd
            )
            SELECT
                recordmonth, wheel_id,
                MIN(trainspeed)        AS min_trainspeed,   MAX(trainspeed)        AS max_trainspeed,
                MIN(wheeldiameter)     AS min_wheeldiameter,MAX(wheeldiameter)     AS max_wheeldiameter,
                MIN(flangeheight)      AS min_flangeheight, MAX(flangeheight)      AS max_flangeheight,
                MIN(flangethickness)   AS min_flangethickness,MAX(flangethickness) AS max_flangethickness,
                MIN(flangeslope)       AS min_flangeslope,  MAX(flangeslope)       AS max_flangeslope,
                MIN(rimthickness)      AS min_rimthickness, MAX(rimthickness)      AS max_rimthickness
            FROM InitialTransform
            GROUP BY wheel_id, recordmonth
            ORDER BY wheel_id, recordmonth
        );
    """)


def aggregate_thd(con: duckdb.DuckDBPyConnection) -> None:
    """Roll up Truck Hunting Detector (THD) readings to monthly min/max per bogie.

    THD captures side-to-side oscillation. We summarise the range over each month
    because a single spike matters just as much as a sustained pattern.
    """
    con.execute("""
        CREATE OR REPLACE TABLE transformed_thd AS (
            WITH InitialTransform AS (
                SELECT
                    strftime(traindate, '%Y-%m-01') AS recordmonth,
                    CONCAT(equipmentnumber, '_', truck) AS bogie_id,
                    trainspeed, percentload, huntingindex, leftrightimbalance
                FROM thd
            )
            SELECT
                recordmonth, bogie_id,
                MIN(trainspeed)          AS min_trainspeed,          MAX(trainspeed)          AS max_trainspeed,
                MIN(percentload)         AS min_percentload,          MAX(percentload)         AS max_percentload,
                MIN(huntingindex)        AS min_huntingindex,         MAX(huntingindex)        AS max_huntingindex,
                MIN(leftrightimbalance)  AS min_leftrightimbalance,   MAX(leftrightimbalance)  AS max_leftrightimbalance
            FROM InitialTransform
            GROUP BY bogie_id, recordmonth
            ORDER BY bogie_id, recordmonth
        );
    """)


def aggregate_wild(con: duckdb.DuckDBPyConnection) -> None:
    """Roll up Wheel Impact Load Detector (WILD) readings to monthly stats per wheel.

    WILD measures rail forces. We keep min, mean, and max because all three carry
    signal — the peak tells us about worst-case impacts, the average about chronic load.
    """
    con.execute("""
        CREATE OR REPLACE TABLE transformed_wild AS (
            WITH InitialTransform AS (
                SELECT
                    strftime(traindate, '%Y-%m-01') AS recordmonth,
                    CONCAT(equipmentnumber, '_', truck, '_', axle, '_', side) AS wheel_id,
                    averagevertical, maxvertical, dynamicvertical, dynamicratio
                FROM wild
            )
            SELECT
                recordmonth, wheel_id,
                MIN(averagevertical)  AS min_averagevertical,  AVG(averagevertical)  AS avg_averagevertical,  MAX(averagevertical)  AS max_averagevertical,
                MIN(maxvertical)      AS min_maxvertical,       AVG(maxvertical)      AS avg_maxvertical,       MAX(maxvertical)      AS max_maxvertical,
                MIN(dynamicvertical)  AS min_dynamicvertical,   AVG(dynamicvertical)  AS avg_dynamicvertical,   MAX(dynamicvertical)  AS max_dynamicvertical,
                MIN(dynamicratio)     AS min_dynamicratio,      AVG(dynamicratio)     AS avg_dynamicratio,      MAX(dynamicratio)     AS max_dynamicratio
            FROM InitialTransform
            GROUP BY wheel_id, recordmonth
            ORDER BY wheel_id, recordmonth
        );
    """)


def load_and_join(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pull train and test failure records, join with all detector tables, and return one DataFrame.

    Why merge train and test before joining?  Because feature engineering later
    computes rolling/lagged statistics per wheel over time.  If the test period
    (Q1 2025) is isolated, those rolling windows would be empty for the first month.
    Appending test at the bottom lets the groupby see the full history.
    """
    failure      = con.execute("SELECT * FROM failure;").df()
    failure_test = con.execute("SELECT * FROM failure_test;").df()

    # Keep track of which rows came from which source later (the index becomes our submission ID)
    df = pd.concat([failure, failure_test], axis=0).reset_index(names="original_index")
    del failure, failure_test
    gc.collect()

    # Some older records have missing identifiers — fill with sensible placeholders
    df["truck"] = df["truck"].fillna("ENG")
    df["axle"]  = df["axle"].fillna(0)
    df["side"]  = df["side"].fillna("L")

    # Remove rows that are completely identical (same part, same reading, same month)
    df = df.drop_duplicates(subset=df.columns.difference(["original_index"]), keep="first")

    # Wheels that never failed still get a label — they're just "not failed"
    if "failedin30days" in df.columns:
        df.drop(["failedin30days"], axis=1, inplace=True)
    df.fillna({"failurereason": "not failed"}, inplace=True)

    # Encode the asbuilt flag — 'Y'/'N' to 1/0
    df["asbuilt"] = df["asbuilt"].map({"Y": 1, "N": 0}).fillna(0)

    # Downcast dates and mileage to save RAM
    df["recordmonth"] = pd.to_datetime(df["recordmonth"])
    df["applieddate"]  = pd.to_datetime(df["applieddate"])
    df["partmileage"]  = df["partmileage"].astype(np.float32)

    # Push back to DuckDB so we can do the multi-table join in SQL (faster than pandas merge chains)
    con.execute("CREATE OR REPLACE TABLE df_base AS SELECT * FROM df")

    con.execute("""
        CREATE OR REPLACE TABLE final_data AS (
            SELECT
                f.*,
                thd.* EXCLUDE (recordmonth, bogie_id),
                wpd.* EXCLUDE (recordmonth, wheel_id),
                wild.* EXCLUDE (recordmonth, wheel_id),
                m.* EXCLUDE (recordmonth, equipmentnumber)
            FROM df_base AS f
            LEFT JOIN transformed_thd AS thd
                ON thd.recordmonth = f.recordmonth
               AND thd.bogie_id    = CONCAT(f.equipmentnumber, '_', f.truck)
            LEFT JOIN transformed_wpd AS wpd
                ON wpd.recordmonth = f.recordmonth
               AND wpd.wheel_id    = CONCAT(f.equipmentnumber, '_', f.truck, '_', f.axle, '.0_', f.side)
            LEFT JOIN transformed_wild AS wild
                ON wild.recordmonth = f.recordmonth
               AND wild.wheel_id    = CONCAT(f.equipmentnumber, '_', f.truck, '_', f.axle, '_', f.side)
            LEFT JOIN mileage AS m
                ON m.recordmonth    = f.recordmonth
               AND m.equipmentnumber = f.equipmentnumber
        );
    """)

    df = con.execute("SELECT * FROM final_data;").df()

    # Sort by wheel then time so time-series operations later work correctly
    df.sort_values(
        by=["equipmentnumber", "truck", "axle", "side", "applieddate", "recordmonth"],
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded dataset shape: {df.shape}")
    return df
