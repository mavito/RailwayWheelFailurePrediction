# Railway Wheel Failure Prediction

**2025 INFORMS Railway Application Section Problem Solving Competition**

Predicts the type of wheel failure (high flange, high impact, thin flange, other, or not failed) 30 days in advance using seven years of railcar sensor data.

**Competition score with this approach: 0.030 log-loss** (vs 0.037 with a single XGBoost baseline).

---

## What This Is

Wheels wear out. When they do, their profile changes in measurable ways — the flange either grows too tall, too thin, or absorbs too many impacts. Wheel Profile Detectors (WPD), Truck Hunting Detectors (THD), and Wheel Impact Load Detectors (WILD) record these measurements as trains pass fixed stations along the track.

This project combines those readings with monthly mileage data and the railcar's physical specs to predict which type of failure, if any, is coming in the next 30 days.

---

## Project Structure

```
wheel_failure_prediction/
│
├── main.py                  # Single entry point — run this
├── requirements.txt
├── README.md
│
└── src/
    ├── data_loader.py       # DuckDB ingestion + SQL aggregations
    ├── features.py          # All feature engineering (vectorised)
    └── models.py            # CatBoost, LightGBM, XGBoost training + ensemble
```

---

## How to Run

### On Google Colab (recommended — requires GPU)

1. Upload your competition CSV files to the Colab environment.
2. Clone or upload this repository.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python main.py
   ```

The script writes `submission.csv` in the current directory. Upload that to Kaggle.

### Locally (CPU only — no GPU)

If you don't have a GPU, change the device settings in `src/models.py`:

- **CatBoost**: Change `task_type="GPU"` → `task_type="CPU"`
- **LightGBM**: Remove `device="gpu"`
- **XGBoost**: Change `device="cuda"` → remove the parameter (defaults to CPU)

Training will be 5–10× slower but will produce identical results.

---

## Required Data Files

Place these in the same directory as `main.py` before running:

| File | Description |
|---|---|
| `failure_data_masked.csv` | Historical wheel failures (training set) |
| `failure_data_masked_test.csv` | Q1 2025 records for prediction (submission set) |
| `mileage_data_masked.csv` | Monthly mileage per equipment |
| `thd_data_masked.csv` | Truck Hunting Detector readings |
| `wild_data_masked.csv` | Wheel Impact Load Detector readings |
| `wpd_data_masked.csv` | Wheel Profile Detector readings |

All files are provided through the Kaggle competition page.

---

## Approach

### Data Pipeline

1. **DuckDB ingestion**: CSVs are loaded into an in-memory DuckDB database. Aggregating thousands of per-pass sensor readings to monthly per-wheel summaries is done in SQL — it is much faster than doing it in pandas.

2. **Train + test combined early**: `failure_data_masked_test.csv` is appended to the end of the training data *before* feature engineering. This is necessary for rolling window features (mileage lags, detector deltas) to see history when computing values for Q1 2025 rows. Without this, the first month of the test period would have no lag features.

3. **Feature engineering** (see `src/features.py`):
   - `days_since_applied`: How long the wheel has been installed
   - `mileage_intensity`: Miles per day, a wear-rate proxy
   - `lag_partmileage_1` / `travelled_since_lastmonth`: Month-over-month mileage
   - `partmileage_3monthavg`: 3-month rolling average of monthly travel
   - `delta_flangeheight`, `delta_flangethickness`, etc.: Rate of dimensional change per month
   - Cyclic sine/cosine month encoding: Prevents the model treating January and December as distant

4. **Chronological split**: No data leakage. The model trains on records before October 2024, validates on October–December 2024, and predicts on Q1 2025.

### Ensemble Design

Three GPU-accelerated models are trained and blended:

| Model | Weight | Why |
|---|---|---|
| XGBoost | **60%** | Tuned hyperparameters match the best-scoring baseline |
| CatBoost | 20% | Handles categoricals natively, provides diversity |
| LightGBM | 20% | Fast GPU training, adds independent gradient boosting perspective |

**Key design decisions:**

- **No feature scaling**: Tree models split on rank order, not magnitude. Applying `RobustScaler` to test data fitted on training data introduces distribution skew and hurts calibration on the Q1 2025 test period.
- **No `compute_sample_weight`**: Explicit per-sample weights inflated rare-class probabilities, causing high prediction entropy and log-loss penalty. LightGBM's `class_weight='balanced'` is a milder and better-calibrated alternative.
- **`mlogloss` as eval metric for XGBoost**: This directly optimises the competition metric, so early stopping fires at the optimal point instead of at a proxy like AUC.

### Handling Duplicate Rows

`failure_data_masked_test.csv` contains 12 exactly identical rows. The deduplication step removes them from the model's input to avoid training on repeated observations. When building the final submission (which must be exactly 55,330 rows), those 12 gaps are restored by forward-filling from their neighbours, which is valid since they are true duplicates.

---

## Results

| Method | Kaggle Log-Loss |
|---|---|
| Single XGBoost (baseline) | 0.03701 |
| 3-model ensemble (this repo) | **0.0299** |

---

## Reproducing Results Exactly

Set `random_state=42` and `random_seed=42` are already in place in all three models. The only source of non-determinism is GPU order-of-operations in CatBoost and LightGBM, which may cause ±0.001 variation across runs.

If exact bit-for-bit reproducibility is needed, switch all three models to CPU.

---

## Dependencies

- Python 3.10+
- See `requirements.txt` for package versions

All packages install cleanly via `pip install -r requirements.txt` with no system-level dependencies outside of Python.
