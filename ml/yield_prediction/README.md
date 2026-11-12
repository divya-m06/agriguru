# Crop Yield Prediction (Model 3 - Tabular Regression)

Production-quality, reproducible training and inference pipeline predicting crop yields in hectograms per hectare (`hg/ha`) and metric tonnes per hectare (`t/ha`). Designed for seamless integration into FastAPI services.

---

## 1. Problem Formulation & Agronomic Context

- **Task:** Predict expected crop yield for a given agricultural season based on crop identity, historical time trend, annual precipitation, pesticide consumption, and average temperature.
- **Unit Conversion:** 
  $$\text{Yield (t/ha)} = \frac{\text{Yield (hg/ha)}}{10,000}$$
- **Target Skewness Mitigation:** Crop yield spans orders of magnitude across crop families (e.g. Potatoes/Cassava at ~200,000–400,000 hg/ha vs. Cereals at ~20,000–50,000 hg/ha). The training pipeline wraps all regressors inside `TransformedTargetRegressor(func=log1p, inverse_func=expm1)` to eliminate heteroscedasticity and ensure strictly positive predictions.
- **Model Variants:**
  - **Variant A (User Spec):** `Item` (crop), `Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`. (Country `Area` excluded to assess crop climate potential without national bias).
  - **Variant B (Full Info):** Variant A + `Area` (country target-encoded). Captures national agricultural mechanization, subsidies, soil quality, and infrastructure.

---

## 2. Dataset Acquisition

The dataset contains 28,242 records spanning 1990 to 2013 across 101 countries and 10 staple crops.

### Option A: Automatic Ingestion (Default)
`train.py` automatically checks for `data/yield_prediction/yield_df.csv`. If absent, it downloads directly from the verified mirror:
```
https://raw.githubusercontent.com/Faissal-00/Crop-Yield-Prediction/master/yield_df.csv
```

### Option B: Kaggle CLI
```bash
kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data/yield_prediction/ --unzip
```

### Option C: Manual Download
1. Download `yield_df.csv` from Kaggle or the repository mirror.
2. Place the file at `data/yield_prediction/yield_df.csv`.

---

## 3. Strict Temporal Splitting & Leakage Prevention

> **CRITICAL**: Standard random cross-validation produces severe temporal data leakage ($R^2 \approx 0.98$) because it interpolates between years of known country-crop trajectories.

This pipeline enforces strict temporal splitting:
- **Training Set:** 1990 – 2008
- **Validation Set:** 2009 – 2010
- **Held-Out Test Set:** 2011 – 2013 (Evaluated exactly ONCE)
- **Model Selection:** Expanding-window temporal CV on training years.
- **Unseen Country Generalization:** Evaluated via `GroupKFold(n_splits=5, groups=Area)`.

---

## 4. Execution Instructions

### A. Quick Smoke Test (< 30 seconds)
Runs data ingestion, feature transformation, expanding-window CV, tuning, and evaluation on a fast subsample:
```bash
python ml/yield_prediction/train.py --quick --output-dir reports/yield_prediction_quick
```

### B. Full Production Training
Trains candidate regressor families across both variants (baselines, ElasticNet, KNN, and MLP on Variant A; tree models and linear models on Variant B with in-fold target/ordinal encoding), executes hyperparameter tuning, fits prediction interval quantiles, generates diagnostic plots, and exports artifacts.

Stage caching (`reports/yield_prediction/cache/`) automatically persists intermediate results per stage/variant/model, making training 100% resumable if interrupted.

```bash
python ml/yield_prediction/train.py \
  --data-dir data/yield_prediction \
  --output-dir reports/yield_prediction \
  --artifacts-dir artifacts/yield_prediction \
  --seed 42 \
  --export-variant B \
  --n-trials 20
```

CLI Parameters:
- `--data-dir`: Directory for dataset and split CSVs (default: `data/yield_prediction`).
- `--output-dir`: Destination for logs, comparison CSVs, plots, and `REPORT.md` (default: `reports/yield_prediction`).
- `--artifacts-dir`: Destination for serialised joblib and schema JSONs (default: `artifacts/yield_prediction`).
- `--config`: Configuration file path (default: `ml/yield_prediction/config.yaml`).
- `--seed`: Deterministic integer seed (default: 42).
- `--quick`: Rapid smoke test flag.
- `--n-trials`: Number of RandomizedSearchCV parameter configurations (default: 20).
- `--export-variant`: Primary variant to export in `artifacts_dir` (`B` recommended, `A` also saved in `variant_a/`).

### C. Run Test Suite
Verifies inference integrity, bounds checking, unit conversions, and extrapolation warnings:
```bash
pytest ml/yield_prediction/tests/test_inference.py -v
```

---

## 5. Exported Production Artifacts

Artifacts are written to `artifacts/yield_prediction/`:
```
artifacts/yield_prediction/
├── yield_model.joblib          # Scikit-learn Pipeline with TransformedTargetRegressor
├── quantile_low_model.joblib   # 10th percentile LightGBM quantile pipeline
├── quantile_high_model.joblib  # 90th percentile LightGBM quantile pipeline
├── feature_order.json          # Expected input feature names
├── valid_categories.json       # Valid crop types and country names
├── input_ranges.json           # Min, max, mean, std bounds from training distribution
├── metadata.json               # Git commit, metrics, and environment versions
└── variant_a/                  # Standalone Variant A artifacts (no country required)
```

---

## 6. FastAPI Integration Example

Backend engineers can serve predictions directly using `ml/yield_prediction/inference.py`:

```python
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ml.yield_prediction.inference import load_model, predict_yield

app = FastAPI(title="Crop Yield Prediction Service")

# Load model once at startup
predictor = load_model(Path("artifacts/yield_prediction"))

class YieldRequest(BaseModel):
    crop: str = Field(..., example="Maize")
    year: int = Field(..., example=2013)
    rainfall_mm: float = Field(..., ge=0, example=1200.0)
    pesticides_tonnes: float = Field(..., ge=0, example=45.0)
    temperature_c: float = Field(..., example=19.5)
    area: str = Field(None, example="India")

class YieldResponse(BaseModel):
    crop: str
    year: int
    yield_hg_ha: float
    yield_t_ha: float
    prediction_interval_80_hg_ha: list[float] | None = None
    prediction_interval_80_t_ha: list[float] | None = None

@app.post("/predict-yield", response_model=YieldResponse)
def get_crop_yield(req: YieldRequest):
    try:
        result = predict_yield(
            crop=req.crop,
            year=req.year,
            rainfall_mm=req.rainfall_mm,
            pesticides_tonnes=req.pesticides_tonnes,
            temperature_c=req.temperature_c,
            area=req.area,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```
