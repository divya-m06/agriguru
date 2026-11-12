# Fertilizer Recommendation (Model 4 - Small Tabular Multiclass Classification)

Production-quality, reproducible training and inference pipeline recommending top-3 fertilizers with calibrated probabilities based on environmental conditions, soil classification, crop type, and soil N-P-K nutrient contents. Designed for direct serving via FastAPI.

---

## 1. Problem Formulation & Agronomic Context

- **Task:** Multiclass classification recommending optimal fertilizer blends across 7 commercial formulations:
  `Urea`, `DAP`, `14-35-14`, `28-28`, `17-17-17`, `20-20`, `10-26-26`.
- **Input Features:**
  - `temperature` (°C)
  - `humidity` (%)
  - `moisture` (%)
  - `soil_type` (`Sandy`, `Loamy`, `Black`, `Red`, `Clayey`)
  - `crop_type` (`Maize`, `Sugarcane`, `Cotton`, `Tobacco`, `Paddy`, `Barley`, `Wheat`, `Millets`, `Oil seeds`, `Pulses`, `Ground Nuts`)
  - `nitrogen` (N ratio/ppm in soil)
  - `potassium` (K ratio/ppm in soil)
  - `phosphorous` (P ratio/ppm in soil)
- **Small Dataset & Methodological Reality ($N = 99$):**
  - With only 99 rows across 7 classes (smallest class count is 7), single train/test holdouts exhibit extreme statistical noise (~5% shift per misclassified instance).
  - High reported scores in internet literature (e.g., 100% test accuracy) reflect small-sample memorization rather than robust agronomic generalization.
  - To ensure honest generalization, this pipeline replaces fixed splits with **Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 evaluations)** and **Nested CV (outer 5x3 repeats = 15 folds, inner 3 folds)** with 95% confidence intervals.
  - A **200-iteration label permutation test** proves the chosen model beats random chance with statistical significance ($p < 0.01$).

---

## 2. Dataset Acquisition

### Option A: Automatic Ingestion (Default)
`train.py` automatically checks for `data/fertilizer_recommendation/Fertilizer Prediction.csv`. If absent, it downloads directly from verified GitHub mirrors:
- `https://raw.githubusercontent.com/mohansaidinesh/Automated-Farming/master/Fertilizer%20Prediction.csv`
- `https://raw.githubusercontent.com/jayeshpandav/Hackspiration_cultivo/main/Fertilizer%20Prediction.csv`

### Option B: Kaggle CLI
```bash
kaggle datasets download -d gdabhishek/fertilizer-prediction -p data/fertilizer_recommendation/ --unzip
```

### Option C: Manual Download
Download `Fertilizer Prediction.csv` from Kaggle or the repository mirror and place it at:
`data/fertilizer_recommendation/Fertilizer Prediction.csv`.

---

## 3. Execution Instructions

### A. Quick Smoke Test (< 15 seconds)
Executes data ingestion, schema validation, 2-fold repeated CV, stage caching, nested tuning, permutation testing, diagnostic plotting, and artifact export:
```bash
python ml/fertilizer_recommendation/train.py --quick --output-dir reports/fertilizer_recommendation_quick
```

### B. Full Production Training
Runs 25-fold Repeated Stratified CV across all 11 candidate models, 15-fold outer nested CV hyperparameter tuning, 200-iteration label permutation test, diagnostic learning curves, out-of-fold confusion matrix, and artifact export:
```bash
python ml/fertilizer_recommendation/train.py \
  --data-dir data/fertilizer_recommendation \
  --output-dir reports/fertilizer_recommendation \
  --artifacts-dir artifacts/fertilizer_recommendation \
  --seed 42 \
  --n-trials 20
```

CLI Parameters:
- `--data-dir`: Directory for dataset and split CSVs (default: `data/fertilizer_recommendation`).
- `--output-dir`: Destination for logs, comparison CSVs, plots, and `REPORT.md` (default: `reports/fertilizer_recommendation`).
- `--artifacts-dir`: Destination for serialised joblib and schema JSONs (default: `artifacts/fertilizer_recommendation`).
- `--config`: Configuration file path (default: `ml/fertilizer_recommendation/config.yaml`).
- `--seed`: Deterministic integer seed (default: 42).
- `--quick`: Rapid smoke test flag.
- `--n-trials`: Number of hyperparameter search trials (default: 20).

*Stage caching (`reports/fertilizer_recommendation/cache/`) ensures any re-run immediately skips finished work.*

### C. Run Test Suite
```bash
pytest ml/fertilizer_recommendation/tests/test_inference.py -v
```

---

## 4. Exported Production Artifacts

Artifacts are saved under `artifacts/fertilizer_recommendation/`:
```
artifacts/fertilizer_recommendation/
├── fertilizer_model.joblib   # Self-contained fitted Pipeline (preprocessor + classifier)
├── classes.json              # List of 7 fertilizer class names ordered by integer label
├── feature_order.json        # Expected input feature names and column order
├── allowed_categories.json   # Known soil types (5) and crop types (11)
├── numeric_ranges.json       # Min, max, mean, std bounds from training distribution
└── metadata.json             # Git commit, library versions, CV metrics, and vocabulary note
```

---

## 5. FastAPI Integration Example

Backend engineers can serve top-3 fertilizer recommendations with probabilities using `ml/fertilizer_recommendation/inference.py`:

```python
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ml.fertilizer_recommendation.inference import load_model, predict_fertilizer

app = FastAPI(title="Fertilizer Recommendation Service")

# Load model once at startup
predictor = load_model(Path("artifacts/fertilizer_recommendation"))

class FertilizerRecommendationItem(BaseModel):
    fertilizer: str
    probability: float
    confidence_pct: str

class FertilizerRequest(BaseModel):
    temperature: float = Field(..., example=26.0)
    humidity: float = Field(..., ge=0, le=100, example=52.0)
    moisture: float = Field(..., ge=0, example=38.0)
    soil_type: str = Field(..., example="Sandy")
    crop_type: str = Field(..., example="Maize")
    nitrogen: float = Field(..., ge=0, example=37.0)
    potassium: float = Field(..., ge=0, example=0.0)
    phosphorous: float = Field(..., ge=0, example=0.0)
    k: int = Field(3, ge=1, le=7)

class FertilizerResponse(BaseModel):
    recommendations: list[FertilizerRecommendationItem]
    top_fertilizer: str
    top_confidence: float

@app.post("/predict-fertilizer", response_model=FertilizerResponse)
def recommend_fertilizer(req: FertilizerRequest):
    try:
        result = predict_fertilizer(
            temperature=req.temperature,
            humidity=req.humidity,
            moisture=req.moisture,
            soil_type=req.soil_type,
            crop_type=req.crop_type,
            nitrogen=req.nitrogen,
            potassium=req.potassium,
            phosphorous=req.phosphorous,
            k=req.k,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```
