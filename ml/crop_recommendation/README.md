# Production Crop Recommendation Model

Tabular multiclass classification system recommending optimal agricultural crops based on N, P, K soil ratios, temperature, relative humidity, soil pH, and rainfall.

The codebase is built strictly for production-grade reproducibility, zero-leakage cross-validation, and seamless serving integration with FastAPI.

---

## 1. Dataset

- **Source:** Kaggle [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) by Atharva Ingle.
- **File:** `Crop_recommendation.csv` (2,200 rows, 8 columns: 7 continuous features + 1 target label `label`).
- **Classes (22 crops):** `rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`.

### Download Options

#### Option A: Automatic Ingestion (Default)
If `Crop_recommendation.csv` is not present in `data/crop_recommendation/`, `train.py` will automatically download it from the verified mirror and validate checksums.

#### Option B: Kaggle CLI
```bash
kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data/crop_recommendation --unzip
```

#### Option C: Manual Download
1. Download `Crop_recommendation.csv` from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
2. Place the file at `data/crop_recommendation/Crop_recommendation.csv`.

---

## 2. Environment Setup

Python 3.10 or 3.11 is required.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install strictly pinned dependencies
pip install -r requirements.txt
```

---

## 3. Training & Evaluation

### Quick Smoke Test (Fast verification on subsample)
```bash
python ml/crop_recommendation/train.py --quick --output-dir reports/crop_recommendation_quick
```

### Full Production Training Run
```bash
python ml/crop_recommendation/train.py \
  --data-dir data/crop_recommendation \
  --output-dir reports/crop_recommendation \
  --artifacts-dir artifacts/crop_recommendation \
  --seed 42
```

### Command Line Options
- `--data-dir`: Directory storing or receiving `Crop_recommendation.csv` (default: `data/crop_recommendation`).
- `--output-dir`: Destination for reports, plots, and CSV logs (default: `reports/crop_recommendation`).
- `--artifacts-dir`: Destination for deployable joblib/json artifacts (default: `artifacts/crop_recommendation`).
- `--config`: Path to `config.yaml` (default: `ml/crop_recommendation/config.yaml`).
- `--seed`: Integer seed override for deterministic execution across all libraries.
- `--quick`: Subsamples dataset to 10 rows per class and runs streamlined CV for smoke-testing.

---

## 4. Exported Artifacts

Saved in `artifacts/crop_recommendation/`:
- `crop_model.joblib`: The fitted scikit-learn Pipeline (feature transformer + optional scaler + classifier).
- `label_encoder.joblib`: Fitted LabelEncoder for inverse mapping class indices to crop strings.
- `classes.json`: Ordered list of 22 crop names.
- `feature_order.json`: Strict input column sequence (`["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]`).
- `input_ranges.json`: Physical min/max/mean/std per feature from the training split for boundary validation.
- `metadata.json`: Complete provenance metadata (git commit hash, Python version, library versions, test metrics, noise robustness scores, and timestamp).

---

## 5. Inference Usage (FastAPI Ready)

`inference.py` exposes zero-dependency functions that can be imported directly into a FastAPI route:

```python
from ml.crop_recommendation.inference import load_model, predict_top_k

# 1. Initialize once on startup
predictor = load_model("artifacts/crop_recommendation")

# 2. Sample input payload
payload = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
}

# 3. Top-3 recommendations with probabilities
recommendations = predict_top_k(payload, k=3)
print(recommendations)
# [('rice', 0.98), ('jute', 0.015), ('papaya', 0.005)]
```

### Input Validation
The predictor checks:
- Required column presence (fails loudly with missing features).
- Numeric casting and NaN checks.
- Out-of-bounds physical domain validation (with 20% margin).

---

## 6. Running Inference Tests

Verify disk artifacts in a fresh process:
```bash
pytest ml/crop_recommendation/tests/test_inference.py -v
```
