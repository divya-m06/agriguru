# Model 2: Plant Disease Classification (PlantVillage)

Production-grade, reproducible computer vision training and inference pipeline for plant disease diagnosis on the PlantVillage dataset (38 classes across 14 crops).

---

## 1. Problem Formulation & Agronomic Context

Plant disease classification categorizes RGB photographs of crop leaves into 38 disease categories (or healthy).
- **Backbone Architectures Evaluated:** MobileNetV2 (lightweight 3.4M), EfficientNetB0 (primary 5.3M), and ResNet50 (optional 25M).
- **Two-Phase Transfer Learning:** Phase 1 with frozen backbone at $10^{-3}$ lr; Phase 2 unfreezes top 35% layers at $3 \times 10^{-5}$ lr.
- **Embedded Preprocessing:** All architecture-specific normalizations are embedded directly into the saved Keras graph. FastAPI accepts raw RGB images `[0, 255]`.
- **Zero-Leakage Group Splitting:** Near-identical leaves in PlantVillage are clustered using Perceptual Hashing (pHash) and Union-Find. Clusters are split group-aware across train (70%), val (15%), and test (15%).
- **Calibrated Uncertainty:** Post-hoc temperature scaling minimizes Expected Calibration Error (ECE), returning `uncertain: true` for out-of-distribution or low-confidence samples.
- **Explainability:** Grad-CAM saliency heatmaps visualize model focus directly over leaf lesions.

---

## 2. Dataset Setup (PlantVillage)

Download using the Kaggle CLI:
```bash
# Set your Kaggle credentials (or place kaggle.json in ~/.kaggle/)
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/disease_detection/ --unzip
```
Ensure the folder structure contains the color images (e.g., `data/disease_detection/color/` or `data/disease_detection/PlantVillage/`).
Grayscale and segmented variants are automatically ignored by `data.py`.

---

## 3. How to Run Training

### Quick Smoke Test (Local or Colab)
Verify the pipeline on a small subsample in < 1 minute:
```bash
python ml/disease_detection/train.py --quick
```

### Full Production Training (Google Colab with GPU)
```bash
python ml/disease_detection/train.py \
    --data-dir data/disease_detection \
    --output-dir reports/disease_detection \
    --artifacts-dir artifacts/disease_detection \
    --backup-dir /content/drive/MyDrive/agri_disease_checkpoints \
    --seed 42
```

To run a specific architecture:
```bash
python ml/disease_detection/train.py --architecture EfficientNetB0
```

To resume after a Colab timeout or disconnection:
```bash
python ml/disease_detection/train.py \
    --data-dir data/disease_detection \
    --output-dir reports/disease_detection \
    --artifacts-dir artifacts/disease_detection \
    --backup-dir /content/drive/MyDrive/agri_disease_checkpoints \
    --resume
```

---

## 4. Google Colab Training Notebook

A thin Google Colab notebook is provided at [`ml/disease_detection/notebooks/colab_disease_training.ipynb`](notebooks/colab_disease_training.ipynb).
It mounts Google Drive, pulls the latest code from GitHub, checks GPU availability, downloads the dataset using Kaggle API secrets, executes `train.py`, and syncs checkpoints and artifacts back to Google Drive.

---

## 5. Standalone FastAPI Inference Example

```python
from pathlib import Path
from ml.disease_detection.src.inference import load_model, predict_image

# 1. Load model once during FastAPI startup
predictor = load_model("artifacts/disease_detection")

# 2. Predict on incoming image bytes
with open("sample_leaf.jpg", "rb") as f:
    image_bytes = f.read()

result = predictor.predict(image_bytes, top_k=3, return_gradcam=True)

print("Predicted Crop:", result["predicted_crop"])
print("Predicted Disease:", result["predicted_disease"])
print("Calibrated Confidence:", result["confidence"])
print("Is Uncertain:", result["uncertain"])
print("Top 3 Predictions:", result["top_predictions"])
# result["gradcam_base64"] contains JPEG base64 string of the heatmap overlay
```

---

## 6. Critical Limitations

1. **Laboratory Bias:** PlantVillage consists of single leaves against uniform laboratory backgrounds with controlled lighting. Accuracy on field photographs with complex backgrounds and shadows may be lower (domain shift).
2. **No 'Unknown / Non-Leaf' Class:** The dataset does not contain random non-leaf images. The model utilizes temperature-scaled confidence thresholds to flag uncertain predictions, but cannot substitute for expert agricultural lab pathology.
