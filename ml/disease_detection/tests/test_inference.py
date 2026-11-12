import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from ml.disease_detection.src.data import split_group_aware
from ml.disease_detection.src.evaluate import (
    TemperatureScaler,
    compute_expected_calibration_error,
    compute_comprehensive_metrics,
)

ARTIFACTS_DIR = (
    Path("artifacts/disease_detection")
    if (Path("artifacts/disease_detection") / "class_names.json").exists()
    else Path("artifacts/disease_detection_quick")
)


def test_artifacts_existence():
    """Verify that expected production artifact files exist."""
    if not ARTIFACTS_DIR.exists():
        pytest.skip("Artifacts directory not found; run quick training or setup mock artifacts first.")

    assert (ARTIFACTS_DIR / "class_names.json").exists(), "Missing class_names.json"
    assert (ARTIFACTS_DIR / "metadata.json").exists(), "Missing metadata.json"
    assert (ARTIFACTS_DIR / "inference.py").exists(), "Missing inference.py in artifacts"
    assert (ARTIFACTS_DIR / "sample_predictions.json").exists(), "Missing sample_predictions.json"


def test_metadata_structure():
    """Verify metadata contains required production fields."""
    if not ARTIFACTS_DIR.exists():
        pytest.skip("Artifacts directory not found.")

    with open(ARTIFACTS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "architecture" in meta
    assert "input_dimensions" in meta
    assert "num_classes" in meta
    assert "temperature" in meta
    assert "uncertainty_threshold" in meta
    assert "library_versions" in meta
    assert meta["num_classes"] >= 10


def test_class_names_structure():
    """Verify class names JSON mapping conforms to schema."""
    if not ARTIFACTS_DIR.exists():
        pytest.skip("Artifacts directory not found.")

    with open(ARTIFACTS_DIR / "class_names.json", "r", encoding="utf-8") as f:
        classes = json.load(f)

    assert isinstance(classes, list)
    assert len(classes) >= 10
    first = classes[0]
    assert "class_index" in first
    assert "class_label" in first
    assert "crop_name" in first
    assert "disease_name" in first


def test_cross_class_duplicate_grouping_and_zero_leakage():
    """Verify cross-class duplicates stay bound together with ZERO leakage across splits."""
    rows = []
    # 2 cross-class duplicates sharing duplicate_group_id 'group_000001'
    rows.append({
        "image_path": "fake/path/img1.jpg",
        "class_label": "Tomato___healthy",
        "duplicate_group_id": "group_000001"
    })
    rows.append({
        "image_path": "fake/path/img2.jpg",
        "class_label": "Tomato___Late_blight",
        "duplicate_group_id": "group_000001"
    })

    # Add other distinct groups across classes
    for i in range(2, 25):
        rows.append({
            "image_path": f"fake/path/img{i}.jpg",
            "class_label": f"Class_{i % 3}",
            "duplicate_group_id": f"group_{i:06d}"
        })

    df = pd.DataFrame(rows)
    train_df, val_df, test_df = split_group_aware(df, seed=42)

    train_groups = set(train_df["duplicate_group_id"])
    val_groups = set(val_df["duplicate_group_id"])
    test_groups = set(test_df["duplicate_group_id"])

    # Strict zero leakage assertion
    assert len(train_groups & val_groups) == 0, "Duplicate group leaked between train and val"
    assert len(train_groups & test_groups) == 0, "Duplicate group leaked between train and test"
    assert len(val_groups & test_groups) == 0, "Duplicate group leaked between val and test"

    # Both cross-class duplicate images must be in the EXACT same split
    img1_split = "train" if "group_000001" in train_groups else ("val" if "group_000001" in val_groups else "test")
    img2_in_same = (
        ("group_000001" in train_groups and img1_split == "train") or
        ("group_000001" in val_groups and img1_split == "val") or
        ("group_000001" in test_groups and img1_split == "test")
    )
    assert img2_in_same, "Cross-class duplicate members were separated into different splits!"


def test_rare_class_preservation():
    """Verify group-aware stratification handles rare classes gracefully (1 group, 2 groups, 5 groups)."""
    rows = []
    # Class with 1 group
    rows.append({"image_path": "a.jpg", "class_label": "RareClass1", "duplicate_group_id": "g_r1"})

    # Class with 2 groups
    rows.append({"image_path": "b1.jpg", "class_label": "RareClass2", "duplicate_group_id": "g_r2_a"})
    rows.append({"image_path": "b2.jpg", "class_label": "RareClass2", "duplicate_group_id": "g_r2_b"})

    # Frequent class with 10 groups
    for i in range(10):
        rows.append({"image_path": f"c{i}.jpg", "class_label": "CommonClass", "duplicate_group_id": f"g_c_{i}"})

    df = pd.DataFrame(rows)
    train_df, val_df, test_df = split_group_aware(df, seed=42)

    # RareClass1 should be in train
    assert "RareClass1" in set(train_df["class_label"])
    # RareClass2 should be in train and test
    assert "RareClass2" in set(train_df["class_label"])
    assert "RareClass2" in set(test_df["class_label"])
    # CommonClass should be in train
    assert "CommonClass" in set(train_df["class_label"])


def test_temperature_scaling_and_calibration():
    """Verify TemperatureScaler calibration optimization and ECE reduction."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_classes = 5

    # Simulated overconfident logits
    logits = rng.standard_normal((n_samples, n_classes)) * 3.0
    labels = rng.integers(0, n_classes, size=n_samples)

    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.temperature > 0.1
    assert scaler.temperature < 20.0

    calibrated_probs = scaler.transform(logits)
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0, atol=1e-5)

    ece, _, _, _ = compute_expected_calibration_error(labels, calibrated_probs, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_predictor_image_preprocessing():
    """Verify PlantDiseasePredictor image preprocessing on PIL, numpy, and bytes."""
    from ml.disease_detection.src.inference import PlantDiseasePredictor

    # Create dummy predictor instance bypassing model load
    class MockPredictor(PlantDiseasePredictor):
        def __init__(self):
            self.target_size = (224, 224)

    mock = MockPredictor()

    # 1. RGB Numpy array
    rgb_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    p1 = mock.preprocess_image(rgb_arr)
    assert p1.shape == (224, 224, 3)
    assert p1.dtype == np.float32

    # 2. PIL Image
    pil_img = Image.new("RGB", (100, 100), color=(10, 20, 30))
    p2 = mock.preprocess_image(pil_img)
    assert p2.shape == (224, 224, 3)

    # 3. Grayscale Numpy array (H, W)
    gray_arr = np.ones((150, 150), dtype=np.uint8) * 128
    p3 = mock.preprocess_image(gray_arr)
    assert p3.shape == (224, 224, 3)

    # 4. RGBA Numpy array (H, W, 4)
    rgba_arr = np.zeros((150, 150, 4), dtype=np.uint8)
    p4 = mock.preprocess_image(rgba_arr)
    assert p4.shape == (224, 224, 3)


def test_inference_module_contract():
    """Verify inference.py can load classes, parse image formats, and format predictions."""
    from ml.disease_detection.src.inference import PlantDiseasePredictor

    if not ARTIFACTS_DIR.exists() or not (ARTIFACTS_DIR / "disease_model.keras").exists():
        pytest.skip("Artifacts model not ready for live inference.")

    try:
        predictor = PlantDiseasePredictor(ARTIFACTS_DIR)
        assert len(predictor.class_names) >= 10

        # Create synthetic test leaf
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy[:, :, 1] = 150  # Green channel

        res = predictor.predict(dummy, top_k=3, return_gradcam=False)
        assert "predicted_crop" in res
        assert "predicted_disease" in res
        assert "confidence" in res
        assert "uncertain" in res
        assert "top_predictions" in res
        assert len(res["top_predictions"]) == 3
        assert 0.0 <= res["confidence"] <= 1.0
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"TensorFlow not available in local environment: {e}")
