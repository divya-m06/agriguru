import json
from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.crop_recommendation.inference import CropRecommendationPredictor, load_model, predict, predict_top_k

ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "crop_recommendation"
DATA_SPLITS_DIR = Path(__file__).resolve().parents[3] / "data" / "crop_recommendation" / "splits"


@pytest.fixture(scope="module")
def predictor():
    """Load predictor fixture from disk."""
    if not (ARTIFACTS_DIR / "crop_model.joblib").exists():
        pytest.skip(f"Artifacts not yet generated at {ARTIFACTS_DIR}. Run train.py first.")
    return CropRecommendationPredictor(ARTIFACTS_DIR)


def test_artifacts_existence():
    """Verify that all required production artifacts are present on disk."""
    assert (ARTIFACTS_DIR / "crop_model.joblib").exists(), "Missing crop_model.joblib"
    assert (ARTIFACTS_DIR / "label_encoder.joblib").exists(), "Missing label_encoder.joblib"
    assert (ARTIFACTS_DIR / "feature_order.json").exists(), "Missing feature_order.json"
    assert (ARTIFACTS_DIR / "input_ranges.json").exists(), "Missing input_ranges.json"
    assert (ARTIFACTS_DIR / "classes.json").exists(), "Missing classes.json"
    assert (ARTIFACTS_DIR / "metadata.json").exists(), "Missing metadata.json"


def test_predictor_initialization(predictor):
    """Test predictor correctly unpacks classes and feature schema."""
    assert len(predictor.classes) == 22, f"Expected 22 classes, got {len(predictor.classes)}"
    assert len(predictor.feature_order) == 7, f"Expected 7 features, got {len(predictor.feature_order)}"
    assert set(predictor.feature_order) == {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"}


def test_single_prediction(predictor):
    """Test top-1 prediction for typical rice-growing environmental conditions."""
    rice_input = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    result = predictor.predict(rice_input)
    assert isinstance(result, str)
    assert result in predictor.classes
    assert result in ["rice", "jute"]


def test_predict_top_k(predictor):
    """Test top-k prediction output format, ordering, and probability properties."""
    sample_input = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    top_3 = predictor.predict_top_k(sample_input, k=3)
    assert len(top_3) == 3, f"Expected 3 predictions, got {len(top_3)}"

    # Check structure
    for crop, prob in top_3:
        assert isinstance(crop, str)
        assert crop in predictor.classes
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    # Probabilities must be monotonically descending
    probs = [prob for _, prob in top_3]
    assert probs == sorted(probs, reverse=True), "Top-k probabilities are not sorted in descending order"


def test_missing_features_error(predictor):
    """Test that missing required features raises a descriptive ValueError."""
    incomplete_input = {
        "N": 90.0,
        "P": 42.0,
        # missing K
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    with pytest.raises(ValueError, match="Input is missing required features:.*K"):
        predictor.predict(incomplete_input)


def test_out_of_range_error(predictor):
    """Test that out-of-physical-range values trigger validation errors."""
    abnormal_input = {
        "N": 900.0,  # Extreme impossible value
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    with pytest.raises(ValueError, match="out of physical plausible range"):
        predictor.predict(abnormal_input)


def test_fastapi_interface_helpers():
    """Test module-level load_model, predict, and predict_top_k helper functions."""
    pred = load_model(ARTIFACTS_DIR)
    sample_input = {
        "N": 50.0,
        "P": 50.0,
        "K": 50.0,
        "temperature": 25.0,
        "humidity": 70.0,
        "ph": 6.5,
        "rainfall": 100.0,
    }
    top_1 = predict(sample_input)
    assert isinstance(top_1, str)
    top_3 = predict_top_k(sample_input, k=3)
    assert len(top_3) == 3


def test_sample_reproducibility(predictor):
    """Verify test metrics reproduction on held-out test split."""
    test_csv = DATA_SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        pytest.skip(f"Test split CSV not found at {test_csv}")

    df_test = pd.read_csv(test_csv)
    X_test = df_test[predictor.feature_order]
    y_test = df_test["label"]

    # Predict on test split
    preds = predictor.predict(X_test)
    accuracy = np.mean(np.array(preds) == np.array(y_test))

    # Load recorded test accuracy from metadata.json
    with open(ARTIFACTS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    expected_acc = meta["test_metrics"]["accuracy"]

    # Reproducibility check within tight 1e-4 tolerance
    assert abs(accuracy - expected_acc) < 1e-4, (
        f"Inference accuracy {accuracy:.4f} diverged from metadata recorded accuracy {expected_acc:.4f}"
    )
