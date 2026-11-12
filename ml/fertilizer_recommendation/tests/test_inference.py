from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.fertilizer_recommendation.inference import (
    FertilizerPredictor,
    load_model,
    predict_fertilizer,
)

ARTIFACTS_DIR = Path("artifacts/fertilizer_recommendation")
SPLITS_DIR = Path("data/fertilizer_recommendation/splits")


def test_artifacts_existence():
    """Verify that all production artifacts exist in the artifacts export directory."""
    expected_files = [
        "fertilizer_model.joblib",
        "classes.json",
        "feature_order.json",
        "allowed_categories.json",
        "numeric_ranges.json",
        "metadata.json",
    ]
    for fname in expected_files:
        artifact_path = ARTIFACTS_DIR / fname
        assert artifact_path.exists(), f"Missing required artifact: {artifact_path}"


def test_predictor_initialization():
    """Verify that predictor properly initializes categories, classes, and schema."""
    predictor = FertilizerPredictor(ARTIFACTS_DIR)
    assert len(predictor.classes) == 7, f"Expected 7 fertilizer classes, got {len(predictor.classes)}"
    assert "Urea" in predictor.classes
    assert "DAP" in predictor.classes
    assert len(predictor.valid_soils) == 5
    assert len(predictor.valid_crops) == 11
    assert len(predictor.feature_order) == 8


def test_single_prediction_top_k():
    """Verify top-3 recommendation format, descending probabilities, and probability summation."""
    predictor = FertilizerPredictor(ARTIFACTS_DIR)
    soil = list(predictor.allowed_categories["soil_types"])[0]
    crop = list(predictor.allowed_categories["crop_types"])[0]

    result = predictor.predict_top_k(
        temperature=26.0,
        humidity=52.0,
        moisture=38.0,
        soil_type=soil,
        crop_type=crop,
        nitrogen=37.0,
        potassium=0.0,
        phosphorous=0.0,
        k=3,
    )

    assert "recommendations" in result
    recs = result["recommendations"]
    assert len(recs) == 3

    # Probabilities must be descending and valid
    probs = [r["probability"] for r in recs]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs[0] >= probs[1] >= probs[2]

    # Top fertilizer matches first recommendation
    assert result["top_fertilizer"] == recs[0]["fertilizer"]
    assert np.isclose(result["top_confidence"], recs[0]["probability"])


def test_unknown_soil_type_error():
    """Verify that an unobserved soil type raises a descriptive ValueError."""
    predictor = FertilizerPredictor(ARTIFACTS_DIR)
    crop = list(predictor.allowed_categories["crop_types"])[0]

    with pytest.raises(ValueError, match="Unknown soil type 'Volcanic'"):
        predictor.predict_top_k(
            temperature=26.0,
            humidity=52.0,
            moisture=38.0,
            soil_type="Volcanic",
            crop_type=crop,
            nitrogen=30.0,
            potassium=10.0,
            phosphorous=10.0,
        )


def test_unknown_crop_type_error():
    """Verify that an unobserved crop type raises a descriptive ValueError."""
    predictor = FertilizerPredictor(ARTIFACTS_DIR)
    soil = list(predictor.allowed_categories["soil_types"])[0]

    with pytest.raises(ValueError, match="Unknown crop type 'DragonFruit'"):
        predictor.predict_top_k(
            temperature=26.0,
            humidity=52.0,
            moisture=38.0,
            soil_type=soil,
            crop_type="DragonFruit",
            nitrogen=30.0,
            potassium=10.0,
            phosphorous=10.0,
        )


def test_negative_physical_inputs_error():
    """Verify that negative nutrient quantities or out-of-range humidity raise ValueError."""
    predictor = FertilizerPredictor(ARTIFACTS_DIR)
    soil = list(predictor.allowed_categories["soil_types"])[0]
    crop = list(predictor.allowed_categories["crop_types"])[0]

    with pytest.raises(ValueError, match="Nitrogen cannot be negative"):
        predictor.predict_top_k(
            temperature=26.0,
            humidity=52.0,
            moisture=38.0,
            soil_type=soil,
            crop_type=crop,
            nitrogen=-5.0,
            potassium=10.0,
            phosphorous=10.0,
        )

    with pytest.raises(ValueError, match="Humidity must be in"):
        predictor.predict_top_k(
            temperature=26.0,
            humidity=150.0,
            moisture=38.0,
            soil_type=soil,
            crop_type=crop,
            nitrogen=20.0,
            potassium=10.0,
            phosphorous=10.0,
        )


def test_module_convenience_functions():
    """Verify module-level load_model and predict_fertilizer for FastAPI integration."""
    loaded_pred = load_model(ARTIFACTS_DIR)
    assert loaded_pred is not None

    soil = list(loaded_pred.allowed_categories["soil_types"])[0]
    crop = list(loaded_pred.allowed_categories["crop_types"])[0]

    res = predict_fertilizer(
        temperature=28.0,
        humidity=60.0,
        moisture=35.0,
        soil_type=soil,
        crop_type=crop,
        nitrogen=25.0,
        potassium=15.0,
        phosphorous=20.0,
        k=3,
    )
    assert "recommendations" in res
    assert len(res["recommendations"]) == 3


def test_sample_reproducibility():
    """Verify reproducible inference across sanity holdout samples if manifest exists."""
    holdout_file = SPLITS_DIR / "holdout_test.csv"
    if holdout_file.exists():
        df_holdout = pd.read_csv(holdout_file)
        predictor = FertilizerPredictor(ARTIFACTS_DIR, strict_range_check=False)

        for _, row in df_holdout.head(5).iterrows():
            res = predictor.predict_top_k(
                temperature=row["temperature"],
                humidity=row["humidity"],
                moisture=row["moisture"],
                soil_type=row["soil_type"],
                crop_type=row["crop_type"],
                nitrogen=row["nitrogen"],
                potassium=row["potassium"],
                phosphorous=row["phosphorous"],
                k=3,
            )
            assert len(res["recommendations"]) == 3
            assert res["recommendations"][0]["probability"] >= res["recommendations"][1]["probability"]
