from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest

from ml.yield_prediction.inference import CropYieldPredictor, load_model, predict_yield

ARTIFACTS_DIR = (
    Path("artifacts/yield_prediction")
    if (Path("artifacts/yield_prediction") / "yield_model.joblib").exists()
    else Path("artifacts/yield_prediction_quick")
)
VARIANT_A_DIR = ARTIFACTS_DIR / "variant_a"
SPLITS_DIR = Path("data/yield_prediction/splits")


def test_artifacts_existence():
    """Verify that all core model artifacts exist in the production export directory."""
    expected_files = [
        "yield_model.joblib",
        "feature_order.json",
        "valid_categories.json",
        "input_ranges.json",
        "metadata.json",
    ]
    for fname in expected_files:
        artifact_path = ARTIFACTS_DIR / fname
        assert artifact_path.exists(), f"Missing required artifact: {artifact_path}"


def test_predictor_initialization():
    """Verify that predictor loads categories, ranges, and feature schema properly."""
    predictor = CropYieldPredictor(ARTIFACTS_DIR)
    assert len(predictor.valid_crops) >= 10, f"Expected at least 10 crops, found {len(predictor.valid_crops)}"
    assert "Maize" in predictor.valid_crops
    assert "Potatoes" in predictor.valid_crops
    assert "Wheat" in predictor.valid_crops
    assert len(predictor.feature_order) >= 5


def test_single_prediction_output_format():
    """Verify inference output format, positive yield guarantee, and unit conversion."""
    predictor = CropYieldPredictor(ARTIFACTS_DIR)
    crop = "Maize"
    year = 2013
    rainfall = 1000.0
    pesticides = 50.0
    temp = 18.0
    area = list(predictor.valid_areas)[0] if predictor.valid_areas else None

    res = predictor.predict_yield(
        crop=crop,
        year=year,
        rainfall_mm=rainfall,
        pesticides_tonnes=pesticides,
        temperature_c=temp,
        area=area,
    )

    assert "crop" in res and res["crop"] == crop
    assert "year" in res and res["year"] == year
    assert "yield_hg_ha" in res and res["yield_hg_ha"] >= 0.0
    assert "yield_t_ha" in res and res["yield_t_ha"] >= 0.0
    # Conversion check: 1 t/ha = 10,000 hg/ha
    assert np.isclose(res["yield_t_ha"], res["yield_hg_ha"] / 10000.0, atol=1e-3)


def test_prediction_intervals():
    """Verify that LightGBM quantile prediction intervals exist and obey low <= high."""
    predictor = CropYieldPredictor(ARTIFACTS_DIR)
    crop = "Potatoes"
    year = 2012
    rainfall = 1200.0
    pesticides = 100.0
    temp = 15.0
    area = list(predictor.valid_areas)[0] if predictor.valid_areas else None

    res = predictor.predict_yield(
        crop=crop,
        year=year,
        rainfall_mm=rainfall,
        pesticides_tonnes=pesticides,
        temperature_c=temp,
        area=area,
    )

    if predictor.quantile_low is not None and predictor.quantile_high is not None:
        assert "prediction_interval_80_hg_ha" in res
        low_hg, high_hg = res["prediction_interval_80_hg_ha"]
        assert low_hg >= 0.0
        assert high_hg >= low_hg

        assert "prediction_interval_80_t_ha" in res
        low_t, high_t = res["prediction_interval_80_t_ha"]
        assert low_t >= 0.0
        assert high_t >= low_t


def test_unknown_crop_error():
    """Verify that an unknown crop name raises a descriptive ValueError with valid choices."""
    predictor = CropYieldPredictor(ARTIFACTS_DIR)
    area = list(predictor.valid_areas)[0] if predictor.valid_areas else None

    with pytest.raises(ValueError, match="Unknown crop 'DragonFruit'"):
        predictor.predict_yield(
            crop="DragonFruit",
            year=2012,
            rainfall_mm=1000.0,
            pesticides_tonnes=50.0,
            temperature_c=20.0,
            area=area,
        )


def test_unknown_area_error():
    """Verify that an unknown area raises ValueError when the model expects Area."""
    predictor = CropYieldPredictor(ARTIFACTS_DIR)
    if "Area" in predictor.feature_order:
        with pytest.raises(ValueError, match="Unknown area 'Atlantis'"):
            predictor.predict_yield(
                crop="Maize",
                year=2012,
                rainfall_mm=1000.0,
                pesticides_tonnes=50.0,
                temperature_c=20.0,
                area="Atlantis",
            )


def test_negative_physical_inputs_error():
    """Verify that negative rainfall or pesticide quantities raise ValueError."""
    predictor = CropYieldPredictor(ARTIFACTS_DIR)
    area = list(predictor.valid_areas)[0] if predictor.valid_areas else None

    with pytest.raises(ValueError, match="Rainfall cannot be negative"):
        predictor.predict_yield(
            crop="Maize",
            year=2012,
            rainfall_mm=-150.0,
            pesticides_tonnes=50.0,
            temperature_c=20.0,
            area=area,
        )

    with pytest.raises(ValueError, match="Pesticides cannot be negative"):
        predictor.predict_yield(
            crop="Maize",
            year=2012,
            rainfall_mm=800.0,
            pesticides_tonnes=-10.0,
            temperature_c=20.0,
            area=area,
        )


def test_year_extrapolation_warning():
    """Verify that forecasting beyond 2013 raises a clear UserWarning about tree trend saturation."""
    predictor = CropYieldPredictor(ARTIFACTS_DIR)
    area = list(predictor.valid_areas)[0] if predictor.valid_areas else None

    with pytest.warns(UserWarning, match="is outside the model training range"):
        predictor.predict_yield(
            crop="Wheat",
            year=2025,
            rainfall_mm=900.0,
            pesticides_tonnes=60.0,
            temperature_c=17.0,
            area=area,
        )


def test_module_convenience_functions():
    """Verify load_model and predict_yield module functions work as expected for serving."""
    loaded_pred = load_model(ARTIFACTS_DIR)
    assert loaded_pred is not None

    area = list(loaded_pred.valid_areas)[0] if loaded_pred.valid_areas else None
    res = predict_yield(
        crop="Rice, paddy",
        year=2012,
        rainfall_mm=1400.0,
        pesticides_tonnes=80.0,
        temperature_c=24.0,
        area=area,
    )
    assert res["yield_hg_ha"] > 0
    assert res["yield_t_ha"] > 0


def test_variant_a_inference():
    """Verify that Variant A artifact (no country required) predicts cleanly without Area."""
    if VARIANT_A_DIR.exists():
        predictor_a = CropYieldPredictor(VARIANT_A_DIR)
        assert "Area" not in predictor_a.feature_order

        res = predictor_a.predict_yield(
            crop="Maize",
            year=2012,
            rainfall_mm=1100.0,
            pesticides_tonnes=45.0,
            temperature_c=21.0,
            area=None,  # No area required
        )
        assert res["yield_hg_ha"] > 0
        assert res["yield_t_ha"] > 0
