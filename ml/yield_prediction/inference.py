import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import joblib
import numpy as np
import pandas as pd


class CropYieldPredictor:
    """Production inference wrapper for Crop Yield regression model."""

    def __init__(self, artifact_dir: Path | str, strict_range_check: bool = True):
        self.artifact_dir = Path(artifact_dir)
        self.strict_range_check = strict_range_check

        model_path = self.artifact_dir / "yield_model.joblib"
        features_path = self.artifact_dir / "feature_order.json"
        categories_path = self.artifact_dir / "valid_categories.json"
        ranges_path = self.artifact_dir / "input_ranges.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Yield model artifact not found at {model_path}")

        self.model = joblib.load(model_path)

        # Optional quantile models for prediction intervals
        self.quantile_low = None
        self.quantile_high = None
        if (self.artifact_dir / "quantile_low_model.joblib").exists():
            self.quantile_low = joblib.load(self.artifact_dir / "quantile_low_model.joblib")
        if (self.artifact_dir / "quantile_high_model.joblib").exists():
            self.quantile_high = joblib.load(self.artifact_dir / "quantile_high_model.joblib")

        with open(features_path, "r", encoding="utf-8") as f:
            self.feature_order: List[str] = json.load(f)

        with open(categories_path, "r", encoding="utf-8") as f:
            self.valid_categories: Dict[str, List[str]] = json.load(f)

        with open(ranges_path, "r", encoding="utf-8") as f:
            self.input_ranges: Dict[str, Dict[str, float]] = json.load(f)

        self.valid_crops = set(self.valid_categories.get("crops", []))
        self.valid_areas = set(self.valid_categories.get("areas", []))

    def validate_and_prepare_input(
        self,
        crop: str,
        year: int,
        rainfall_mm: float,
        pesticides_tonnes: float,
        temperature_c: float,
        area: Optional[str] = None,
    ) -> pd.DataFrame:
        """Validate input parameters against agronomic domain boundaries and format DataFrame.

        Args:
            crop: Crop name (Item).
            year: Year of crop season.
            rainfall_mm: Average annual rainfall in mm.
            pesticides_tonnes: Total pesticide usage in tonnes.
            temperature_c: Average temperature in Celsius.
            area: Optional country / Area name.

        Returns:
            Validated 1-row DataFrame ready for pipeline ingestion.

        Raises:
            ValueError: If inputs are invalid or outside domain constraints.
        """
        # 1. Validate crop
        if crop not in self.valid_crops:
            raise ValueError(f"Unknown crop '{crop}'. Valid crops: {sorted(list(self.valid_crops))}")

        # 2. Check Area if model expects Area
        if "Area" in self.feature_order:
            if not area:
                raise ValueError("Model variant requires 'area' (country name), but none was provided.")
            if area not in self.valid_areas:
                raise ValueError(f"Unknown area '{area}'. Valid areas: {sorted(list(self.valid_areas))}")

        # 3. Numeric conversions & bounds
        try:
            year_int = int(year)
            rain_flt = float(rainfall_mm)
            pest_flt = float(pesticides_tonnes)
            temp_flt = float(temperature_c)
        except Exception as e:
            raise ValueError(f"Numeric feature conversion failed: {e}")

        if rain_flt < 0:
            raise ValueError(f"Rainfall cannot be negative, got {rain_flt}")
        if pest_flt < 0:
            raise ValueError(f"Pesticides cannot be negative, got {pest_flt}")

        # 4. Year extrapolation warning
        year_min = int(self.input_ranges["Year"]["min"])
        year_max = int(self.input_ranges["Year"]["max"])
        if year_int < year_min or year_int > year_max:
            warnings.warn(
                f"Year {year_int} is outside the model training range [{year_min}, {year_max}]. "
                "Tree-based regressors cannot extrapolate temporal trends; predictions may saturate.",
                UserWarning,
                stacklevel=2,
            )

        # 5. Domain range check (with 20% margin)
        if self.strict_range_check:
            for feat, val in [
                ("average_rain_fall_mm_per_year", rain_flt),
                ("pesticides_tonnes", pest_flt),
                ("avg_temp", temp_flt),
            ]:
                if feat in self.input_ranges:
                    obs_min = self.input_ranges[feat]["min"]
                    obs_max = self.input_ranges[feat]["max"]
                    span = obs_max - obs_min
                    allowed_min = max(0.0, obs_min - 0.20 * span) if obs_min >= 0 else obs_min - 0.20 * span
                    allowed_max = obs_max + 0.20 * span
                    if val < allowed_min or val > allowed_max:
                        raise ValueError(
                            f"Input '{feat}'={val} is outside plausible physical range [{allowed_min:.2f}, {allowed_max:.2f}] "
                            f"(training bounds: [{obs_min:.2f}, {obs_max:.2f}])."
                        )

        # Construct DataFrame
        row_dict = {
            "Item": crop,
            "Year": year_int,
            "average_rain_fall_mm_per_year": rain_flt,
            "pesticides_tonnes": pest_flt,
            "avg_temp": temp_flt,
        }
        if "Area" in self.feature_order:
            row_dict["Area"] = area

        df = pd.DataFrame([row_dict])
        return df

    def predict_yield(
        self,
        crop: str,
        year: int,
        rainfall_mm: float,
        pesticides_tonnes: float,
        temperature_c: float,
        area: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict crop yield in both hectograms/hectare (hg/ha) and tonnes/hectare (t/ha).

        Args:
            crop: Crop name (e.g. 'Maize', 'Potatoes', 'Wheat').
            year: Year of crop season.
            rainfall_mm: Annual rainfall in mm.
            pesticides_tonnes: National pesticide usage in tonnes.
            temperature_c: Mean temperature in Celsius.
            area: Country name (required if model was trained with Area).

        Returns:
            Dictionary with predicted yields and optional 10th-90th quantile interval.
        """
        df = self.validate_and_prepare_input(
            crop=crop,
            year=year,
            rainfall_mm=rainfall_mm,
            pesticides_tonnes=pesticides_tonnes,
            temperature_c=temperature_c,
            area=area,
        )

        pred_hg_ha = float(self.model.predict(df)[0])
        # Prevent any negative yield artifacts
        pred_hg_ha = max(0.0, pred_hg_ha)
        pred_t_ha = pred_hg_ha / 10000.0

        result: Dict[str, Any] = {
            "crop": crop,
            "year": int(year),
            "yield_hg_ha": round(pred_hg_ha, 2),
            "yield_t_ha": round(pred_t_ha, 4),
        }

        # Optional prediction intervals if quantile models exist
        if self.quantile_low is not None and self.quantile_high is not None:
            low_hg_ha = max(0.0, float(self.quantile_low.predict(df)[0]))
            high_hg_ha = max(low_hg_ha, float(self.quantile_high.predict(df)[0]))
            result["prediction_interval_80_hg_ha"] = (round(low_hg_ha, 2), round(high_hg_ha, 2))
            result["prediction_interval_80_t_ha"] = (round(low_hg_ha / 10000.0, 4), round(high_hg_ha / 10000.0, 4))

        return result


# Module-level convenience functions for FastAPI
_cached_predictor: Optional[CropYieldPredictor] = None


def load_model(artifact_dir: Path | str, strict_range_check: bool = True) -> CropYieldPredictor:
    """Load and cache crop yield predictor."""
    global _cached_predictor
    _cached_predictor = CropYieldPredictor(artifact_dir, strict_range_check=strict_range_check)
    return _cached_predictor


def predict_yield(
    crop: str,
    year: int,
    rainfall_mm: float,
    pesticides_tonnes: float,
    temperature_c: float,
    area: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict crop yield using the cached predictor."""
    if _cached_predictor is None:
        raise RuntimeError("Model is not loaded. Call load_model(artifact_dir) first.")
    return _cached_predictor.predict_yield(
        crop=crop,
        year=year,
        rainfall_mm=rainfall_mm,
        pesticides_tonnes=pesticides_tonnes,
        temperature_c=temperature_c,
        area=area,
    )
