import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import joblib
import numpy as np
import pandas as pd


class FertilizerPredictor:
    """Production inference engine for Fertilizer Recommendation multiclass classification."""

    def __init__(self, artifact_dir: Path | str, strict_range_check: bool = True):
        self.artifact_dir = Path(artifact_dir)
        self.strict_range_check = strict_range_check

        model_path = self.artifact_dir / "fertilizer_model.joblib"
        classes_path = self.artifact_dir / "classes.json"
        features_path = self.artifact_dir / "feature_order.json"
        categories_path = self.artifact_dir / "allowed_categories.json"
        ranges_path = self.artifact_dir / "numeric_ranges.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Fertilizer model artifact not found at {model_path}")

        self.model = joblib.load(model_path)

        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes: List[str] = json.load(f)

        with open(features_path, "r", encoding="utf-8") as f:
            self.feature_order: List[str] = json.load(f)

        with open(categories_path, "r", encoding="utf-8") as f:
            self.allowed_categories: Dict[str, List[str]] = json.load(f)

        with open(ranges_path, "r", encoding="utf-8") as f:
            self.numeric_ranges: Dict[str, Dict[str, float]] = json.load(f)

        self.valid_soils = {s.lower(): s for s in self.allowed_categories.get("soil_types", [])}
        self.valid_crops = {c.lower(): c for c in self.allowed_categories.get("crop_types", [])}

    def validate_and_prepare_input(
        self,
        temperature: float,
        humidity: float,
        moisture: float,
        soil_type: str,
        crop_type: str,
        nitrogen: float,
        potassium: float,
        phosphorous: float,
    ) -> pd.DataFrame:
        """Validate input features against category registries and physical boundaries.

        Args:
            temperature: Ambient temperature in Celsius.
            humidity: Relative humidity in percentage [0, 100].
            moisture: Soil moisture percentage.
            soil_type: Soil category (e.g. 'Sandy', 'Loamy', 'Black', 'Red', 'Clayey').
            crop_type: Crop category (e.g. 'Maize', 'Sugarcane', 'Cotton', 'Paddy', etc.).
            nitrogen: Nitrogen nutrient level in soil (N).
            potassium: Potassium nutrient level in soil (K).
            phosphorous: Phosphorous nutrient level in soil (P).

        Returns:
            Validated single-row DataFrame.

        Raises:
            ValueError: If categories or physical boundaries are violated.
        """
        # 1. Validate Soil Type
        if not soil_type or not isinstance(soil_type, str):
            raise ValueError(f"Invalid soil_type: must be non-empty string. Allowed: {sorted(list(self.valid_soils.values()))}")
        soil_clean = soil_type.strip().lower()
        if soil_clean not in self.valid_soils:
            raise ValueError(f"Unknown soil type '{soil_type}'. Allowed soil types: {sorted(list(self.valid_soils.values()))}")
        canonical_soil = self.valid_soils[soil_clean]

        # 2. Validate Crop Type
        if not crop_type or not isinstance(crop_type, str):
            raise ValueError(f"Invalid crop_type: must be non-empty string. Allowed: {sorted(list(self.valid_crops.values()))}")
        crop_clean = crop_type.strip().lower()
        if crop_clean not in self.valid_crops:
            raise ValueError(f"Unknown crop type '{crop_type}'. Allowed crop types: {sorted(list(self.valid_crops.values()))}")
        canonical_crop = self.valid_crops[crop_clean]

        # 3. Numeric conversions & non-physical negative checks
        try:
            temp_flt = float(temperature)
            hum_flt = float(humidity)
            moist_flt = float(moisture)
            n_flt = float(nitrogen)
            k_flt = float(potassium)
            p_flt = float(phosphorous)
        except Exception as e:
            raise ValueError(f"Numeric feature conversion error: {e}")

        if n_flt < 0:
            raise ValueError(f"Nitrogen cannot be negative, got {n_flt}")
        if k_flt < 0:
            raise ValueError(f"Potassium cannot be negative, got {k_flt}")
        if p_flt < 0:
            raise ValueError(f"Phosphorous cannot be negative, got {p_flt}")
        if moist_flt < 0:
            raise ValueError(f"Moisture cannot be negative, got {moist_flt}")
        if hum_flt < 0 or hum_flt > 100:
            raise ValueError(f"Humidity must be in [0, 100]%, got {hum_flt}")

        # 4. Physical range boundaries (with 25% margin from training extremes)
        if self.strict_range_check:
            checks = [
                ("temperature", temp_flt),
                ("humidity", hum_flt),
                ("moisture", moist_flt),
                ("nitrogen", n_flt),
                ("potassium", k_flt),
                ("phosphorous", p_flt),
            ]
            for feat, val in checks:
                if feat in self.numeric_ranges:
                    obs_min = self.numeric_ranges[feat]["min"]
                    obs_max = self.numeric_ranges[feat]["max"]
                    span = obs_max - obs_min
                    allowed_min = max(0.0, obs_min - 0.25 * span) if obs_min >= 0 else obs_min - 0.25 * span
                    allowed_max = obs_max + 0.25 * span
                    if val < allowed_min or val > allowed_max:
                        raise ValueError(
                            f"Input '{feat}'={val} is outside plausible agronomic range [{allowed_min:.1f}, {allowed_max:.1f}] "
                            f"(training bounds: [{obs_min:.1f}, {obs_max:.1f}])."
                        )

        # 5. Build 1-row DataFrame aligned with feature_order
        row_dict = {
            "temperature": temp_flt,
            "humidity": hum_flt,
            "moisture": moist_flt,
            "soil_type": canonical_soil,
            "crop_type": canonical_crop,
            "nitrogen": n_flt,
            "potassium": k_flt,
            "phosphorous": p_flt,
        }
        df = pd.DataFrame([row_dict])[self.feature_order]
        return df

    def predict_top_k(
        self,
        temperature: float,
        humidity: float,
        moisture: float,
        soil_type: str,
        crop_type: str,
        nitrogen: float,
        potassium: float,
        phosphorous: float,
        k: int = 3,
    ) -> Dict[str, Any]:
        """Predict top-k recommended fertilizers with sorted calibrated probabilities.

        Args:
            temperature: Temperature in Celsius.
            humidity: Relative humidity in %.
            moisture: Soil moisture content.
            soil_type: Soil type.
            crop_type: Crop type.
            nitrogen: Soil nitrogen (N).
            potassium: Soil potassium (K).
            phosphorous: Soil phosphorous (P).
            k: Number of recommendations to return (default: 3).

        Returns:
            Dictionary with ranked top-k recommendations, confidence percentages, and inputs.
        """
        df = self.validate_and_prepare_input(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type,
            nitrogen=nitrogen,
            potassium=potassium,
            phosphorous=phosphorous,
        )

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df)[0]
        else:
            # Fallback for models without predict_proba
            pred_idx = int(self.model.predict(df)[0])
            probs = np.zeros(len(self.classes))
            probs[pred_idx] = 1.0

        # Sort descending by probability
        sorted_indices = np.argsort(probs)[::-1]
        effective_k = min(k, len(self.classes))

        recommendations = []
        for idx in sorted_indices[:effective_k]:
            class_name = self.classes[idx]
            prob = float(probs[idx])
            recommendations.append({
                "fertilizer": class_name,
                "probability": round(prob, 4),
                "confidence_pct": f"{prob * 100:.1f}%",
            })

        return {
            "recommendations": recommendations,
            "top_fertilizer": recommendations[0]["fertilizer"],
            "top_confidence": recommendations[0]["probability"],
            "inputs": {
                "temperature": float(temperature),
                "humidity": float(humidity),
                "moisture": float(moisture),
                "soil_type": soil_type,
                "crop_type": crop_type,
                "nitrogen": float(nitrogen),
                "potassium": float(potassium),
                "phosphorous": float(phosphorous),
            },
        }


# Module-level convenience functions for FastAPI
_cached_predictor: Optional[FertilizerPredictor] = None


def load_model(artifact_dir: Path | str, strict_range_check: bool = True) -> FertilizerPredictor:
    """Load and cache fertilizer predictor."""
    global _cached_predictor
    _cached_predictor = FertilizerPredictor(artifact_dir, strict_range_check=strict_range_check)
    return _cached_predictor


def predict_fertilizer(
    temperature: float,
    humidity: float,
    moisture: float,
    soil_type: str,
    crop_type: str,
    nitrogen: float,
    potassium: float,
    phosphorous: float,
    k: int = 3,
) -> Dict[str, Any]:
    """Predict top-k fertilizers using cached predictor."""
    if _cached_predictor is None:
        raise RuntimeError("Model is not loaded. Call load_model(artifact_dir) first.")
    return _cached_predictor.predict_top_k(
        temperature=temperature,
        humidity=humidity,
        moisture=moisture,
        soil_type=soil_type,
        crop_type=crop_type,
        nitrogen=nitrogen,
        potassium=potassium,
        phosphorous=phosphorous,
        k=k,
    )
