import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd


class CropRecommendationPredictor:
    """Production inference wrapper for Crop Recommendation model."""

    def __init__(self, artifact_dir: Path | str, strict_range_check: bool = True):
        self.artifact_dir = Path(artifact_dir)
        self.strict_range_check = strict_range_check

        model_path = self.artifact_dir / "crop_model.joblib"
        le_path = self.artifact_dir / "label_encoder.joblib"
        features_path = self.artifact_dir / "feature_order.json"
        classes_path = self.artifact_dir / "classes.json"
        ranges_path = self.artifact_dir / "input_ranges.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        if not le_path.exists():
            raise FileNotFoundError(f"Label encoder artifact not found at {le_path}")

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(le_path)

        with open(features_path, "r", encoding="utf-8") as f:
            self.feature_order: List[str] = json.load(f)

        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes: List[str] = json.load(f)

        with open(ranges_path, "r", encoding="utf-8") as f:
            self.input_ranges: Dict[str, Dict[str, float]] = json.load(f)

    def validate_and_prepare_input(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
    ) -> pd.DataFrame:
        """Validate input features against schema and physical domain boundaries.

        Args:
            features: Dictionary of feature names to values, or DataFrame.

        Returns:
            Validated pd.DataFrame matching exact feature order.

        Raises:
            ValueError: If features are missing, non-numeric, or outside valid physical ranges.
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise TypeError(f"Expected dict or pd.DataFrame, got {type(features)}")

        # 1. Check for missing columns
        missing_features = [col for col in self.feature_order if col not in df.columns]
        if missing_features:
            raise ValueError(f"Input is missing required features: {missing_features}. Required: {self.feature_order}")

        # Ensure exact column ordering
        df = df[self.feature_order]

        # 2. Check numeric types and non-null
        for col in self.feature_order:
            if df[col].isnull().any():
                raise ValueError(f"Feature '{col}' contains null or NaN values.")
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception as e:
                raise ValueError(f"Feature '{col}' must be numeric. Conversion failed: {e}")

        # 3. Check physical ranges (with 20% margin beyond observed extremes)
        if self.strict_range_check:
            for col in self.feature_order:
                observed_min = self.input_ranges[col]["min"]
                observed_max = self.input_ranges[col]["max"]
                span = observed_max - observed_min
                allowed_min = max(0.0, observed_min - 0.20 * span) if observed_min >= 0 else observed_min - 0.20 * span
                allowed_max = observed_max + 0.20 * span

                col_min = df[col].min()
                col_max = df[col].max()

                if col_min < allowed_min or col_max > allowed_max:
                    raise ValueError(
                        f"Feature '{col}' value is out of physical plausible range: "
                        f"observed [{col_min}, {col_max}], allowed [{allowed_min:.2f}, {allowed_max:.2f}] "
                        f"(training bounds: [{observed_min:.2f}, {observed_max:.2f}])."
                    )

        return df

    def predict(self, features: Union[Dict[str, Any], pd.DataFrame]) -> Union[str, List[str]]:
        """Predict top-1 recommended crop for given features.

        Args:
            features: Dictionary or DataFrame with required feature columns.

        Returns:
            Single crop name string (if single input) or list of crop name strings.
        """
        df = self.validate_and_prepare_input(features)
        y_pred = self.model.predict(df)
        labels = self.label_encoder.inverse_transform(y_pred)
        if isinstance(features, dict) or len(labels) == 1:
            return str(labels[0])
        return [str(l) for l in labels]

    def predict_top_k(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
        k: int = 3,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Predict top-k recommended crops with probabilities.

        Args:
            features: Feature dict or DataFrame.
            k: Number of top recommendations to return (default: 3).

        Returns:
            List of (crop_name, probability) tuples for single input,
            or list of such lists for multiple inputs.
        """
        df = self.validate_and_prepare_input(features)
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Loaded model does not support probability estimation (predict_proba).")

        probabilities = self.model.predict_proba(df)
        k = min(k, len(self.classes))

        all_results = []
        for prob_row in probabilities:
            top_k_indices = np.argsort(prob_row)[::-1][:k]
            row_recommendations = [
                (self.classes[idx], float(prob_row[idx]))
                for idx in top_k_indices
            ]
            all_results.append(row_recommendations)

        if isinstance(features, dict) or len(all_results) == 1:
            return all_results[0]
        return all_results


# Module-level convenience functions for FastAPI
_cached_predictor: Optional[CropRecommendationPredictor] = None


def load_model(artifact_dir: Path | str, strict_range_check: bool = True) -> CropRecommendationPredictor:
    """Load and cache the crop recommendation predictor.

    Args:
        artifact_dir: Path to exported artifacts directory.
        strict_range_check: Whether to strictly validate physical feature boundaries.

    Returns:
        CropRecommendationPredictor instance.
    """
    global _cached_predictor
    _cached_predictor = CropRecommendationPredictor(
        artifact_dir=artifact_dir,
        strict_range_check=strict_range_check,
    )
    return _cached_predictor


def predict(features: Union[Dict[str, Any], pd.DataFrame]) -> Union[str, List[str]]:
    """Predict top-1 crop using cached predictor."""
    if _cached_predictor is None:
        raise RuntimeError("Model is not loaded. Call load_model(artifact_dir) first.")
    return _cached_predictor.predict(features)


def predict_top_k(
    features: Union[Dict[str, Any], pd.DataFrame],
    k: int = 3,
) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    """Predict top-k crops with probabilities using cached predictor."""
    if _cached_predictor is None:
        raise RuntimeError("Model is not loaded. Call load_model(artifact_dir) first.")
    return _cached_predictor.predict_top_k(features, k=k)
