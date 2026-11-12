from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KNOWN_SOIL_TYPES = [
    "Black",
    "Clayey",
    "Loamy",
    "Red",
    "Sandy",
]

KNOWN_CROP_TYPES = [
    "Barley",
    "Cotton",
    "Ground Nuts",
    "Maize",
    "Millets",
    "Oil seeds",
    "Paddy",
    "Pulses",
    "Sugarcane",
    "Tobacco",
    "Wheat",
]


class FertilizerFeatureEngineering(BaseEstimator, TransformerMixin):
    """Minimal feature engineering ablation for nutrient ratios and total nutrient load."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[np.ndarray] = None) -> "FertilizerFeatureEngineering":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            n = X["nitrogen"].to_numpy(dtype=float)
            k = X["potassium"].to_numpy(dtype=float)
            p = X["phosphorous"].to_numpy(dtype=float)
            raw_numeric = X.to_numpy(dtype=float)
        else:
            # Assumes order: temperature, humidity, moisture, nitrogen, potassium, phosphorous
            n = X[:, 3].astype(float)
            k = X[:, 4].astype(float)
            p = X[:, 5].astype(float)
            raw_numeric = X.astype(float)

        nutrient_total = (n + p + k).reshape(-1, 1)
        total_safe = nutrient_total + self.eps

        ratio_n_total = (n.reshape(-1, 1)) / total_safe
        ratio_p_total = (p.reshape(-1, 1)) / total_safe
        ratio_k_total = (k.reshape(-1, 1)) / total_safe

        ratio_n_p = (n / (p + 1.0)).reshape(-1, 1)
        ratio_n_k = (n / (k + 1.0)).reshape(-1, 1)
        ratio_p_k = (p / (k + 1.0)).reshape(-1, 1)

        engineered = np.hstack([
            raw_numeric,
            nutrient_total,
            ratio_n_total,
            ratio_p_total,
            ratio_k_total,
            ratio_n_p,
            ratio_n_k,
            ratio_p_k,
        ])
        return engineered


def create_fertilizer_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
    add_engineered_features: bool = False,
) -> ColumnTransformer:
    """Create ColumnTransformer preprocessing categorical and numeric features strictly in-pipeline.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        scale_numeric: Whether to apply StandardScaler to continuous features.
        add_engineered_features: Whether to compute NPK ratios in ablation.

    Returns:
        Configured ColumnTransformer.
    """
    # Categorical pipeline: OneHotEncoder with known categories fallback
    cat_transformer = OneHotEncoder(
        categories=[KNOWN_SOIL_TYPES, KNOWN_CROP_TYPES],
        handle_unknown="ignore",
        sparse_output=False,
    )

    # Numeric pipeline
    num_steps = []
    if add_engineered_features:
        num_steps.append(("engineering", FertilizerFeatureEngineering()))

    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    else:
        num_steps.append(("scaler", "passthrough"))

    num_transformer = Pipeline(num_steps) if num_steps else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor
