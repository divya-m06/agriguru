from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
ENGINEERED_FEATURES = [
    "NPK_total",
    "N_ratio",
    "P_ratio",
    "K_ratio",
    "temp_x_humidity",
    "rain_per_humidity",
]


class CropFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for engineering agronomic interaction features.

    Computes:
    - NPK_total: N + P + K
    - N_ratio: N / (NPK_total + eps)
    - P_ratio: P / (NPK_total + eps)
    - K_ratio: K / (NPK_total + eps)
    - temp_x_humidity: temperature * humidity
    - rain_per_humidity: rainfall / (humidity + eps)
    """

    def __init__(self, engineer_features: bool = False, eps: float = 1e-5):
        self.engineer_features = engineer_features
        self.eps = eps
        self.feature_names_: List[str] = list(BASE_FEATURES)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[np.ndarray] = None) -> "CropFeatureEngineer":
        """Fit transformer (records feature schema)."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = list(BASE_FEATURES)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform inputs into base or feature-engineered array."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_)

        if not self.engineer_features:
            return df[BASE_FEATURES].to_numpy(dtype=np.float64)

        # Compute engineered interaction features
        npk_total = df["N"] + df["P"] + df["K"]
        n_ratio = df["N"] / (npk_total + self.eps)
        p_ratio = df["P"] / (npk_total + self.eps)
        k_ratio = df["K"] / (npk_total + self.eps)
        temp_x_humidity = df["temperature"] * df["humidity"]
        rain_per_humidity = df["rainfall"] / (df["humidity"] + self.eps)

        engineered_df = df[BASE_FEATURES].copy()
        engineered_df["NPK_total"] = npk_total
        engineered_df["N_ratio"] = n_ratio
        engineered_df["P_ratio"] = p_ratio
        engineered_df["K_ratio"] = k_ratio
        engineered_df["temp_x_humidity"] = temp_x_humidity
        engineered_df["rain_per_humidity"] = rain_per_humidity

        return engineered_df.to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return output feature names after transformation."""
        if not self.engineer_features:
            return list(BASE_FEATURES)
        return list(BASE_FEATURES) + list(ENGINEERED_FEATURES)
