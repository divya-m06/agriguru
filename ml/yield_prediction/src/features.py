from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

BASE_ITEMS = [
    "Cassava",
    "Maize",
    "Plantains and others",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Sweet potatoes",
    "Wheat",
    "Yams",
]


class YieldFeatureTransformer(BaseEstimator, TransformerMixin):
    """Production feature engineering and preprocessing transformer for crop yield regression.

    Handles:
    - Strict one-hot encoding of crop Item (first N columns).
    - In-fold target or ordinal encoding for Area (country) in Variant B (no one-hot encoding).
    - Log1p transformation of skewed continuous variables (pesticides_tonnes).
    - Interaction term ablation: rainfall_x_temp.
    - Agronomic climate anomaly ablation: deviation from crop historical mean.
    - Year handling toggle for temporal extrapolation analysis.
    """

    def __init__(
        self,
        include_area: bool = False,
        include_year: bool = True,
        use_rainfall_x_temp: bool = False,
        use_climate_anomaly: bool = False,
        area_encoding: str = "target",
    ):
        self.include_area = include_area
        self.include_year = include_year
        self.use_rainfall_x_temp = use_rainfall_x_temp
        self.use_climate_anomaly = use_climate_anomaly
        self.area_encoding = area_encoding

        self.item_encoder_: Optional[OneHotEncoder] = None
        self.area_encoder_: Optional[Any] = None
        self.crop_climate_means_: Dict[str, Dict[str, float]] = {}
        self.global_rain_mean_: float = 0.0
        self.global_temp_mean_: float = 0.0
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[np.ndarray] = None) -> "YieldFeatureTransformer":
        """Fit categorical encoders and historical climate reference tables on training data only."""
        if not isinstance(X, pd.DataFrame):
            cols = ["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
            if not self.include_area and "Area" not in cols:
                cols = ["Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])
        else:
            df = X.copy()

        # 1. Fit Item (crop) OneHotEncoder (always first columns)
        items_in_data = sorted(df["Item"].unique())
        all_items = sorted(list(set(items_in_data) | set(BASE_ITEMS)))
        self.item_encoder_ = OneHotEncoder(
            categories=[all_items],
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.item_encoder_.fit(df[["Item"]])

        # 2. Fit Area In-Fold Encoder (Variant B only: never one-hot, uses TargetEncoder or OrdinalEncoder)
        if self.include_area and "Area" in df.columns:
            if self.area_encoding == "ordinal":
                self.area_encoder_ = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1.0,
                )
                self.area_encoder_.fit(df[["Area"]])
            else:
                if y is None:
                    raise ValueError("Target array y is required to fit TargetEncoder for Area in Variant B.")
                y_float = np.asarray(y, dtype=np.float64)
                self.area_encoder_ = TargetEncoder(smooth="auto", target_type="continuous")
                self.area_encoder_.fit(df[["Area"]], y_float)

        # 3. Fit historical climate anomaly baselines (training set only)
        if self.use_climate_anomaly:
            self.global_rain_mean_ = float(df["average_rain_fall_mm_per_year"].mean())
            self.global_temp_mean_ = float(df["avg_temp"].mean())
            grouped = df.groupby("Item")[["average_rain_fall_mm_per_year", "avg_temp"]].mean()
            self.crop_climate_means_ = {}
            for crop, row in grouped.iterrows():
                self.crop_climate_means_[str(crop)] = {
                    "mean_rain": float(row["average_rain_fall_mm_per_year"]),
                    "mean_temp": float(row["avg_temp"]),
                }
            self.crop_rain_means_map_ = {str(k): v["mean_rain"] for k, v in self.crop_climate_means_.items()}
            self.crop_temp_means_map_ = {str(k): v["mean_temp"] for k, v in self.crop_climate_means_.items()}

        # Build feature names
        feature_names = []
        item_feature_names = [f"Item_{cat}" for cat in self.item_encoder_.categories_[0]]
        feature_names.extend(item_feature_names)

        if self.include_area:
            feature_names.append(f"Area_{self.area_encoding}_encoded")

        if self.include_year:
            feature_names.append("Year")

        feature_names.extend(["average_rain_fall_mm_per_year", "pesticides_tonnes_log1p", "avg_temp"])

        if self.use_rainfall_x_temp:
            feature_names.append("rainfall_x_temp")

        if self.use_climate_anomaly:
            feature_names.extend(["rain_crop_anomaly", "temp_crop_anomaly"])

        self.feature_names_out_ = feature_names
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform input features with zero lookahead leakage."""
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        parts = []

        # 1. One-hot encoded Item (always first)
        item_ohe = self.item_encoder_.transform(df[["Item"]])
        parts.append(item_ohe)

        # 2. Area in-fold encoding (Variant B)
        if self.include_area:
            if "Area" in df.columns and self.area_encoder_ is not None:
                area_enc = self.area_encoder_.transform(df[["Area"]])
                parts.append(area_enc)
            else:
                parts.append(np.zeros((len(df), 1)))

        # 3. Year
        if self.include_year:
            year_vals = df["Year"].to_numpy(dtype=float).reshape(-1, 1)
            parts.append(year_vals)

        # 4. Continuous features (with log1p on pesticides)
        rainfall = df["average_rain_fall_mm_per_year"].to_numpy(dtype=float).reshape(-1, 1)
        pesticides_log1p = np.log1p(np.maximum(0.0, df["pesticides_tonnes"].to_numpy(dtype=float))).reshape(-1, 1)
        temp = df["avg_temp"].to_numpy(dtype=float).reshape(-1, 1)
        parts.extend([rainfall, pesticides_log1p, temp])

        # 5. Interaction: rainfall x temp
        if self.use_rainfall_x_temp:
            rx_t = (rainfall * temp).reshape(-1, 1)
            parts.append(rx_t)

        # 6. Climate anomaly relative to crop baseline (vectorized)
        if self.use_climate_anomaly:
            item_series = df["Item"].astype(str)
            rain_baseline = item_series.map(self.crop_rain_means_map_).fillna(self.global_rain_mean_).to_numpy(dtype=np.float32).reshape(-1, 1)
            temp_baseline = item_series.map(self.crop_temp_means_map_).fillna(self.global_temp_mean_).to_numpy(dtype=np.float32).reshape(-1, 1)
            rain_anom_arr = rainfall - rain_baseline
            temp_anom_arr = temp - temp_baseline
            parts.extend([rain_anom_arr, temp_anom_arr])

        result = np.hstack(parts).astype(np.float32)
        return result

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return list(self.feature_names_out_)
