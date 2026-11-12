import logging
from typing import Any, Dict, List, Tuple
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.yield_prediction.src.features import BASE_ITEMS, YieldFeatureTransformer

logger = logging.getLogger("ml.yield_prediction.models")

SCALE_SENSITIVE_MODELS = {"Ridge", "ElasticNet", "KNN", "MLPRegressor"}


class PerCropBaselineRegressor(BaseEstimator, RegressorMixin):
    """Agronomic baseline predicting per-crop historical mean or median yield."""

    def __init__(self, strategy: str = "mean", num_crop_categories: int = 10):
        self.strategy = strategy
        self.num_crop_categories = num_crop_categories
        self.crop_stats_: Dict[int, float] = {}
        self.global_stat_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerCropBaselineRegressor":
        # First num_crop_categories columns are one-hot encoded Item
        if hasattr(X, "ndim") and X.ndim == 2 and X.shape[1] >= self.num_crop_categories:
            crop_indices = np.argmax(X[:, :self.num_crop_categories], axis=1)
        else:
            crop_indices = np.zeros(len(y), dtype=int)

        self.global_stat_ = float(np.mean(y) if self.strategy == "mean" else np.median(y))
        for idx in range(self.num_crop_categories):
            mask = (crop_indices == idx)
            if np.any(mask):
                vals = y[mask]
                self.crop_stats_[idx] = float(np.mean(vals) if self.strategy == "mean" else np.median(vals))
            else:
                self.crop_stats_[idx] = self.global_stat_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(X, "ndim") and X.ndim == 2 and X.shape[1] >= self.num_crop_categories:
            crop_indices = np.argmax(X[:, :self.num_crop_categories], axis=1)
        else:
            crop_indices = np.zeros(len(X), dtype=int)
        return np.array([self.crop_stats_.get(idx, self.global_stat_) for idx in crop_indices], dtype=float)


def create_yield_pipeline(
    model_name: str,
    estimator_params: Dict[str, Any] = None,
    include_area: bool = False,
    include_year: bool = True,
    use_rainfall_x_temp: bool = False,
    use_climate_anomaly: bool = False,
    area_encoding: str = "target",
    random_state: int = 42,
) -> Pipeline:
    """Create a fully self-contained Pipeline integrating feature transformation,
    scaling, and TransformedTargetRegressor(log1p).

    Args:
        model_name: Identifier of model architecture.
        estimator_params: Dictionary of kwargs for regressor.
        include_area: Flag to include country Area (Variant B).
        include_year: Flag to include Year feature.
        use_rainfall_x_temp: Flag to add interaction term.
        use_climate_anomaly: Flag to add climate anomaly features.
        area_encoding: Encoding for Area ('target' or 'ordinal').
        random_state: Random state for deterministic execution.

    Returns:
        Configured Pipeline.
    """
    params = dict(estimator_params or {})

    if model_name == "Baseline_Mean":
        base_reg = PerCropBaselineRegressor(strategy="mean", num_crop_categories=len(BASE_ITEMS))
    elif model_name == "Baseline_Median":
        base_reg = PerCropBaselineRegressor(strategy="median", num_crop_categories=len(BASE_ITEMS))
    elif model_name == "Ridge":
        base_reg = Ridge(random_state=random_state, **params)
    elif model_name == "ElasticNet":
        base_reg = ElasticNet(random_state=random_state, max_iter=2000, **params)
    elif model_name == "KNN":
        base_reg = KNeighborsRegressor(n_jobs=-1, **params)
    elif model_name == "RandomForest":
        base_reg = RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "ExtraTrees":
        base_reg = ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "GradientBoosting":
        base_reg = GradientBoostingRegressor(
            random_state=random_state,
            **params,
        )
    elif model_name == "HistGradientBoosting":
        base_reg = HistGradientBoostingRegressor(
            random_state=random_state,
            **params,
        )
    elif model_name == "LightGBM":
        base_reg = LGBMRegressor(
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    elif model_name == "XGBoost":
        base_reg = XGBRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "CatBoost":
        base_reg = CatBoostRegressor(
            random_seed=random_state,
            thread_count=-1,
            verbose=False,
            **params,
        )
    elif model_name == "MLPRegressor":
        base_reg = MLPRegressor(
            random_state=random_state,
            max_iter=params.pop("max_iter", 500),
            early_stopping=True,
            **params,
        )
    else:
        raise ValueError(f"Unknown regression model: {model_name}")

    # Wrap regressor in TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)
    target_regressor = TransformedTargetRegressor(
        regressor=base_reg,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    feature_transformer = YieldFeatureTransformer(
        include_area=include_area,
        include_year=include_year,
        use_rainfall_x_temp=use_rainfall_x_temp,
        use_climate_anomaly=use_climate_anomaly,
        area_encoding=area_encoding,
    )

    steps: List[Tuple[str, Any]] = [("features", feature_transformer)]
    if model_name in SCALE_SENSITIVE_MODELS:
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("scaler", "passthrough"))

    steps.append(("regressor", target_regressor))

    return Pipeline(steps)


def create_quantile_regressors(
    include_area: bool = False,
    include_year: bool = True,
    alphas: Tuple[float, float] = (0.1, 0.9),
    random_state: int = 42,
) -> Tuple[Pipeline, Pipeline]:
    """Create LightGBM quantile regression models for lower and upper prediction intervals.

    Args:
        include_area: Variant B flag.
        include_year: Year toggle.
        alphas: Tuple of lower and upper quantiles (default: 0.1, 0.9).
        random_state: Random state.

    Returns:
        Tuple of (lower_quantile_pipe, upper_quantile_pipe).
    """
    pipe_low = create_yield_pipeline(
        "LightGBM",
        estimator_params={"objective": "quantile", "alpha": alphas[0]},
        include_area=include_area,
        include_year=include_year,
        random_state=random_state,
    )
    pipe_high = create_yield_pipeline(
        "LightGBM",
        estimator_params={"objective": "quantile", "alpha": alphas[1]},
        include_area=include_area,
        include_year=include_year,
        random_state=random_state,
    )
    return pipe_low, pipe_high


def get_yield_tuning_distributions(model_name: str) -> Dict[str, Any]:
    """Return hyperparameter search distributions for yield regressors."""
    if model_name in ["RandomForest", "ExtraTrees"]:
        return {
            "regressor__regressor__n_estimators": [200, 300, 500],
            "regressor__regressor__max_depth": [10, 15, 20, 25, None],
            "regressor__regressor__min_samples_leaf": [1, 2, 4],
            "regressor__regressor__max_features": [0.5, 0.7, 1.0],
        }
    elif model_name == "LightGBM":
        return {
            "regressor__regressor__n_estimators": [200, 400, 600],
            "regressor__regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__regressor__num_leaves": [31, 63, 127],
            "regressor__regressor__max_depth": [5, 8, 12, -1],
            "regressor__regressor__subsample": [0.7, 0.85, 1.0],
            "regressor__regressor__colsample_bytree": [0.6, 0.8, 1.0],
            "regressor__regressor__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
    elif model_name == "XGBoost":
        return {
            "regressor__regressor__n_estimators": [200, 400, 600],
            "regressor__regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__regressor__max_depth": [4, 6, 8, 10],
            "regressor__regressor__subsample": [0.7, 0.85, 1.0],
            "regressor__regressor__colsample_bytree": [0.6, 0.8, 1.0],
            "regressor__regressor__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
    elif model_name == "CatBoost":
        return {
            "regressor__regressor__iterations": [300, 500, 800],
            "regressor__regressor__learning_rate": [0.03, 0.07, 0.12],
            "regressor__regressor__depth": [4, 6, 8],
            "regressor__regressor__l2_leaf_reg": [1.0, 3.0, 7.0],
        }
    elif model_name in ["Ridge", "ElasticNet"]:
        return {
            "regressor__regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }
    elif model_name == "KNN":
        return {
            "regressor__regressor__n_neighbors": [3, 5, 9, 15],
            "regressor__regressor__weights": ["uniform", "distance"],
        }
    elif model_name == "MLPRegressor":
        return {
            "regressor__regressor__hidden_layer_sizes": [(64, 32), (128, 64)],
            "regressor__regressor__alpha": [1e-4, 1e-3, 1e-2],
        }
    return {}
