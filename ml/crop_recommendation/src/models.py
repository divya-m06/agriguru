import logging
from typing import Any, Dict, List, Tuple
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml.crop_recommendation.src.features import CropFeatureEngineer

logger = logging.getLogger("ml.crop_recommendation.models")

# List of models that require standard feature scaling
SCALE_SENSITIVE_MODELS = {"GaussianNB", "LogisticRegression", "KNN", "SVC", "MLPClassifier"}


def create_pipeline(
    model_name: str,
    estimator_params: Dict[str, Any] = None,
    engineer_features: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Create a self-contained Scikit-Learn Pipeline combining feature engineering,
    conditional scaling, and the classifier.

    Args:
        model_name: Identifier for the model architecture.
        estimator_params: Dictionary of kwargs for the classifier.
        engineer_features: Boolean flag to enable interaction feature engineering.
        random_state: Random seed for stochastic algorithms.

    Returns:
        Configured Pipeline instance.
    """
    params = dict(estimator_params or {})

    # Classifier instantiation
    if model_name == "GaussianNB":
        clf = GaussianNB(**params)
    elif model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=params.pop("max_iter", 1000),
            random_state=random_state,
            **params,
        )
    elif model_name == "KNN":
        clf = KNeighborsClassifier(**params)
    elif model_name == "SVC":
        clf = SVC(
            probability=True,  # Required for predict_proba
            random_state=random_state,
            **params,
        )
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "ExtraTrees":
        clf = ExtraTreesClassifier(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            random_state=random_state,
            **params,
        )
    elif model_name == "LightGBM":
        clf = LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    elif model_name == "XGBoost":
        clf = XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            **params,
        )
    elif model_name == "MLPClassifier":
        clf = MLPClassifier(
            random_state=random_state,
            early_stopping=True,
            max_iter=params.pop("max_iter", 500),
            **params,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    feature_transformer = CropFeatureEngineer(engineer_features=engineer_features)

    steps: List[Tuple[str, Any]] = [("features", feature_transformer)]
    if model_name in SCALE_SENSITIVE_MODELS:
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("scaler", "passthrough"))

    steps.append(("classifier", clf))

    return Pipeline(steps)


def get_hyperparameter_distributions(model_name: str) -> Dict[str, Any]:
    """Return randomized search distributions for each candidate model family.

    Args:
        model_name: Model identifier.

    Returns:
        Dict of parameter distributions mapped to pipeline classifier step.
    """
    if model_name in ["RandomForest", "ExtraTrees"]:
        return {
            "classifier__n_estimators": [100, 200, 300, 400, 500],
            "classifier__max_depth": [6, 10, 15, 20, 25, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", 0.5, None],
        }
    elif model_name == "LightGBM":
        return {
            "classifier__n_estimators": [100, 200, 300, 400],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.15],
            "classifier__max_depth": [3, 5, 7, -1],
            "classifier__num_leaves": [15, 31, 63],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
    elif model_name == "XGBoost":
        return {
            "classifier__n_estimators": [100, 200, 300, 400],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.15],
            "classifier__max_depth": [3, 4, 6, 8],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
    elif model_name == "SVC":
        return {
            "classifier__C": [0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }
    elif model_name == "KNN":
        return {
            "classifier__n_neighbors": [3, 5, 7, 9, 13, 17, 21, 25],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        }
    elif model_name == "LogisticRegression":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 50.0],
            "classifier__solver": ["lbfgs"],
        }
    elif model_name == "MLPClassifier":
        return {
            "classifier__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "classifier__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "classifier__learning_rate_init": [0.001, 0.005, 0.01],
        }
    elif model_name == "HistGradientBoosting":
        return {
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__max_iter": [100, 200, 300],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__min_samples_leaf": [10, 20, 30],
        }
    elif model_name == "GaussianNB":
        return {
            "classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        }
    return {}


def build_soft_voting_ensemble(
    top_models: List[Tuple[str, Pipeline]],
    random_state: int = 42,
) -> Pipeline:
    """Build a soft-voting ensemble from the best calibrated top-k pipelines.

    Args:
        top_models: List of tuples (name, fitted_or_configured_pipeline).
        random_state: Random state for deterministic execution.

    Returns:
        Pipeline wrapping the VotingClassifier.
    """
    named_estimators = [(name, pipe) for name, pipe in top_models]
    ensemble = VotingClassifier(
        estimators=named_estimators,
        voting="soft",
        n_jobs=-1,
    )
    return Pipeline([("voting_ensemble", ensemble)])
