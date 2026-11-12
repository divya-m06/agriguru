import logging
from typing import Any, Dict, List, Optional
from catboost import CatBoostClassifier
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.fertilizer_recommendation.src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml.fertilizer_recommendation.src.features import create_fertilizer_preprocessor

logger = logging.getLogger("ml.fertilizer_recommendation.models")

SCALE_SENSITIVE_MODELS = {"LogisticRegression", "KNN", "SVC", "GaussianNB"}


def create_fertilizer_pipeline(
    model_name: str,
    estimator_params: Optional[Dict[str, Any]] = None,
    add_engineered_features: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Create self-contained Pipeline coupling ColumnTransformer preprocessing and classifier.

    Args:
        model_name: Architecture identifier.
        estimator_params: Kwargs for classifier.
        add_engineered_features: Flag for NPK ratio ablation.
        random_state: Deterministic random seed.

    Returns:
        Configured scikit-learn Pipeline.
    """
    params = dict(estimator_params or {})

    if model_name == "Dummy_Majority":
        clf = DummyClassifier(strategy="most_frequent")
    elif model_name == "Dummy_Stratified":
        clf = DummyClassifier(strategy="stratified", random_state=random_state)
    elif model_name == "LogisticRegression":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=params.pop("max_iter", 1000),
            random_state=random_state,
            **params,
        )
    elif model_name == "KNN":
        clf = KNeighborsClassifier(n_jobs=-1, **params)
    elif model_name == "SVC":
        clf = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
            **params,
        )
    elif model_name == "GaussianNB":
        clf = GaussianNB(**params)
    elif model_name == "DecisionTree":
        clf = DecisionTreeClassifier(
            max_depth=params.pop("max_depth", 4),
            min_samples_leaf=params.pop("min_samples_leaf", 3),
            class_weight="balanced",
            random_state=random_state,
            **params,
        )
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 200),
            max_depth=params.pop("max_depth", 4),
            min_samples_leaf=params.pop("min_samples_leaf", 2),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "ExtraTrees":
        clf = ExtraTreesClassifier(
            n_estimators=params.pop("n_estimators", 200),
            max_depth=params.pop("max_depth", 4),
            min_samples_leaf=params.pop("min_samples_leaf", 2),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "XGBoost":
        clf = XGBClassifier(
            n_estimators=params.pop("n_estimators", 100),
            max_depth=params.pop("max_depth", 2),
            learning_rate=params.pop("learning_rate", 0.05),
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            **params,
        )
    elif model_name == "CatBoost":
        clf = CatBoostClassifier(
            iterations=params.pop("iterations", 300),
            depth=params.pop("depth", 3),
            l2_leaf_reg=params.pop("l2_leaf_reg", 10.0),
            random_seed=random_state,
            thread_count=-1,
            verbose=False,
            **params,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    scale_numeric = model_name in SCALE_SENSITIVE_MODELS
    preprocessor = create_fertilizer_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        scale_numeric=scale_numeric,
        add_engineered_features=add_engineered_features,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    return pipeline


def get_fertilizer_tuning_grid(model_name: str) -> Dict[str, Any]:
    """Return regularized hyperparameter search grids tailored for N=99 dataset."""
    if model_name == "LogisticRegression":
        return {
            "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        }
    elif model_name == "SVC":
        return {
            "classifier__C": [0.1, 1.0, 5.0, 10.0, 30.0],
            "classifier__gamma": ["scale", "auto"],
        }
    elif model_name == "KNN":
        return {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
        }
    elif model_name == "DecisionTree":
        return {
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__min_samples_leaf": [2, 3, 4],
        }
    elif model_name in ["RandomForest", "ExtraTrees"]:
        return {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__min_samples_leaf": [2, 3, 4],
        }
    elif model_name == "XGBoost":
        return {
            "classifier__max_depth": [2, 3],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__n_estimators": [50, 100, 150],
        }
    elif model_name == "CatBoost":
        return {
            "classifier__depth": [3, 4, 5],
            "classifier__l2_leaf_reg": [3.0, 10.0, 20.0],
            "classifier__iterations": [200, 300, 400],
        }
    return {}
