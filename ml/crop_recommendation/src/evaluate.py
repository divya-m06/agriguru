import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import Pipeline

from ml.common.metrics import calculate_classification_metrics
from ml.crop_recommendation.src.models import (
    build_soft_voting_ensemble,
    create_pipeline,
    get_hyperparameter_distributions,
)

logger = logging.getLogger("ml.crop_recommendation.evaluate")


def run_cross_validation_comparison(
    candidate_names: List[str],
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int = 5,
    n_repeats: int = 3,
    engineer_features: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Repeated Stratified K-Fold Cross Validation across all candidate algorithms.

    Args:
        candidate_names: List of model candidate identifiers.
        X: Feature DataFrame (train+val).
        y: Integer target labels (train+val).
        cv_splits: Number of folds per repeat.
        n_repeats: Number of CV repeats.
        engineer_features: Boolean flag to apply feature engineering.
        seed: Random state.

    Returns:
        DataFrame summarizing mean and std of accuracy and macro-F1.
    """
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=n_repeats, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }

    results = []
    logger.info("Starting CV model comparison across %d candidates (%d folds, %d repeats)...",
                len(candidate_names), cv_splits, n_repeats)

    for name in candidate_names:
        pipe = create_pipeline(name, engineer_features=engineer_features, random_state=seed)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        acc_mean = float(np.mean(scores["test_accuracy"]))
        acc_std = float(np.std(scores["test_accuracy"]))
        f1_mean = float(np.mean(scores["test_macro_f1"]))
        f1_std = float(np.std(scores["test_macro_f1"]))
        fit_time_mean = float(np.mean(scores["fit_time"]))

        results.append({
            "model": name,
            "mean_macro_f1": f1_mean,
            "std_macro_f1": f1_std,
            "mean_accuracy": acc_mean,
            "std_accuracy": acc_std,
            "fit_time_seconds": fit_time_mean,
            "engineer_features": engineer_features,
        })
        logger.info("[%s] Macro-F1: %.4f (+/- %.4f) | Accuracy: %.4f (+/- %.4f)",
                    name, f1_mean, f1_std, acc_mean, acc_std)

    df_results = pd.DataFrame(results).sort_values(by="mean_macro_f1", ascending=False).reset_index(drop=True)
    return df_results


def run_feature_ablation(
    candidate_names: List[str],
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> Tuple[pd.DataFrame, bool]:
    """Compare baseline features vs engineered features. Keep engineering only if gain > std.

    Args:
        candidate_names: Subset of top models to test ablation on.
        X: Feature DataFrame.
        y: Target array.
        cv_splits: Number of CV folds.
        n_repeats: Number of CV repeats.
        seed: Random state.

    Returns:
        Tuple of (ablation_dataframe, should_keep_engineering_bool).
    """
    logger.info("Running feature engineering ablation study...")
    df_base = run_cross_validation_comparison(
        candidate_names, X, y, cv_splits, n_repeats, engineer_features=False, seed=seed
    )
    df_eng = run_cross_validation_comparison(
        candidate_names, X, y, cv_splits, n_repeats, engineer_features=True, seed=seed
    )

    df_merged = pd.merge(
        df_base[["model", "mean_macro_f1", "std_macro_f1"]],
        df_eng[["model", "mean_macro_f1", "std_macro_f1"]],
        on="model",
        suffixes=("_baseline", "_engineered"),
    )

    df_merged["f1_gain"] = df_merged["mean_macro_f1_engineered"] - df_merged["mean_macro_f1_baseline"]
    # Criterion: Keep only if average gain exceeds the baseline standard deviation
    avg_gain = float(df_merged["f1_gain"].mean())
    avg_std = float(df_merged["std_macro_f1_baseline"].mean())

    keep_engineering = avg_gain > avg_std
    logger.info("Feature ablation result: Avg Gain = %.5f, Baseline Std = %.5f -> Keep: %s",
                avg_gain, avg_std, keep_engineering)
    return df_merged, keep_engineering


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    engineer_features: bool = False,
    n_iter: int = 30,
    cv_splits: int = 5,
    seed: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """Perform hyperparameter tuning via RandomizedSearchCV inside stratified CV.

    Args:
        model_name: Candidate model identifier.
        X: Training features.
        y: Training labels.
        engineer_features: Whether to use engineered features.
        n_iter: Number of random parameter samples.
        cv_splits: Number of CV folds.
        seed: Random state.

    Returns:
        Tuple of (best_pipeline, best_params, best_cv_score).
    """
    logger.info("Tuning hyperparameters for %s (n_iter=%d)...", model_name, n_iter)
    base_pipe = create_pipeline(model_name, engineer_features=engineer_features, random_state=seed)
    param_dist = get_hyperparameter_distributions(model_name)

    if not param_dist:
        logger.info("No hyperparameter distribution for %s; fitting default pipeline.", model_name)
        base_pipe.fit(X, y)
        return base_pipe, {}, 0.0

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    logger.info("[%s] Best CV Macro-F1: %.4f with params: %s",
                model_name, search.best_score_, search.best_params_)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def select_best_model_occam(
    cv_table: pd.DataFrame,
    simplicity_order: Optional[List[str]] = None,
) -> str:
    """Select the simplest model within 1 standard error of the highest CV score.

    Standard error = std / sqrt(n_folds * n_repeats)
    """
    if simplicity_order is None:
        # Default simplicity hierarchy: simpler interpretable models first
        simplicity_order = [
            "GaussianNB",
            "KNN",
            "LogisticRegression",
            "RandomForest",
            "ExtraTrees",
            "HistGradientBoosting",
            "LightGBM",
            "XGBoost",
            "MLPClassifier",
            "SVC",
            "SoftVotingEnsemble",
        ]

    best_row = cv_table.iloc[0]
    best_f1 = best_row["mean_macro_f1"]
    best_std = best_row["std_macro_f1"]
    # 5 folds x 3 repeats = 15 total fold evaluations
    total_evals = 15
    one_se = best_std / np.sqrt(total_evals)
    threshold = best_f1 - one_se

    logger.info("Best CV Model: %s (Macro-F1=%.4f). 1-SE threshold: %.4f",
                best_row["model"], best_f1, threshold)

    eligible_models = cv_table[cv_table["mean_macro_f1"] >= threshold]["model"].tolist()
    logger.info("Eligible models within 1-SE: %s", eligible_models)

    # Pick the simplest model in the simplicity ranking
    for candidate in simplicity_order:
        if candidate in eligible_models:
            logger.info("Selected model by Occam's razor: %s", candidate)
            return candidate

    return str(best_row["model"])


def evaluate_on_test_set(
    pipeline: Pipeline,
    X_train_val: pd.DataFrame,
    y_train_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, Any]:
    """Fit final selected pipeline on train+val and evaluate ONCE on held-out test set."""
    logger.info("Refitting chosen model on train+val (%d samples)...", len(X_train_val))
    pipeline.fit(X_train_val, y_train_val)

    logger.info("Evaluating chosen model on test set (%d samples)...", len(X_test))
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        classes=classes,
        k=3,
    )
    logger.info("Test Set Results -> Accuracy: %.4f | Macro-F1: %.4f | Top-3 Acc: %.4f",
                metrics["accuracy"], metrics["macro_f1"], metrics.get("top_3_accuracy", 0.0))
    return metrics


def check_and_apply_calibration(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[Pipeline, bool, Dict[str, float]]:
    """Test if CalibratedClassifierCV improves validation log loss before applying."""
    pipeline_copy = copy.deepcopy(pipeline)
    pipeline_copy.fit(X_train, y_train)

    val_prob_raw = pipeline_copy.predict_proba(X_val)
    raw_metrics = calculate_classification_metrics(y_val, np.argmax(val_prob_raw, axis=1), val_prob_raw)
    raw_loss = raw_metrics.get("log_loss", float("inf"))

    try:
        min_class_samples = int(np.min(np.bincount(y_train)))
        cal_cv = min(3, max(2, min_class_samples))
        calibrated_model = CalibratedClassifierCV(estimator=pipeline_copy, method="sigmoid", cv=cal_cv)
        calibrated_model.fit(X_train, y_train)
        cal_prob = calibrated_model.predict_proba(X_val)
        cal_metrics = calculate_classification_metrics(y_val, np.argmax(cal_prob, axis=1), cal_prob)
        cal_loss = cal_metrics.get("log_loss", float("inf"))
    except Exception as e:
        logger.warning("Calibration attempt failed: %s", e)
        return pipeline, False, {"raw_log_loss": raw_loss, "calibrated_log_loss": raw_loss}

    should_calibrate = cal_loss < raw_loss
    logger.info("Calibration check -> Raw Log Loss: %.4f, Calibrated Log Loss: %.4f. Improve: %s",
                raw_loss, cal_loss, should_calibrate)

    if should_calibrate:
        return calibrated_model, True, {"raw_log_loss": raw_loss, "calibrated_log_loss": cal_loss}
    return pipeline, False, {"raw_log_loss": raw_loss, "calibrated_log_loss": cal_loss}


def run_noise_robustness_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_stds: Dict[str, float],
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model robustness against 0%, 5%, and 10% Gaussian noise injection."""
    rng = np.random.RandomState(seed)
    noise_levels = [0.0, 0.05, 0.10]
    results = {}

    for alpha in noise_levels:
        X_perturbed = X_test.copy()
        for col in X_test.columns:
            std_val = feature_stds.get(col, X_test[col].std())
            noise = rng.normal(0, alpha * std_val, size=len(X_test))
            X_perturbed[col] = X_perturbed[col] + noise

        y_pred = pipeline.predict(X_perturbed)
        y_prob = pipeline.predict_proba(X_perturbed) if hasattr(pipeline, "predict_proba") else None
        m = calculate_classification_metrics(y_test, y_pred, y_prob)

        label = f"{int(alpha * 100)}% noise"
        results[label] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "top_3_accuracy": m.get("top_3_accuracy", m["accuracy"]),
            "log_loss": m.get("log_loss"),
        }
        logger.info("[Robustness %s] Acc: %.4f | Macro-F1: %.4f", label, m["accuracy"], m["macro_f1"])

    return results


def compute_learning_curve_data(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute learning curve data over 5 train set fractions."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="f1_macro",
        n_jobs=-1,
        random_state=seed,
    )
    return train_sizes, train_scores, val_scores


def compute_calibration_curve_data(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute multiclass calibration reliability diagram data using top-1 predicted probability."""
    y_prob = pipeline.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correctness = (y_pred == y_test).astype(int)

    prob_true, prob_pred = calibration_curve(correctness, confidences, n_bins=n_bins, strategy="uniform")
    return prob_true, prob_pred


def compute_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int = 42,
    n_repeats: int = 10,
) -> Dict[str, Any]:
    """Compute permutation feature importance on held-out test set."""
    res = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return {
        "importances_mean": res.importances_mean.tolist(),
        "importances_std": res.importances_std.tolist(),
        "features": list(X_test.columns),
    }
