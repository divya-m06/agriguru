from typing import Any, Dict, Optional, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """Calculate multiclass Brier score: mean squared difference between predicted probabilities
    and one-hot encoded ground truth.

    Args:
        y_true: 1D array of true integer class labels.
        y_prob: 2D array of predicted class probabilities shape (N, C).
        n_classes: Total number of classes.

    Returns:
        Float Brier score (lower is better, range [0, 2]).
    """
    n_samples = len(y_true)
    one_hot = np.zeros((n_samples, n_classes), dtype=float)
    one_hot[np.arange(n_samples), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def calculate_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[Sequence[Any]] = None,
    k: int = 3,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics including top-k accuracy, log loss,
    macro-F1, and multiclass Brier score.

    Args:
        y_true: True class labels (integer encoded).
        y_pred: Predicted class labels (integer encoded).
        y_prob: Optional predicted class probabilities (N, C).
        classes: Unique class names or label list.
        k: Value for top-k accuracy.

    Returns:
        Dictionary of computed scalar metrics and structured reports.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    if y_prob is not None:
        n_classes = y_prob.shape[1]
        all_labels = list(range(n_classes))

        # Top-k accuracy (handle case where k >= n_classes)
        effective_k = min(k, n_classes)
        try:
            top_k_acc = float(top_k_accuracy_score(y_true_arr, y_prob, k=effective_k, labels=all_labels))
            metrics[f"top_{effective_k}_accuracy"] = top_k_acc
        except Exception:
            metrics[f"top_{effective_k}_accuracy"] = acc

        # Multiclass Log Loss
        try:
            ll = float(log_loss(y_true_arr, y_prob, labels=all_labels))
            metrics["log_loss"] = ll
        except Exception:
            metrics["log_loss"] = None

        # Multiclass Brier score
        try:
            brier = multiclass_brier_score(y_true_arr, y_prob, n_classes)
            metrics["brier_score"] = brier
        except Exception:
            metrics["brier_score"] = None

    # Classification report dict
    target_names = [str(c) for c in classes] if classes is not None else None
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report_dict

    # Raw and normalized confusion matrices
    cm_raw = confusion_matrix(y_true_arr, y_pred_arr)
    cm_norm = confusion_matrix(y_true_arr, y_pred_arr, normalize="true")
    metrics["confusion_matrix_raw"] = cm_raw.tolist()
    metrics["confusion_matrix_normalized"] = cm_norm.tolist()

    return metrics


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE = (100% / n) * sum( 2 * |y - y_hat| / (|y| + |y_hat| + eps) )
    Range: 0% to 200%. Handles low/zero yields stably without division by zero.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr) + eps
    smape = np.mean(2.0 * np.abs(y_pred_arr - y_true_arr) / denominator) * 100.0
    return float(smape)


def calculate_regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Calculate core regression metrics in hg/ha and t/ha.

    Args:
        y_true: Ground truth target values (hg/ha).
        y_pred: Predicted target values (hg/ha).

    Returns:
        Dictionary containing R2, RMSE (hg/ha & t/ha), MAE (hg/ha & t/ha), and sMAPE (%).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    smape = calculate_smape(y_true_arr, y_pred_arr)

    # 1 tonne/hectare (t/ha) = 10,000 hectograms/hectare (hg/ha)
    return {
        "r2": r2,
        "rmse_hg_ha": rmse,
        "mae_hg_ha": mae,
        "rmse_t_ha": rmse / 10000.0,
        "mae_t_ha": mae / 10000.0,
        "smape_pct": smape,
    }


def calculate_per_crop_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    crops: Sequence[str] | np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute per-crop regression metrics breakdown to diagnose variance dominance across crops.

    Args:
        y_true: Ground truth yields.
        y_pred: Predicted yields.
        crops: Array of crop (Item) labels corresponding to each sample.

    Returns:
        Dictionary mapping each crop name to its specific regression metrics and sample count.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    crops_arr = np.asarray(crops)

    unique_crops = np.unique(crops_arr)
    breakdown = {}

    for crop in unique_crops:
        mask = (crops_arr == crop)
        if np.sum(mask) == 0:
            continue
        crop_metrics = calculate_regression_metrics(y_true_arr[mask], y_pred_arr[mask])
        crop_metrics["sample_count"] = int(np.sum(mask))
    return breakdown


def calculate_confidence_interval(
    scores: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Calculate sample mean, std, standard error, and two-sided t-distribution confidence interval.

    Args:
        scores: Array of metric values across folds or repeats.
        confidence: Desired confidence level (default: 0.95).

    Returns:
        Dictionary with mean, std, sem, ci_lower, and ci_upper.
    """
    import scipy.stats as stats

    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0

    if n > 1 and std > 0:
        h = sem * float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    else:
        h = 0.0

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": mean - h,
        "ci_upper": mean + h,
    }

