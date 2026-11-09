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
