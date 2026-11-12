import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

logger = logging.getLogger("ml.disease_detection.evaluate")


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    k: int = 3,
) -> Dict[str, Any]:
    """Calculate accuracy, macro/weighted precision, recall, F1, top-k accuracy,
    and per-class precision/recall/F1 metrics.
    """
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        top_k_acc = float(top_k_accuracy_score(y_true, y_prob, k=k, labels=np.arange(len(class_names))))
    except Exception:
        top_k_acc = acc

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for idx, name in enumerate(class_names):
        if name in report_dict:
            per_class[name] = {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1-score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "top_k_accuracy": top_k_acc,
        "per_class": per_class,
        "classification_report": report_dict,
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Expected Calibration Error (ECE) and bin statistics."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_counts.append(int(np.sum(in_bin)))

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(float(accuracy_in_bin))
            bin_confs.append(float(avg_confidence_in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lower + bin_upper) / 2))

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


class TemperatureScaler:
    """Post-hoc confidence calibration via temperature scaling optimized on validation data."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Optimize temperature T > 0 using negative log-likelihood (cross entropy)."""
        num_classes = logits.shape[1]
        y_one_hot = np.eye(num_classes)[y_true]

        def _nll(T: np.ndarray) -> float:
            scaled = logits / max(1e-3, float(T[0]))
            # Softmax
            exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            return float(-np.mean(np.sum(y_one_hot * np.log(probs), axis=1)))

        res = minimize(_nll, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        logger.info("Fitted temperature scaling parameter: T = %.4f (Optimized NLL: %.4f)", self.temperature, res.fun)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply fitted temperature to logits and return calibrated probabilities."""
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def plot_reliability_diagram(
    bin_accs: np.ndarray,
    bin_confs: np.ndarray,
    ece: float,
    output_path: Path | str,
    title: str = "Reliability Diagram",
) -> None:
    """Save reliability diagram (calibration curve)."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax.plot(bin_confs, bin_accs, "s-", color="#2b5c8f", label=f"Model (ECE={ece:.4f})")
    ax.bar(bin_confs, bin_accs, width=0.05, alpha=0.2, color="#2b5c8f", edgecolor="#2b5c8f")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path | str,
) -> None:
    """Save normalized confusion matrix heatmap for 38 PlantVillage classes."""
    cm_norm = cm.astype("float") / np.maximum(1, cm.sum(axis=1)[:, np.newaxis])
    fig, ax = plt.subplots(figsize=(18, 16), dpi=150)
    cax = ax.matshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("PlantVillage Normalized Confusion Matrix (38 Classes)", fontsize=14, pad=20)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def extract_top_confused_pairs(
    cm: np.ndarray,
    class_names: List[str],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Identify the most frequently confused pairs of classes (excluding diagonal)."""
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                    "error_rate": float(cm[i, j] / max(1, cm[i].sum())),
                })

    pairs = sorted(pairs, key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]
