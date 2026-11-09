from pathlib import Path
from typing import List, Optional, Sequence
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: Sequence[str],
    output_path: Path | str,
    title: str = "Feature Correlation Heatmap",
) -> None:
    """Plot Pearson correlation heatmap for input features."""
    plt.figure(figsize=(8, 6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, cbar=True)
    plt.title(title, fontsize=14, pad=12)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Sequence[str],
    output_path: Path | str,
    target_col: Optional[str] = None,
) -> None:
    """Plot histograms and KDEs for input features."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(data=df, x=feature, kde=True, ax=ax, color="#1f77b4")
        ax.set_title(f"Distribution of {feature}", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature Distributions (Overall)", fontsize=14, y=1.01)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm_normalized: np.ndarray,
    class_names: Sequence[str],
    output_path: Path | str,
    title: str = "Normalized Confusion Matrix",
) -> None:
    """Plot normalized confusion matrix heatmap."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    output_path: Path | str,
    title: str = "Learning Curve",
    metric_name: str = "Accuracy",
) -> None:
    """Plot training and validation scores against training set size."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", color="#1f77b4", label=f"Training {metric_name}")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#1f77b4",
    )
    plt.plot(train_sizes, val_mean, "s-", color="#2ca02c", label=f"CV {metric_name}")
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#2ca02c",
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Training Samples", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curve(
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
    output_path: Path | str,
    title: str = "Multiclass Calibration Curve (Top-1 Probability)",
) -> None:
    """Plot reliability diagram comparing predicted probability with observed empirical frequency."""
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", color="#d62728", label="Model Calibration")
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Empirical True Fraction", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    feature_names: Sequence[str],
    importances_mean: np.ndarray,
    importances_std: np.ndarray,
    output_path: Path | str,
    title: str = "Permutation Feature Importance",
) -> None:
    """Plot horizontal bar chart of permutation feature importances."""
    sorted_idx = np.argsort(importances_mean)
    pos = np.arange(sorted_idx.shape[0])

    plt.figure(figsize=(9, 6))
    plt.barh(
        pos,
        importances_mean[sorted_idx],
        xerr=importances_std[sorted_idx],
        align="center",
        color="#3b528b",
        ecolor="gray",
        capsize=4,
    )
    plt.yticks(pos, [feature_names[i] for i in sorted_idx], fontsize=11)
    plt.xlabel("Mean Decrease in Macro-F1", fontsize=12)
    plt.title(title, fontsize=14, pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_noise_robustness(
    noise_levels: Sequence[str],
    accuracies: Sequence[float],
    f1_scores: Sequence[float],
    output_path: Path | str,
    title: str = "Model Noise Robustness (Test Set Perturbation)",
) -> None:
    """Plot metric degradation under feature Gaussian noise injection."""
    x = np.arange(len(noise_levels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, accuracies, width, label="Accuracy", color="#1f77b4")
    plt.bar(x + width / 2, f1_scores, width, label="Macro-F1", color="#ff7f0e")

    for i in range(len(noise_levels)):
        plt.text(x[i] - width / 2, accuracies[i] + 0.01, f"{accuracies[i]:.3f}", ha="center", fontsize=9)
        plt.text(x[i] + width / 2, f1_scores[i] + 0.01, f"{f1_scores[i]:.3f}", ha="center", fontsize=9)

    plt.xticks(x, noise_levels, fontsize=11)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.title(title, fontsize=14, pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
