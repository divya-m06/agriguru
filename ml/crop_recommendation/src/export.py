from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("ml.crop_recommendation.export")


def get_git_commit_hash() -> str:
    """Safely obtain current git commit hash or return a fallback."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "unversioned_workspace"


def export_artifacts(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    feature_order: List[str],
    input_ranges: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Any],
    cv_comparison_table: List[Dict[str, Any]],
    chosen_model_name: str,
    output_dir: Path | str,
    config: Dict[str, Any],
    seed: int,
    noise_robustness: Dict[str, Any],
    feature_engineering_used: bool,
    run_mode: str = "full",
    train_row_count: Optional[int] = None,
) -> Path:
    """Export all production artifacts for deployment and reproducibility.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        label_encoder: Fitted LabelEncoder.
        feature_order: List of feature names in required ingestion order.
        input_ranges: Min/max/mean/std per feature from training data.
        test_metrics: Evaluation metrics on the held-out test set.
        cv_comparison_table: CV summary of all candidate models.
        chosen_model_name: Name of chosen algorithm.
        output_dir: Destination directory for artifacts.
        config: Full configuration dictionary.
        seed: Random seed used.
        noise_robustness: Robustness evaluation results.
        feature_engineering_used: Boolean indicating if feature engineering is active.

    Returns:
        Path to output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save Pipeline artifact
    model_file = out_path / "crop_model.joblib"
    joblib.dump(pipeline, model_file)
    logger.info("Saved pipeline model artifact to %s", model_file)

    # 2. Save LabelEncoder artifact
    le_file = out_path / "label_encoder.joblib"
    joblib.dump(label_encoder, le_file)
    logger.info("Saved label encoder artifact to %s", le_file)

    # 3. Save classes.json
    classes_list = [str(c) for c in label_encoder.classes_]
    classes_file = out_path / "classes.json"
    with open(classes_file, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, indent=2)

    # 4. Save feature_order.json
    features_file = out_path / "feature_order.json"
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2)

    # 5. Save input_ranges.json
    ranges_file = out_path / "input_ranges.json"
    with open(ranges_file, "w", encoding="utf-8") as f:
        json.dump(input_ranges, f, indent=2)

    # 6. Gather environment versions
    import pandas as pd
    lib_versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for opt_pkg in ["xgboost", "lightgbm"]:
        try:
            mod = __import__(opt_pkg)
            lib_versions[opt_pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # Sample inference payload
    example_input = {feat: round(input_ranges[feat]["mean"], 2) for feat in feature_order}

    # 7. Save metadata.json
    metadata = {
        "model_name": chosen_model_name,
        "artifact_version": "1.0.0",
        "run_mode": run_mode,
        "train_row_count": train_row_count,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "seed": seed,
        "feature_engineering_used": feature_engineering_used,
        "feature_order": feature_order,
        "classes_count": len(classes_list),
        "classes": classes_list,
        "library_versions": lib_versions,
        "test_metrics": {
            "accuracy": test_metrics.get("accuracy"),
            "macro_f1": test_metrics.get("macro_f1"),
            "top_3_accuracy": test_metrics.get("top_3_accuracy"),
            "log_loss": test_metrics.get("log_loss"),
            "brier_score": test_metrics.get("brier_score"),
        },
        "noise_robustness": noise_robustness,
        "cv_comparison": cv_comparison_table,
        "full_config": config,
        "example_input": example_input,
    }

    metadata_file = out_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", metadata_file)
    return out_path
