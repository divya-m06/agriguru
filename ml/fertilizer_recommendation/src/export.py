from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List
import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

logger = logging.getLogger("ml.fertilizer_recommendation.export")


def get_git_commit_hash() -> str:
    """Safely obtain current git commit hash or return fallback."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "unversioned_workspace"


def export_fertilizer_artifacts(
    pipeline: Pipeline,
    classes: List[str],
    feature_order: List[str],
    allowed_categories: Dict[str, List[str]],
    numeric_ranges: Dict[str, Dict[str, float]],
    cv_metrics: Dict[str, Any],
    nested_metrics: Dict[str, Any],
    permutation_test_metrics: Dict[str, Any],
    chosen_model_name: str,
    chosen_params: Dict[str, Any],
    output_dir: Path | str,
    config: Dict[str, Any],
    seed: int,
) -> Path:
    """Export all production artifacts for deployment and reproducibility.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        classes: Ordered list of fertilizer class names.
        feature_order: List of required input feature names.
        allowed_categories: Dictionary of valid soil and crop categories.
        numeric_ranges: Dictionary of min, max, mean, std bounds.
        cv_metrics: Candidate CV metrics summary.
        nested_metrics: Nested CV evaluation metrics.
        permutation_test_metrics: Null distribution and empirical p-value.
        chosen_model_name: Name of selected model.
        chosen_params: Final hyperparameter configuration.
        output_dir: Destination directory.
        config: Full YAML config.
        seed: Random seed.

    Returns:
        Path to output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save main classification pipeline artifact
    model_file = out_path / "fertilizer_model.joblib"
    joblib.dump(pipeline, model_file)
    logger.info("Saved fertilizer model pipeline artifact to %s", model_file)

    # 2. Save classes.json
    with open(out_path / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    # 3. Save feature_order.json
    with open(out_path / "feature_order.json", "w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2)

    # 4. Save allowed_categories.json
    with open(out_path / "allowed_categories.json", "w", encoding="utf-8") as f:
        json.dump(allowed_categories, f, indent=2)

    # 5. Save numeric_ranges.json
    with open(out_path / "numeric_ranges.json", "w", encoding="utf-8") as f:
        json.dump(numeric_ranges, f, indent=2)

    # 6. Build environment library versions
    import pandas as pd
    lib_versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for opt_pkg in ["xgboost", "catboost", "optuna", "scipy"]:
        try:
            mod = __import__(opt_pkg)
            lib_versions[opt_pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # Example payload
    example_input = {
        "temperature": 26.0,
        "humidity": 52.0,
        "moisture": 38.0,
        "soil_type": allowed_categories["soil_types"][0],
        "crop_type": allowed_categories["crop_types"][0],
        "nitrogen": 37.0,
        "potassium": 0.0,
        "phosphorous": 0.0,
    }

    # 7. Metadata
    metadata = {
        "model_name": chosen_model_name,
        "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "seed": seed,
        "libraries": lib_versions,
        "dataset_characteristics": {
            "num_samples": 99,
            "num_classes": len(classes),
            "classes": classes,
            "note": "Extremely small dataset (99 rows). Evaluation conducted via Repeated Stratified Nested CV.",
        },
        "chosen_hyperparameters": chosen_params,
        "nested_cv_metrics": nested_metrics,
        "candidate_cv_metrics": cv_metrics,
        "label_permutation_test": {
            "observed_macro_f1": permutation_test_metrics.get("observed_score"),
            "null_macro_f1_mean": permutation_test_metrics.get("null_mean"),
            "p_value": permutation_test_metrics.get("p_value"),
            "n_permutations": permutation_test_metrics.get("n_permutations"),
        },
        "vocabulary_mapping_note": (
            "Crop categories (e.g. 'Paddy', 'Millets', 'Oil seeds', 'Ground Nuts') reflect raw dataset nomenclature "
            "and differ from Crop Recommendation (Model 1) / Yield (Model 3). Harmonization should occur at the service layer."
        ),
        "example_input": example_input,
        "config": config,
    }

    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", out_path / "metadata.json")
    return out_path
