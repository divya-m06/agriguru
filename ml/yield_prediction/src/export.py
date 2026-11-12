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

logger = logging.getLogger("ml.yield_prediction.export")


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


def export_yield_artifacts(
    pipeline: Pipeline,
    feature_order: List[str],
    valid_categories: Dict[str, List[str]],
    input_ranges: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, float],
    per_crop_metrics: Dict[str, Dict[str, float]],
    group_kfold_metrics: Dict[str, float],
    leaky_random_metrics: Dict[str, float],
    chosen_model_name: str,
    variant_name: str,
    output_dir: Path | str,
    config: Dict[str, Any],
    seed: int,
    lower_quantile_pipe: Optional[Pipeline] = None,
    upper_quantile_pipe: Optional[Pipeline] = None,
) -> Path:
    """Export all production artifacts for deployment and reproducibility."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save main regression model
    model_file = out_path / "yield_model.joblib"
    joblib.dump(pipeline, model_file)
    logger.info("Saved yield model pipeline artifact to %s", model_file)

    # Optional quantile models
    if lower_quantile_pipe is not None:
        joblib.dump(lower_quantile_pipe, out_path / "quantile_low_model.joblib")
    if upper_quantile_pipe is not None:
        joblib.dump(upper_quantile_pipe, out_path / "quantile_high_model.joblib")

    # 2. Save feature_order.json
    with open(out_path / "feature_order.json", "w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2)

    # 3. Save valid_categories.json (crops and areas)
    with open(out_path / "valid_categories.json", "w", encoding="utf-8") as f:
        json.dump(valid_categories, f, indent=2)

    # 4. Save input_ranges.json
    with open(out_path / "input_ranges.json", "w", encoding="utf-8") as f:
        json.dump(input_ranges, f, indent=2)

    # 5. Environment versions
    import pandas as pd
    lib_versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for opt_pkg in ["xgboost", "lightgbm", "catboost", "optuna"]:
        try:
            mod = __import__(opt_pkg)
            lib_versions[opt_pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # Sample inference payload
    example_input = {
        "crop": "Maize",
        "year": 2013,
        "rainfall_mm": float(input_ranges["average_rain_fall_mm_per_year"]["mean"]),
        "pesticides_tonnes": float(input_ranges["pesticides_tonnes"]["mean"]),
        "temperature_c": float(input_ranges["avg_temp"]["mean"]),
    }
    if "Area" in feature_order:
        example_input["area"] = valid_categories["areas"][0]

    # 6. Save metadata.json
    metadata = {
        "model_name": chosen_model_name,
        "variant": variant_name,
        "artifact_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "seed": seed,
        "split_description": "Strict Temporal Split: Train <=2008, Val 2009-2010, Test 2011-2013",
        "target_transform": "TransformedTargetRegressor(func=log1p, inverse_func=expm1)",
        "feature_order": feature_order,
        "valid_crops_count": len(valid_categories.get("crops", [])),
        "valid_areas_count": len(valid_categories.get("areas", [])),
        "library_versions": lib_versions,
        "test_metrics_temporal": test_metrics,
        "group_kfold_metrics": group_kfold_metrics,
        "leaky_random_split_metrics": leaky_random_metrics,
        "per_crop_metrics": per_crop_metrics,
        "full_config": config,
        "example_input": example_input,
    }

    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", out_path / "metadata.json")
    return out_path
