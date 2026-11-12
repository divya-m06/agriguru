import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("ml.fertilizer_recommendation.data")

DEFAULT_COLUMN_RENAMES = {
    "Temparature": "temperature",
    "Humidity ": "humidity",
    "Humidity": "humidity",
    "Moisture": "moisture",
    "Soil Type": "soil_type",
    "Crop Type": "crop_type",
    "Nitrogen": "nitrogen",
    "Potassium": "potassium",
    "Phosphorous": "phosphorous",
    "Fertilizer Name": "fertilizer_name",
}

NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "moisture",
    "nitrogen",
    "potassium",
    "phosphorous",
]

CATEGORICAL_FEATURES = [
    "soil_type",
    "crop_type",
]


def load_raw_dataset(
    data_dir: Path | str,
    filename: str = "Fertilizer Prediction.csv",
    download_urls: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load raw fertilizer dataset from disk, or auto-download from verified mirrors if absent.

    Args:
        data_dir: Directory containing or to contain dataset.
        filename: Dataset filename.
        download_urls: Fallback URLs to download from.

    Returns:
        pd.DataFrame of raw loaded data.
    """
    data_path = Path(data_dir) / filename
    if not data_path.exists():
        if not download_urls:
            raise FileNotFoundError(f"Dataset not found at {data_path} and download_urls is empty.")

        download_success = False
        data_path.parent.mkdir(parents=True, exist_ok=True)

        for url in download_urls:
            try:
                logger.info("Attempting download from mirror: %s", url)
                resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                with open(data_path, "wb") as f:
                    f.write(resp.content)
                logger.info("Successfully downloaded dataset (%d bytes) to %s", len(resp.content), data_path)
                download_success = True
                break
            except Exception as e:
                logger.warning("Failed download from %s: %s", url, e)

        if not download_success:
            raise RuntimeError(f"Failed to download dataset from all provided mirrors: {download_urls}")

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset from %s with raw shape %s", data_path, df.shape)
    return df


def clean_and_validate_data(
    df: pd.DataFrame,
    column_renames: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], LabelEncoder]:
    """Clean column names, normalize text, validate physical bounds, and encode target classes.

    Args:
        df: Raw DataFrame.
        column_renames: Optional mapping for column renaming.

    Returns:
        Tuple of (clean_df, data_summary_dict, fitted_label_encoder).
    """
    df_clean = df.copy()

    # 1. Clean and standardize column names
    renames = column_renames or DEFAULT_COLUMN_RENAMES
    # Strip whitespace from raw columns first
    clean_cols = {}
    for c in df_clean.columns:
        c_stripped = c.strip()
        matched = False
        for raw_k, new_k in renames.items():
            if raw_k.strip().lower() == c_stripped.lower():
                clean_cols[c] = new_k
                matched = True
                break
        if not matched:
            clean_cols[c] = c_stripped.lower().replace(" ", "_")

    df_clean = df_clean.rename(columns=clean_cols)

    # 2. Verify all required columns are present
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["fertilizer_name"]
    missing_cols = set(required_cols) - set(df_clean.columns)
    if missing_cols:
        raise ValueError(f"Schema violation: missing required columns {missing_cols}")

    initial_rows = len(df_clean)

    # 3. Clean string columns (strip whitespace)
    for cat_col in CATEGORICAL_FEATURES + ["fertilizer_name"]:
        df_clean[cat_col] = df_clean[cat_col].astype(str).str.strip()

    # 4. Check exact duplicates
    duplicates_count = int(df_clean.duplicated().sum())
    if duplicates_count > 0:
        logger.info("Identified %d exact duplicate rows in dataset.", duplicates_count)

    # 5. Check null values
    null_counts = df_clean[required_cols].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Dataset contains null values:\n{null_counts[null_counts > 0]}")

    # 6. Numeric conversions and domain validity checks
    for num_col in NUMERIC_FEATURES:
        df_clean[num_col] = pd.to_numeric(df_clean[num_col], errors="raise")

    if (df_clean["nitrogen"] < 0).any():
        raise ValueError("Dataset contains negative Nitrogen values.")
    if (df_clean["potassium"] < 0).any():
        raise ValueError("Dataset contains negative Potassium values.")
    if (df_clean["phosphorous"] < 0).any():
        raise ValueError("Dataset contains negative Phosphorous values.")
    if (df_clean["moisture"] < 0).any():
        raise ValueError("Dataset contains negative Moisture values.")
    if (df_clean["humidity"] < 0).any() or (df_clean["humidity"] > 100).any():
        raise ValueError("Dataset contains Humidity values outside [0, 100]%.")

    # 7. Encode target variable
    label_encoder = LabelEncoder()
    df_clean["fertilizer_label"] = label_encoder.fit_transform(df_clean["fertilizer_name"])

    # Extract ranges and categories
    numeric_ranges = {}
    for col in NUMERIC_FEATURES:
        numeric_ranges[col] = {
            "min": float(df_clean[col].min()),
            "max": float(df_clean[col].max()),
            "mean": float(df_clean[col].mean()),
            "std": float(df_clean[col].std()),
        }

    allowed_categories = {
        "soil_types": sorted(df_clean["soil_type"].unique().tolist()),
        "crop_types": sorted(df_clean["crop_type"].unique().tolist()),
    }

    class_counts = df_clean["fertilizer_name"].value_counts().to_dict()

    summary = {
        "initial_rows": initial_rows,
        "clean_rows": len(df_clean),
        "duplicates_count": duplicates_count,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "class_counts": class_counts,
        "allowed_categories": allowed_categories,
        "numeric_ranges": numeric_ranges,
    }

    logger.info("Cleaning complete: %d rows across %d fertilizer classes, %d soil types, and %d crop types.",
                len(df_clean), summary["num_classes"],
                len(allowed_categories["soil_types"]),
                len(allowed_categories["crop_types"]))

    return df_clean, summary, label_encoder


def create_stratified_sanity_holdout(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
    data_dir: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified sanity holdout partition (~20% of data, ~20 rows).

    Note: With N=99, holdout evaluations are subject to extreme variance (+/- 5% per point).
    Primary model benchmarking relies on Repeated Stratified 5-Fold Cross-Validation.

    Args:
        df: Clean DataFrame.
        test_size: Holdout fraction.
        seed: Random seed.
        data_dir: Optional directory to persist CSV manifests.

    Returns:
        Tuple of (df_train_val, df_holdout).
    """
    df_train_val, df_holdout = train_test_split(
        df,
        test_size=test_size,
        stratify=df["fertilizer_label"],
        random_state=seed,
    )
    df_train_val = df_train_val.reset_index(drop=True)
    df_holdout = df_holdout.reset_index(drop=True)

    if data_dir:
        splits_dir = Path(data_dir) / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        df_train_val.to_csv(splits_dir / "train_val.csv", index=False)
        df_holdout.to_csv(splits_dir / "holdout_test.csv", index=False)

        manifest = {
            "total_samples": len(df),
            "train_val_samples": len(df_train_val),
            "holdout_samples": len(df_holdout),
            "train_val_indices": df_train_val.index.tolist(),
            "holdout_indices": df_holdout.index.tolist(),
            "note": "Holdout is a sanity check only; repeated stratified CV is primary validation.",
        }
        with open(splits_dir / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    logger.info("Created sanity holdout: TrainVal=%d rows, Holdout=%d rows",
                len(df_train_val), len(df_holdout))

    return df_train_val, df_holdout
