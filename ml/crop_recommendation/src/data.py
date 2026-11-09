import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("ml.crop_recommendation.data")

EXPECTED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"
EXPECTED_COLUMNS = EXPECTED_FEATURES + [TARGET_COLUMN]


def load_dataset(
    data_dir: Path | str,
    filename: str = "Crop_recommendation.csv",
    download_url: Optional[str] = None,
) -> pd.DataFrame:
    """Load Crop_recommendation.csv from data_dir, downloading it automatically if not found.

    Args:
        data_dir: Directory containing or designated to contain the dataset.
        filename: Name of the CSV file.
        download_url: Fallback URL to download if file does not exist locally.

    Returns:
        pd.DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If file is absent and no download URL is provided or reachable.
    """
    data_path = Path(data_dir) / filename
    if not data_path.exists():
        if download_url:
            logger.info("Dataset not found at %s. Attempting download from %s...", data_path, download_url)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            with open(data_path, "wb") as f:
                f.write(response.content)
            logger.info("Successfully downloaded dataset to %s (%d bytes).", data_path, len(response.content))
        else:
            raise FileNotFoundError(
                f"Dataset not found at {data_path} and no download_url was specified."
            )

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset from %s with shape %s", data_path, df.shape)
    return df


def validate_schema(df: pd.DataFrame, expected_classes: int = 22) -> None:
    """Perform strict schema validation on the crop dataset. Fails loudly on schema violations.

    Args:
        df: Input DataFrame.
        expected_classes: Expected number of unique classes.

    Raises:
        ValueError: If columns, types, nulls, or class counts violate expectations.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema violation: missing required columns {missing_cols}")

    # Check for unexpected extra columns
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        logger.warning("Dataset contains additional unexpected columns: %s", extra_cols)

    # Check missing values
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Dataset contains null values:\n{null_counts[null_counts > 0]}")

    # Check numeric types
    for feature in EXPECTED_FEATURES:
        if not np.issubdtype(df[feature].dtype, np.number):
            raise ValueError(f"Feature '{feature}' must be numeric, got {df[feature].dtype}")

    # Check unique classes
    unique_classes = df[TARGET_COLUMN].nunique()
    if unique_classes != expected_classes:
        logger.warning(
            "Expected %d unique classes, but found %d in dataset.",
            expected_classes,
            unique_classes,
        )


def compute_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute exploratory data analysis metrics, distributions, and outlier reports.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary summarizing summary stats, duplicates, outliers, and class balance.
    """
    duplicates_count = int(df.duplicated().sum())

    # Summary statistics
    describe_df = df[EXPECTED_FEATURES].describe().T
    summary_stats = describe_df.to_dict(orient="index")

    # Class distribution
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()

    # Outliers per feature via IQR (1.5 * IQR) - reported, not removed
    outlier_summary = {}
    for col in EXPECTED_FEATURES:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(len(outliers)),
            "outlier_percentage": float(len(outliers) / len(df) * 100),
        }

    return {
        "total_rows": int(len(df)),
        "duplicates_count": duplicates_count,
        "class_counts": class_counts,
        "features_summary": summary_stats,
        "outlier_summary": outlier_summary,
    }


def split_and_save_data(
    df: pd.DataFrame,
    data_dir: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    LabelEncoder,
    Dict[str, Dict[str, float]],
]:
    """Perform stratified 70/15/15 train/val/test split and persist split manifests and CSVs.

    Args:
        df: Clean validated DataFrame.
        data_dir: Directory where split manifests and CSVs will be saved.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val, label_encoder, input_ranges).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    X = df[EXPECTED_FEATURES].copy()
    y_raw = df[TARGET_COLUMN].copy()

    # Fit LabelEncoder on target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    indices = np.arange(len(df))

    # Split into train_val (85%) and test (15%)
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )

    # Relative val ratio in train_val: 0.15 / 0.85
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_relative_ratio,
        random_state=seed,
        stratify=y[idx_train_val],
    )

    # Manifests
    save_path = Path(data_dir) / "splits"
    save_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": seed,
        "train_size": len(idx_train),
        "val_size": len(idx_val),
        "test_size": len(idx_test),
        "train_indices": idx_train.tolist(),
        "val_indices": idx_val.tolist(),
        "test_indices": idx_test.tolist(),
    }
    with open(save_path / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save split CSVs for exact re-runs
    df.iloc[idx_train].to_csv(save_path / "train.csv", index=False)
    df.iloc[idx_val].to_csv(save_path / "val.csv", index=False)
    df.iloc[idx_test].to_csv(save_path / "test.csv", index=False)
    df.iloc[idx_train_val].to_csv(save_path / "train_val.csv", index=False)

    logger.info(
        "Saved data splits: Train=%d, Val=%d, Test=%d to %s",
        len(idx_train),
        len(idx_val),
        len(idx_test),
        save_path,
    )

    X_train = X.iloc[idx_train].reset_index(drop=True)
    y_train = y[idx_train]

    X_val = X.iloc[idx_val].reset_index(drop=True)
    y_val = y[idx_val]

    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_test = y[idx_test]

    X_train_val = X.iloc[idx_train_val].reset_index(drop=True)
    y_train_val = y[idx_train_val]

    # Calculate valid physical ranges from training data ONLY for serving input validation
    input_ranges = {}
    for col in EXPECTED_FEATURES:
        input_ranges[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
        }

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_val,
        y_train_val,
        label_encoder,
        input_ranges,
    )
