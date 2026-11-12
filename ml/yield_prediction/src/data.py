import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("ml.yield_prediction.data")

REQUIRED_COLUMNS = [
    "Area",
    "Item",
    "Year",
    "hg/ha_yield",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]


def load_raw_dataset(
    data_dir: Path | str,
    filename: str = "yield_df.csv",
    download_url: Optional[str] = None,
) -> pd.DataFrame:
    """Load yield_df.csv from disk or download it automatically from mirror if absent.

    Args:
        data_dir: Directory to locate or store yield_df.csv.
        filename: CSV filename.
        download_url: Mirror URL if not present on disk.

    Returns:
        pd.DataFrame loaded from disk.
    """
    data_path = Path(data_dir) / filename
    if not data_path.exists():
        if download_url:
            logger.info("Dataset not found at %s. Downloading from %s...", data_path, download_url)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            with open(data_path, "wb") as f:
                f.write(response.content)
            logger.info("Successfully downloaded dataset (%d bytes) to %s.", len(response.content), data_path)
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path} and download_url was None.")

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset from %s with raw shape %s", data_path, df.shape)
    return df


def clean_and_validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean dataset, remove stray index columns, drop exact duplicates, and validate domain bounds.

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple of (clean_df, cleaning_summary_dict).
    """
    df_clean = df.copy()

    # 1. Drop stray index columns like 'Unnamed: 0'
    for col in list(df_clean.columns):
        if "unnamed" in col.lower():
            df_clean = df_clean.drop(columns=[col])
            logger.info("Dropped stray index column: %s", col)

    # Normalize column names: strip whitespace
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Verify presence of all required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df_clean.columns)
    if missing_cols:
        raise ValueError(f"Schema violation: missing required columns {missing_cols}")

    initial_rows = len(df_clean)

    # 2. Remove exact duplicate rows
    duplicates_count = int(df_clean.duplicated().sum())
    if duplicates_count > 0:
        logger.info("Identified %d exact duplicate rows. Removing duplicates...", duplicates_count)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 3. Check for null values
    null_counts = df_clean[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Dataset contains null values:\n{null_counts[null_counts > 0]}")

    # 4. Check for impossible domain values
    if (df_clean["hg/ha_yield"] <= 0).any():
        raise ValueError("Dataset contains non-positive crop yield values (hg/ha_yield <= 0).")
    if (df_clean["average_rain_fall_mm_per_year"] < 0).any():
        raise ValueError("Dataset contains negative rainfall values.")
    if (df_clean["pesticides_tonnes"] < 0).any():
        raise ValueError("Dataset contains negative pesticide values.")

    # 5. Type enforcement
    df_clean["Year"] = df_clean["Year"].astype(int)
    df_clean["average_rain_fall_mm_per_year"] = pd.to_numeric(df_clean["average_rain_fall_mm_per_year"], errors="coerce")
    df_clean["pesticides_tonnes"] = pd.to_numeric(df_clean["pesticides_tonnes"], errors="coerce")
    df_clean["avg_temp"] = pd.to_numeric(df_clean["avg_temp"], errors="coerce")
    df_clean["hg/ha_yield"] = pd.to_numeric(df_clean["hg/ha_yield"], errors="coerce")
    df_clean["Area"] = df_clean["Area"].astype(str).str.strip()
    df_clean["Item"] = df_clean["Item"].astype(str).str.strip()

    # Drop any rows where coercion created NaNs (e.g. malformed string numbers)
    df_clean = df_clean.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    # 6. Compute skewness summary
    skew_summary = {
        "yield_raw_skew": float(df_clean["hg/ha_yield"].skew()),
        "yield_log1p_skew": float(np.log1p(df_clean["hg/ha_yield"]).skew()),
        "pesticides_raw_skew": float(df_clean["pesticides_tonnes"].skew()),
        "pesticides_log1p_skew": float(np.log1p(df_clean["pesticides_tonnes"]).skew()),
        "rainfall_skew": float(df_clean["average_rain_fall_mm_per_year"].skew()),
        "temp_skew": float(df_clean["avg_temp"].skew()),
    }

    cleaning_summary = {
        "initial_rows": initial_rows,
        "clean_rows": len(df_clean),
        "duplicates_removed": duplicates_count,
        "unique_areas_count": int(df_clean["Area"].nunique()),
        "unique_items_count": int(df_clean["Item"].nunique()),
        "year_min": int(df_clean["Year"].min()),
        "year_max": int(df_clean["Year"].max()),
        "skewness": skew_summary,
    }
    logger.info("Cleaning complete: %d rows retained across %d countries and %d crops (%d-%d).",
                len(df_clean), cleaning_summary["unique_areas_count"], cleaning_summary["unique_items_count"],
                cleaning_summary["year_min"], cleaning_summary["year_max"])

    return df_clean, cleaning_summary


def split_temporal_data(
    df: pd.DataFrame,
    data_dir: Path | str,
    train_end_year: int = 2008,
    val_start_year: int = 2009,
    val_end_year: int = 2010,
    test_start_year: int = 2011,
    test_end_year: int = 2013,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Dict[str, float]],
    Dict[str, List[str]],
]:
    """Perform non-leaking primary temporal split (Train <= 2008, Val 2009-2010, Test 2011-2013).

    Args:
        df: Clean DataFrame.
        data_dir: Destination directory for manifests and split CSVs.
        train_end_year: Final year of training split.
        val_start_year: Start year of validation split.
        val_end_year: End year of validation split.
        test_start_year: Start year of held-out test split.
        test_end_year: End year of held-out test split.

    Returns:
        Tuple of (df_train, df_val, df_test, df_train_val, input_ranges, valid_categories).
    """
    train_mask = (df["Year"] <= train_end_year)
    val_mask = (df["Year"] >= val_start_year) & (df["Year"] <= val_end_year)
    test_mask = (df["Year"] >= test_start_year) & (df["Year"] <= test_end_year)
    train_val_mask = train_mask | val_mask

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)
    df_train_val = df[train_val_mask].copy().reset_index(drop=True)

    splits_dir = Path(data_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Save indices manifests
    manifest = {
        "train_years": [int(df_train["Year"].min()), train_end_year],
        "val_years": [val_start_year, val_end_year],
        "test_years": [test_start_year, test_end_year],
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "train_indices": df.index[train_mask].tolist(),
        "val_indices": df.index[val_mask].tolist(),
        "test_indices": df.index[test_mask].tolist(),
    }
    with open(splits_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save split CSVs for full evaluation reproducibility
    df_train.to_csv(splits_dir / "train.csv", index=False)
    df_val.to_csv(splits_dir / "val.csv", index=False)
    df_test.to_csv(splits_dir / "test.csv", index=False)
    df_train_val.to_csv(splits_dir / "train_val.csv", index=False)

    logger.info("Saved temporal splits: Train=%d, Val=%d, Test=%d to %s",
                len(df_train), len(df_val), len(df_test), splits_dir)

    # Extract input ranges and valid categories strictly from training split
    numeric_features = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    input_ranges = {}
    for col in numeric_features:
        input_ranges[col] = {
            "min": float(df_train[col].min()),
            "max": float(df_train[col].max()),
            "mean": float(df_train[col].mean()),
            "std": float(df_train[col].std()),
        }

    valid_categories = {
        "crops": sorted(df_train["Item"].unique().tolist()),
        "areas": sorted(df_train["Area"].unique().tolist()),
    }

    return df_train, df_val, df_test, df_train_val, input_ranges, valid_categories
