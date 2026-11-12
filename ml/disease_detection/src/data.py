import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger("ml.disease_detection.data")

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def parse_plantvillage_label(folder_name: str) -> Tuple[str, str]:
    """Parse raw PlantVillage folder name into clean (crop_name, disease_name).

    Examples:
        'Tomato___Late_blight' -> ('Tomato', 'Late blight')
        'Corn_(maize)___healthy' -> ('Corn (maize)', 'healthy')
        'Apple___Cedar_apple_rust' -> ('Apple', 'Cedar apple rust')
    """
    if "___" in folder_name:
        crop_part, disease_part = folder_name.split("___", 1)
    elif "__" in folder_name:
        crop_part, disease_part = folder_name.split("__", 1)
    else:
        crop_part, disease_part = folder_name, "unknown"

    clean_crop = crop_part.replace("_", " ").strip()
    clean_disease = disease_part.replace("_", " ").strip()
    return clean_crop, clean_disease


def discover_color_image_directory(data_dir: Path | str, min_classes: int = 2) -> Path:
    """Find the root directory containing original color PlantVillage images,
    skipping grayscale and segmented variants.
    """
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base_path}")

    # Check direct common subpaths
    candidates = [
        base_path / "color",
        base_path / "plantvillage dataset" / "color",
        base_path / "PlantVillage" / "color",
        base_path / "PlantVillage",
        base_path,
    ]

    for cand in candidates:
        if cand.is_dir():
            subdirs = [d for d in cand.iterdir() if d.is_dir()]
            # Look for subdirectories containing '___' or characteristic PlantVillage class names
            pv_subdirs = [d for d in subdirs if "___" in d.name and "gray" not in d.name.lower() and "segmented" not in d.name.lower()]
            if len(pv_subdirs) >= min_classes:
                logger.info("Discovered PlantVillage color image directory at: %s (%d classes)", cand, len(pv_subdirs))
                return cand

    # If no candidate matched, raise informative error
    raise ValueError(
        f"Could not locate PlantVillage color image subfolder under {base_path}. "
        "Expected subfolders with format '<Crop>___<Disease>'."
    )


def compute_perceptual_hash(image_path: Path | str, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) for an image.
    Falls back to a fast deterministic downsampled pixel hash if imagehash is not installed.
    """
    if HAS_IMAGEHASH:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img, hash_size=hash_size))
    else:
        # Standalone lightweight fallback 64-bit mean perceptual hash
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(gray, dtype=np.float32).ravel()
            avg = pixels.mean()
            bits = "".join(["1" if p > avg else "0" for p in pixels])
            return hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Calculate Hamming distance between two hexadecimal hash strings."""
    try:
        val1 = int(hex_hash1, 16)
        val2 = int(hex_hash2, 16)
        return bin(val1 ^ val2).count("1")
    except ValueError:
        return 999


class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression for duplicate clustering."""

    def __init__(self, elements: List[Any]):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, item: Any) -> Any:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: Any, item2: Any) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 == root2:
            return
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1


def build_dataset_manifest(
    color_dir: Path | str,
    output_manifest_path: Optional[Path | str] = None,
    hamming_threshold: int = 4,
    valid_extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
    quick: bool = False,
    quick_sample_size: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Scan PlantVillage color image folder, extract metadata, compute pHash,
    and cluster exact and near-duplicates using Union-Find.
    """
    color_path = Path(color_dir)
    records = []

    logger.info("Scanning color image directory: %s", color_path)
    class_folders = sorted([d for d in color_path.iterdir() if d.is_dir() and "___" in d.name])
    if not class_folders:
        class_folders = sorted([d for d in color_path.iterdir() if d.is_dir()])

    for folder in class_folders:
        crop_name, disease_name = parse_plantvillage_label(folder.name)
        image_files = [p for p in folder.iterdir() if p.suffix in valid_extensions and p.is_file()]

        for img_p in image_files:
            records.append({
                "image_path": str(img_p.resolve()),
                "relative_path": str(img_p.relative_to(color_path)),
                "class_label": folder.name,
                "crop_name": crop_name,
                "disease_name": disease_name,
            })

    df = pd.DataFrame(records)
    logger.info("Found %d total valid images across %d classes.", len(df), df["class_label"].nunique())

    if quick:
        logger.info("Quick smoke test: Subsampling %d images across classes...", quick_sample_size)
        df = df.groupby("class_label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(2, quick_sample_size // max(1, len(class_folders)))), random_state=seed)
        ).reset_index(drop=True)
        logger.info("Subsampled to %d images for smoke test.", len(df))

    # Compute pHash for every image
    logger.info("Computing perceptual hashes for %d images...", len(df))
    phashes = []
    formats = []
    dimensions = []

    for path_str in df["image_path"]:
        try:
            with Image.open(path_str) as img:
                formats.append(img.format)
                dimensions.append(f"{img.width}x{img.height}")
            ph = compute_perceptual_hash(path_str)
            phashes.append(ph)
        except Exception as e:
            logger.warning("Could not read image %s: %s", path_str, e)
            formats.append("CORRUPTED")
            dimensions.append("0x0")
            phashes.append("0" * 16)

    df["file_format"] = formats
    df["dimensions"] = dimensions
    df["phash"] = phashes

    # Filter out corrupted
    df = df[df["file_format"] != "CORRUPTED"].reset_index(drop=True)

    # Near-duplicate clustering across the ENTIRE dataset (cross-class and intra-class)
    # Uses 6-chunk multi-index hashing for O(N) candidate generation with 100% recall for dist <= 5
    logger.info("Clustering near-duplicates globally across all classes (Hamming threshold <= %d)...", hamming_threshold)
    uf = UnionFind(list(range(len(df))))

    # Pre-parse 64-bit integer hashes
    int_hashes: List[int] = []
    for h in df["phash"]:
        try:
            int_hashes.append(int(h, 16))
        except (ValueError, TypeError):
            int_hashes.append(0)

    # 6 chunks of 10-11 bits (total 64 bits)
    # By Pigeonhole Principle, <= 4 bit flips across 6 chunks guarantees >= 2 exact chunk matches
    chunk_masks = [
        (0, 0x7FF),    # bits 0..10 (11 bits)
        (11, 0x7FF),   # bits 11..21 (11 bits)
        (22, 0x7FF),   # bits 22..32 (11 bits)
        (33, 0x7FF),   # bits 33..43 (11 bits)
        (44, 0x7FF),   # bits 44..54 (11 bits)
        (55, 0x1FF),   # bits 55..63 (9 bits)
    ]

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, h_int in enumerate(int_hashes):
        for c_i, (shift, mask) in enumerate(chunk_masks):
            key = (c_i, (h_int >> shift) & mask)
            buckets.setdefault(key, []).append(idx)

    # Compare candidates within matching buckets
    checked_pairs: Set[Tuple[int, int]] = set()
    for idx, h_int in enumerate(int_hashes):
        candidates: Set[int] = set()
        for c_i, (shift, mask) in enumerate(chunk_masks):
            key = (c_i, (h_int >> shift) & mask)
            candidates.update(buckets.get(key, []))

        for c_idx in candidates:
            if idx >= c_idx:
                continue
            pair = (idx, c_idx)
            if pair not in checked_pairs:
                checked_pairs.add(pair)
                # Fast CPU popcount
                dist = (h_int ^ int_hashes[c_idx]).bit_count()
                if dist <= hamming_threshold:
                    uf.union(idx, c_idx)

    # Assign duplicate group IDs
    group_ids = [f"group_{uf.find(i):06d}" for i in range(len(df))]
    df["duplicate_group_id"] = group_ids

    # Check for cross-class duplicate groups
    group_class_counts = df.groupby("duplicate_group_id")["class_label"].nunique()
    cross_class_groups = group_class_counts[group_class_counts > 1]
    if not cross_class_groups.empty:
        logger.warning(
            "Found %d duplicate groups spanning MULTIPLE class labels (cross-class duplicates). "
            "All images in each duplicate group will be kept strictly in the same split to avoid data leakage.",
            len(cross_class_groups)
        )

    num_unique_groups = df["duplicate_group_id"].nunique()
    logger.info("Duplicate clustering complete: %d images grouped into %d distinct groups (%d duplicate images collapsed).",
                len(df), num_unique_groups, len(df) - num_unique_groups)

    if output_manifest_path:
        out_p = Path(output_manifest_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info("Saved complete manifest to %s", out_p)

    return df


def split_group_aware(
    manifest_df: pd.DataFrame,
    output_dir: Optional[Path | str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform group-aware, stratified train/val/test splitting so that near-duplicate
    groups NEVER leak across splits.
    """
    rng = np.random.default_rng(seed)

    # Group table: one entry per duplicate group with its predominant class
    group_table = manifest_df.groupby("duplicate_group_id").agg({
        "class_label": lambda s: s.mode()[0] if not s.empty else s.iloc[0],
        "image_path": "count"
    }).rename(columns={"image_path": "image_count"}).reset_index()

    train_groups: List[str] = []
    val_groups: List[str] = []
    test_groups: List[str] = []

    # Stratify by class label at the group level
    for label, grp in group_table.groupby("class_label"):
        group_list = grp["duplicate_group_id"].tolist()
        rng.shuffle(group_list)
        n_g = len(group_list)

        if n_g >= 3:
            n_val = max(1, int(round(n_g * val_ratio)))
            n_test = max(1, int(round(n_g * test_ratio)))
            n_train = n_g - n_val - n_test
            # Ensure at least 1 group in train
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
        elif n_g == 2:
            # 1 group in train, 1 group in test
            n_train = 1
            n_val = 0
            n_test = 1
        else:  # n_g == 1
            # Single group goes to train
            n_train = 1
            n_val = 0
            n_test = 0

        train_g = group_list[:n_train]
        val_g = group_list[n_train:n_train + n_val]
        test_g = group_list[n_train + n_val:n_train + n_val + n_test]

        train_groups.extend(train_g)
        val_groups.extend(val_g)
        test_groups.extend(test_g)

    train_df = manifest_df[manifest_df["duplicate_group_id"].isin(set(train_groups))].copy().reset_index(drop=True)
    val_df = manifest_df[manifest_df["duplicate_group_id"].isin(set(val_groups))].copy().reset_index(drop=True)
    test_df = manifest_df[manifest_df["duplicate_group_id"].isin(set(test_groups))].copy().reset_index(drop=True)

    # Verification: Zero duplicate group leakage across splits
    train_set = set(train_df["duplicate_group_id"])
    val_set = set(val_df["duplicate_group_id"])
    test_set = set(test_df["duplicate_group_id"])

    assert len(train_set & val_set) == 0, "FATAL: Duplicate group leakage between train and val!"
    assert len(train_set & test_set) == 0, "FATAL: Duplicate group leakage between train and test!"
    assert len(val_set & test_set) == 0, "FATAL: Duplicate group leakage between val and test!"

    # Audit rare-class preservation: Report any classes missing from a split
    all_classes = set(manifest_df["class_label"].unique())
    missing_in_train = all_classes - set(train_df["class_label"].unique())
    missing_in_val = all_classes - set(val_df["class_label"].unique())
    missing_in_test = all_classes - set(test_df["class_label"].unique())

    if missing_in_train:
        logger.warning("Classes completely missing from TRAIN split (%d): %s", len(missing_in_train), sorted(missing_in_train))
    if missing_in_val:
        logger.warning("Rare classes missing from VAL split (%d): %s", len(missing_in_val), sorted(missing_in_val))
    if missing_in_test:
        logger.warning("Rare classes missing from TEST split (%d): %s", len(missing_in_test), sorted(missing_in_test))
    if not (missing_in_train or missing_in_val or missing_in_test):
        logger.info("Class Preservation Check: All %d classes successfully present in Train, Val, and Test splits.", len(all_classes))

    logger.info("Split Complete (Group-Aware, Zero Leakage):")
    logger.info("  Train: %d images (%.1f%%)", len(train_df), len(train_df) / len(manifest_df) * 100)
    logger.info("  Val:   %d images (%.1f%%)", len(val_df), len(val_df) / len(manifest_df) * 100)
    logger.info("  Test:  %d images (%.1f%%)", len(test_df), len(test_df) / len(manifest_df) * 100)

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_p / "train_manifest.csv", index=False)
        val_df.to_csv(out_p / "val_manifest.csv", index=False)
        test_df.to_csv(out_p / "test_manifest.csv", index=False)
        logger.info("Saved split manifests to %s", out_p)

    return train_df, val_df, test_df


def build_tf_dataset(
    manifest_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer: int = 2048,
    seed: int = 42,
) -> Any:
    """Construct an optimized, high-throughput tf.data.Dataset pipeline.
    Streams image paths, decodes on the fly, resizes, and prefetches.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow is required to build tf.data.Dataset.")

    paths = manifest_df["image_path"].values.astype(str)
    labels = np.array([class_to_idx[lbl] for lbl in manifest_df["class_label"]], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(manifest_df), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)

    def _load_and_preprocess(path_tensor, label_tensor):
        img_bytes = tf.io.read_file(path_tensor)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img.set_shape([image_size[0], image_size[1], 3])
        # Note: Keep in [0, 255] float32; architecture-specific normalization is inside the model!
        img = tf.cast(img, tf.float32)
        return img, label_tensor

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
