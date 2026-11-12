import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from ml.disease_detection.src.data import (
    build_dataset_manifest,
    compute_perceptual_hash,
    discover_color_image_directory,
    hamming_distance,
    parse_plantvillage_label,
    split_group_aware,
)


def test_parse_plantvillage_label():
    crop, disease = parse_plantvillage_label("Tomato___Late_blight")
    assert crop == "Tomato"
    assert disease == "Late blight"

    crop_h, disease_h = parse_plantvillage_label("Potato___healthy")
    assert crop_h == "Potato"
    assert disease_h == "healthy"


def test_phash_hamming_distance():
    h1 = "ffff0000ffff0000"
    h2 = "ffff0000ffff0001"  # 1 bit flip
    assert hamming_distance(h1, h2) == 1
    assert hamming_distance(h1, h1) == 0


def test_full_synthetic_data_pipeline():
    """Create a temporary mini-PlantVillage directory structure, generate images,
    run manifest creation with pHash clustering, and perform group-aware split.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        color_dir = tmp_p / "color"
        color_dir.mkdir(parents=True)

        # Create two class folders
        class1_dir = color_dir / "Tomato___Early_blight"
        class2_dir = color_dir / "Tomato___healthy"
        class3_dir = color_dir / "Apple___Apple_scab"
        class1_dir.mkdir()
        class2_dir.mkdir()
        class3_dir.mkdir()

        # Generate base synthetic image
        base_img = Image.new("RGB", (64, 64), color=(34, 139, 34))

        # Class 1: 5 images (2 are exact duplicates)
        base_img.save(class1_dir / "img1.jpg")
        base_img.save(class1_dir / "img2.jpg")  # Duplicate of img1
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(class1_dir / "img3.jpg")
        Image.new("RGB", (64, 64), color=(200, 50, 50)).save(class1_dir / "img4.jpg")
        Image.new("RGB", (64, 64), color=(50, 200, 50)).save(class1_dir / "img5.jpg")

        # Class 2: 4 images, one is a CROSS-CLASS duplicate of img1
        base_img.save(class2_dir / "img_cross_dup.jpg")  # CROSS-CLASS duplicate!
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(class2_dir / "img6.jpg")
        Image.new("RGB", (64, 64), color=(40, 50, 60)).save(class2_dir / "img7.jpg")
        Image.new("RGB", (64, 64), color=(70, 80, 90)).save(class2_dir / "img8.jpg")

        # Class 3: 3 images
        Image.new("RGB", (64, 64), color=(11, 22, 33)).save(class3_dir / "img9.jpg")
        Image.new("RGB", (64, 64), color=(44, 55, 66)).save(class3_dir / "img10.jpg")
        Image.new("RGB", (64, 64), color=(77, 88, 99)).save(class3_dir / "img11.jpg")

        # 1. Discover color directory
        found_dir = discover_color_image_directory(tmp_p)
        assert found_dir == color_dir

        # 2. Build manifest with pHash clustering
        manifest_path = tmp_p / "manifest.csv"
        manifest_df = build_dataset_manifest(
            color_dir=found_dir,
            output_manifest_path=manifest_path,
            hamming_threshold=4,
            seed=42,
        )

        assert manifest_path.exists()
        assert len(manifest_df) == 12

        # Verify duplicate grouping
        # img1, img2, and img_cross_dup MUST all share the same duplicate_group_id!
        img1_group = manifest_df[manifest_df["image_path"].str.contains("img1.jpg")]["duplicate_group_id"].iloc[0]
        img2_group = manifest_df[manifest_df["image_path"].str.contains("img2.jpg")]["duplicate_group_id"].iloc[0]
        cross_group = manifest_df[manifest_df["image_path"].str.contains("img_cross_dup.jpg")]["duplicate_group_id"].iloc[0]

        assert img1_group == img2_group, "Intra-class duplicate not grouped together!"
        assert img1_group == cross_group, "Cross-class duplicate not grouped together!"

        # 3. Perform group-aware split
        splits_dir = tmp_p / "splits"
        train_df, val_df, test_df = split_group_aware(
            manifest_df=manifest_df,
            output_dir=splits_dir,
            seed=42,
        )

        # Assert zero leakage
        train_groups = set(train_df["duplicate_group_id"])
        val_groups = set(val_df["duplicate_group_id"])
        test_groups = set(test_df["duplicate_group_id"])

        assert len(train_groups & val_groups) == 0
        assert len(train_groups & test_groups) == 0
        assert len(val_groups & test_groups) == 0

        # All 3 duplicate images must be in the exact same split
        all_dup_splits = [
            "train" if g in train_groups else ("val" if g in val_groups else "test")
            for g in [img1_group, img2_group, cross_group]
        ]
        assert len(set(all_dup_splits)) == 1, "Duplicates were split across different sets!"
