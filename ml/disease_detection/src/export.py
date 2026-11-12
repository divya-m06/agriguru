import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger("ml.disease_detection.export")


def export_disease_detection_artifacts(
    model: Any,
    class_names_meta: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_artifacts_dir: Path | str,
    sample_images: Optional[List[Tuple[str, np.ndarray, str]]] = None,
) -> Path:
    """Package and export all production inference artifacts.

    Exports:
    - disease_model.keras (Standalone model with embedded preprocessing)
    - class_names.json (Class metadata with parsed crop and disease names)
    - metadata.json (Full hyperparams, GPU info, calibration, library versions)
    - inference.py and gradcam.py (Self-contained for FastAPI consumption)
    - sample_images/ and sample_predictions.json (For fresh-process testing)
    """
    artifacts_path = Path(output_artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # 1. Save Keras model (.keras format)
    model_save_path = artifacts_path / "disease_model.keras"
    if hasattr(model, "save"):
        model.save(model_save_path)
        logger.info("Saved model artifact to %s (%.2f MB)",
                    model_save_path, model_save_path.stat().st_size / (1024 * 1024))
    else:
        logger.warning("Mock model supplied; touching %s for interface compliance.", model_save_path)
        model_save_path.touch()

    # 2. Save class_names.json
    class_names_path = artifacts_path / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names_meta, f, indent=2)
    logger.info("Saved class metadata to %s (%d classes)", class_names_path, len(class_names_meta))

    # 3. Save metadata.json
    metadata_path = artifacts_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved run metadata to %s", metadata_path)

    # 4. Copy inference.py and gradcam.py to artifacts for standalone FastAPI portability
    src_dir = Path(__file__).resolve().parent
    for fname in ["inference.py", "gradcam.py"]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, artifacts_path / fname)

    # 5. Save sample images and sample predictions for test suites
    sample_dir = artifacts_path / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_predictions = []

    if sample_images:
        for idx, (img_name, img_arr, true_label) in enumerate(sample_images[:12]):
            out_img_path = sample_dir / f"sample_{idx:02d}_{img_name}.jpg"
            img_pil = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
            img_pil.save(out_img_path, format="JPEG", quality=90)
            sample_predictions.append({
                "sample_id": idx,
                "file_name": out_img_path.name,
                "true_label": true_label,
            })
    else:
        # Generate 10 synthetic leaf sample images with slight green variance if no raw images passed
        for idx in range(10):
            out_img_path = sample_dir / f"sample_{idx:02d}_leaf.jpg"
            rng = np.random.default_rng(42 + idx)
            dummy_leaf = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_leaf[:, :, 1] = rng.integers(120, 180, size=(224, 224), dtype=np.uint8)  # Green leaf
            dummy_leaf[:, :, 0] = rng.integers(40, 80, size=(224, 224), dtype=np.uint8)
            dummy_leaf[:, :, 2] = rng.integers(20, 60, size=(224, 224), dtype=np.uint8)
            Image.fromarray(dummy_leaf).save(out_img_path, format="JPEG")
            sample_predictions.append({
                "sample_id": idx,
                "file_name": out_img_path.name,
                "true_label": class_names_meta[idx % len(class_names_meta)]["class_label"],
            })

    sample_preds_path = artifacts_path / "sample_predictions.json"
    with open(sample_preds_path, "w", encoding="utf-8") as f:
        json.dump(sample_predictions, f, indent=2)
    logger.info("Saved sample verification test images to %s", sample_dir)

    return artifacts_path
