import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import yaml

# Allow direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.common.logger import setup_logger
from ml.common.seed import seed_everything
from ml.disease_detection.src.data import (
    build_dataset_manifest,
    build_tf_dataset,
    discover_color_image_directory,
    parse_plantvillage_label,
    split_group_aware,
)
from ml.disease_detection.src.evaluate import (
    TemperatureScaler,
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    extract_top_confused_pairs,
    plot_confusion_matrix_heatmap,
    plot_reliability_diagram,
)
from ml.disease_detection.src.export import export_disease_detection_artifacts
from ml.disease_detection.src.models import (
    build_disease_model,
    set_backbone_trainable,
)

logger = logging.getLogger("ml.disease_detection.train")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plant Disease Detection Training Pipeline (PlantVillage)")
    parser.add_argument("--data-dir", type=str, default="data/disease_detection", help="Dataset directory containing PlantVillage")
    parser.add_argument("--output-dir", type=str, default="reports/disease_detection", help="Reports, logs, and plots directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/disease_detection", help="Artifacts export directory")
    parser.add_argument("--config", type=str, default="ml/disease_detection/config.yaml", help="Configuration YAML file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test mode (subsample & 1 epoch)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted experiment from latest checkpoint")
    parser.add_argument("--architecture", type=str, default="", help="Specific architecture to run ('MobileNetV2', 'EfficientNetB0', 'ResNet50')")
    parser.add_argument("--backup-dir", type=str, default="", help="Optional Google Drive backup directory path for Colab checkpoints")
    return parser.parse_args()


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def setup_gpu_environment(mixed_precision_cfg: str = "auto") -> Dict[str, Any]:
    """Detect available GPU, enable memory growth, and configure mixed precision."""
    gpu_info: Dict[str, Any] = {
        "gpu_available": False,
        "device_name": "CPU",
        "num_gpus": 0,
        "mixed_precision_enabled": False,
    }

    if not HAS_TF:
        logger.warning("TensorFlow not installed. Running in CPU inspection mode.")
        return gpu_info

    gpus = tf.config.list_physical_devices("GPU")
    gpu_info["num_gpus"] = len(gpus)

    if gpus:
        gpu_info["gpu_available"] = True
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_info["device_name"] = details.get("device_name", gpus[0].name)
            logger.info("Detected GPU: %s with dynamic memory growth enabled.", gpu_info["device_name"])
        except Exception as e:
            logger.warning("Could not set memory growth: %s", e)

        # Mixed precision setup
        compute_capability = 0.0
        if "compute_capability" in details:
            cc = details["compute_capability"]
            compute_capability = float(f"{cc[0]}.{cc[1]}")

        should_enable_mp = False
        if mixed_precision_cfg is True or mixed_precision_cfg == "true":
            should_enable_mp = True
        elif mixed_precision_cfg == "auto" and compute_capability >= 7.0:
            should_enable_mp = True

        if should_enable_mp:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                gpu_info["mixed_precision_enabled"] = True
                logger.info("Enabled mixed precision policy: mixed_float16 (Compute capability: %.1f)", compute_capability)
            except Exception as e:
                logger.warning("Failed to enable mixed precision: %s", e)
    else:
        logger.warning("=" * 80)
        logger.warning("WARNING: No GPU detected! Running on CPU. Training 54k images on CPU will be extremely slow.")
        logger.warning("For full training, run on Google Colab with GPU runtime enabled.")
        logger.warning("=" * 80)

    return gpu_info


def calculate_class_weights(train_df: pd.DataFrame, class_to_idx: Dict[str, int]) -> Dict[int, float]:
    """Calculate balanced class weights from training set only."""
    total_samples = len(train_df)
    n_classes = len(class_to_idx)
    counts = train_df["class_label"].value_counts().to_dict()

    weights = {}
    for lbl, idx in class_to_idx.items():
        c = counts.get(lbl, 1)
        weights[idx] = float(total_samples / (n_classes * c))
    return weights


class DriveBackupCallback(tf.keras.callbacks.Callback if HAS_TF else object):
    """Syncs checkpoints, weights, and logs to Google Drive backup directory after each epoch."""
    def __init__(self, local_dir: Path, backup_dir: Optional[str] = None, arch_name: str = ""):
        super().__init__()
        self.local_dir = local_dir
        self.backup_dir = Path(backup_dir) / arch_name.lower() if backup_dir else None

    def on_epoch_end(self, epoch, logs=None):
        if not self.backup_dir:
            return
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for f in self.local_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, self.backup_dir / f.name)
        except Exception as e:
            logger.warning("Failed to backup checkpoint to Drive: %s", e)


def train_single_architecture(
    arch_name: str,
    train_ds: Any,
    val_ds: Any,
    num_classes: int,
    class_weights: Dict[int, float],
    training_cfg: Dict[str, Any],
    aug_cfg: Dict[str, Any],
    checkpoints_dir: Path,
    quick: bool = False,
    resume: bool = False,
    backup_dir: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    """Execute two-phase transfer learning with automatic checkpointing and resume support."""
    if not HAS_TF:
        raise ImportError("TensorFlow required to train architectures.")

    logger.info("=== Starting Experiment: %s ===", arch_name)
    t0 = time.perf_counter()

    p1_cfg = training_cfg.get("phase1", {})
    p2_cfg = training_cfg.get("phase2", {})
    cb_cfg = training_cfg.get("callbacks", {})

    p1_epochs = 1 if quick else p1_cfg.get("epochs", 8)
    p2_epochs = 1 if quick else p2_cfg.get("epochs", 20)
    total_target_epochs = p1_epochs + p2_epochs

    arch_ckpt_dir = checkpoints_dir / arch_name.lower()
    arch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = arch_ckpt_dir / "best_weights.weights.h5"
    latest_weights_path = arch_ckpt_dir / "latest_weights.weights.h5"
    csv_log_path = arch_ckpt_dir / "history.csv"

    # Build model
    model, backbone = build_disease_model(
        architecture=arch_name,
        num_classes=num_classes,
        dropout_rate=training_cfg.get("head", {}).get("dropout", 0.3),
        augmentation_config=aug_cfg,
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Check for existing checkpoint / resume state
    completed_epochs = 0
    if resume and csv_log_path.exists():
        try:
            prev_df = pd.read_csv(csv_log_path)
            completed_epochs = len(prev_df)
            logger.info("[%s] Resuming: Found %d completed epochs in %s",
                        arch_name, completed_epochs, csv_log_path)
        except Exception as e:
            logger.warning("[%s] Could not read existing history for resume: %s", arch_name, e)

    cb_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(latest_weights_path),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=resume and completed_epochs > 0),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cb_cfg.get("early_stopping_patience", 4),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cb_cfg.get("reduce_lr_factor", 0.2),
            patience=cb_cfg.get("reduce_lr_patience", 2),
            min_lr=cb_cfg.get("min_lr", 1e-6),
            verbose=1,
        ),
    ]

    if backup_dir:
        cb_list.append(DriveBackupCallback(arch_ckpt_dir, backup_dir, arch_name))

    h1 = None
    # Phase 1: Train classification head only
    if completed_epochs < p1_epochs:
        if resume and completed_epochs > 0 and latest_weights_path.exists():
            logger.info("[%s] Restoring latest weights from %s for Phase 1 resume...", arch_name, latest_weights_path)
            model.load_weights(latest_weights_path)

        logger.info("[%s] Phase 1: Training classification head (from epoch %d to %d)...",
                    arch_name, completed_epochs, p1_epochs)
        optimizer_p1 = tf.keras.optimizers.Adam(learning_rate=p1_cfg.get("learning_rate", 1e-3))
        model.compile(optimizer=optimizer_p1, loss=loss_fn, metrics=["accuracy"])

        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=p1_epochs,
            initial_epoch=completed_epochs,
            class_weight=class_weights,
            callbacks=cb_list,
            verbose=1,
        )
    else:
        logger.info("[%s] Phase 1 already complete (%d epochs >= %d).", arch_name, completed_epochs, p1_epochs)

    # Phase 2: Fine-tuning top backbone layers
    current_p2_epoch = max(p1_epochs, completed_epochs)
    h2 = None

    if current_p2_epoch < total_target_epochs:
        logger.info("[%s] Phase 2: Fine-tuning top backbone layers (from epoch %d to %d)...",
                    arch_name, current_p2_epoch, total_target_epochs)
        set_backbone_trainable(
            backbone,
            unfreeze_percent=p2_cfg.get("unfreeze_top_percent", 0.35),
            keep_batchnorm_frozen=True,
        )

        optimizer_p2 = tf.keras.optimizers.Adam(learning_rate=p2_cfg.get("learning_rate", 3e-5))
        model.compile(optimizer=optimizer_p2, loss=loss_fn, metrics=["accuracy"])

        if resume and current_p2_epoch > p1_epochs and latest_weights_path.exists():
            logger.info("[%s] Restoring latest weights from %s for Phase 2 resume...", arch_name, latest_weights_path)
            model.load_weights(latest_weights_path)
        elif best_weights_path.exists():
            logger.info("[%s] Loading best Phase 1 weights prior to Phase 2 fine-tuning...", arch_name)
            model.load_weights(best_weights_path)

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_target_epochs,
            initial_epoch=current_p2_epoch,
            class_weight=class_weights,
            callbacks=cb_list,
            verbose=1,
        )
    else:
        logger.info("[%s] Phase 2 already complete (%d epochs >= %d).", arch_name, completed_epochs, total_target_epochs)

    # Always ensure best validation weights are restored for evaluation
    if best_weights_path.exists():
        model.load_weights(best_weights_path)
        logger.info("[%s] Restored best validation weights from %s.", arch_name, best_weights_path)

    try:
        model.save(arch_ckpt_dir / "final_model.keras")
    except Exception as e:
        logger.warning("[%s] Could not save full .keras model: %s", arch_name, e)

    total_time = time.perf_counter() - t0
    logger.info("[%s] Completed training in %.2fs (%.1f mins).", arch_name, total_time, total_time / 60)

    # Collect training history
    history = {
        "phase1_val_accuracy": [float(v) for v in (h1.history.get("val_accuracy", []) if h1 else [])],
        "phase2_val_accuracy": [float(v) for v in (h2.history.get("val_accuracy", []) if h2 else [])],
        "total_time_seconds": round(total_time, 2),
    }

    return model, history, total_time


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)

    if args.quick:
        if args.output_dir == "reports/disease_detection":
            output_dir = Path("reports/disease_detection_quick")
        if args.artifacts_dir == "artifacts/disease_detection":
            artifacts_dir = Path("artifacts/disease_detection_quick")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    setup_logger("ml.disease_detection", log_file=output_dir / "train.log")
    logger.info("Starting Plant Disease Detection Pipeline (Quick: %s, Seed: %d)", args.quick, args.seed)
    logger.info("Output dir: %s | Artifacts dir: %s", output_dir, artifacts_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. GPU Detection & Configuration
    gpu_info = setup_gpu_environment(config["training"].get("mixed_precision", "auto"))

    # 2. Locate Color Dataset
    color_dir = discover_color_image_directory(data_dir)

    # 3. Create Dataset Manifest & pHash Duplicate Clustering
    manifest_path = output_dir / "dataset_manifest.csv"
    manifest_df = build_dataset_manifest(
        color_dir=color_dir,
        output_manifest_path=manifest_path,
        hamming_threshold=config["data"]["phash"]["hamming_threshold"],
        quick=args.quick,
        quick_sample_size=300 if args.quick else 54000,
        seed=args.seed,
    )

    # 4. Group-Aware Stratified Split (70/15/15) - ZERO Duplicate Leakage
    splits_dir = data_dir / "splits"
    train_df, val_df, test_df = split_group_aware(
        manifest_df=manifest_df,
        output_dir=splits_dir,
        seed=args.seed,
    )

    unique_classes = sorted(manifest_df["class_label"].unique())
    num_classes = len(unique_classes)
    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    class_names_meta = []
    for idx, c_name in enumerate(unique_classes):
        crop, disease = parse_plantvillage_label(c_name)
        class_names_meta.append({
            "class_index": idx,
            "class_label": c_name,
            "crop_name": crop,
            "disease_name": disease,
        })

    class_weights = calculate_class_weights(train_df, class_to_idx)

    # 5. Build tf.data Streaming Pipelines
    image_size = tuple(config["data"]["image_size"])
    batch_size = config["training"]["batch_size"] if not args.quick else 16

    train_ds = build_tf_dataset(train_df, class_to_idx, image_size=image_size, batch_size=batch_size, is_training=True, seed=args.seed)
    val_ds = build_tf_dataset(val_df, class_to_idx, image_size=image_size, batch_size=batch_size, is_training=False)
    test_ds = build_tf_dataset(test_df, class_to_idx, image_size=image_size, batch_size=batch_size, is_training=False)

    # 6. Architecture Comparison & Training
    candidate_archs = []
    if args.architecture:
        candidate_archs = [args.architecture]
    else:
        candidate_archs = ["MobileNetV2", "EfficientNetB0"]
        if gpu_info["gpu_available"] and not args.quick:
            # Only run ResNet50 if GPU resources allow
            candidate_archs.append("ResNet50")

    comparison_results = []
    trained_models = {}
    current_batch_size = batch_size

    for arch in candidate_archs:
        success = False
        while not success and current_batch_size >= config["training"].get("min_batch_size", 4):
            try:
                model, hist, elapsed = train_single_architecture(
                    arch_name=arch,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    num_classes=num_classes,
                    class_weights=class_weights,
                    training_cfg=config["training"],
                    aug_cfg=config.get("augmentation", {}),
                    checkpoints_dir=checkpoints_dir,
                    quick=args.quick,
                    resume=args.resume,
                    backup_dir=args.backup_dir,
                )
                success = True
            except (tf.errors.ResourceExhaustedError if HAS_TF else RuntimeError) as oom_err:
                logger.warning("[%s] GPU OOM (ResourceExhaustedError) at batch_size=%d: %s",
                               arch, current_batch_size, oom_err)
                current_batch_size = max(4, current_batch_size // 2)
                logger.warning("[%s] Falling back to batch_size=%d and rebuilding tf.data streaming pipelines...",
                               arch, current_batch_size)
                if HAS_TF:
                    tf.keras.backend.clear_session()
                train_ds = build_tf_dataset(train_df, class_to_idx, image_size=image_size, batch_size=current_batch_size, is_training=True, seed=args.seed)
                val_ds = build_tf_dataset(val_df, class_to_idx, image_size=image_size, batch_size=current_batch_size, is_training=False)
                test_ds = build_tf_dataset(test_df, class_to_idx, image_size=image_size, batch_size=current_batch_size, is_training=False)

        if not success:
            raise RuntimeError(f"Failed to train {arch} even at minimum batch size {current_batch_size}.")

        trained_models[arch] = model

        # Evaluate on validation set
        val_probs = model.predict(val_ds, verbose=0)
        y_val_true = np.array([class_to_idx[l] for l in val_df["class_label"]])
        val_metrics = compute_comprehensive_metrics(y_val_true, val_probs, unique_classes)

        comp_record = {
            "architecture": arch,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "training_time_seconds": elapsed,
        }
        comparison_results.append(comp_record)
        logger.info("[%s Validation] Accuracy: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f",
                    arch, val_metrics["accuracy"], val_metrics["macro_f1"], val_metrics["weighted_f1"])

    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(output_dir / "architecture_comparison.csv", index=False)

    # 7. Model Selection Rule:
    # Prefer EfficientNetB0 by default; choose MobileNetV2 if macro-F1 is within 0.5% (faster/lighter)
    eff_f1 = comp_df[comp_df["architecture"] == "EfficientNetB0"]["val_macro_f1"].max() if "EfficientNetB0" in comp_df["architecture"].values else 0.0
    mob_f1 = comp_df[comp_df["architecture"] == "MobileNetV2"]["val_macro_f1"].max() if "MobileNetV2" in comp_df["architecture"].values else 0.0

    if mob_f1 > 0 and (eff_f1 - mob_f1) <= 0.005:
        chosen_arch = "MobileNetV2"
        selection_reason = "MobileNetV2 selected: Macro-F1 is within 0.5% of EfficientNetB0 with significantly lower inference latency and parameter size."
    elif eff_f1 > 0:
        chosen_arch = "EfficientNetB0"
        selection_reason = "EfficientNetB0 selected: Superior parameter-to-accuracy ratio and highest validation generalization score."
    else:
        chosen_arch = comp_df.sort_values(by="val_macro_f1", ascending=False).iloc[0]["architecture"]
        selection_reason = f"Selected {chosen_arch} as highest scoring architecture."

    logger.info("Champion Architecture Selection: %s (%s)", chosen_arch, selection_reason)
    champion_model = trained_models[chosen_arch]

    # 8. Post-Hoc Confidence Calibration (Temperature Scaling)
    val_probs_champ = champion_model.predict(val_ds, verbose=0)
    y_val_true = np.array([class_to_idx[l] for l in val_df["class_label"]])
    logits_val = np.log(np.clip(val_probs_champ, 1e-12, 1.0 - 1e-12))

    scaler = TemperatureScaler().fit(logits_val, y_val_true)
    calibrated_val_probs = scaler.transform(logits_val)

    ece_uncalibrated, b_acc_un, b_conf_un, _ = compute_expected_calibration_error(y_val_true, val_probs_champ)
    ece_calibrated, b_acc_cal, b_conf_cal, _ = compute_expected_calibration_error(y_val_true, calibrated_val_probs)
    logger.info("Calibration Results: ECE before=%.4f -> ECE after=%.4f (T=%.4f)",
                ece_uncalibrated, ece_calibrated, scaler.temperature)

    plot_reliability_diagram(b_acc_cal, b_conf_cal, ece_calibrated, output_dir / "reliability_diagram.png")

    # 9. Final Test Evaluation (Exactly Once on untouched Test Set)
    logger.info("Evaluating Champion (%s) on untouched Test Set (%d images)...", chosen_arch, len(test_df))
    test_probs_raw = champion_model.predict(test_ds, verbose=0)
    test_logits = np.log(np.clip(test_probs_raw, 1e-12, 1.0 - 1e-12))
    calibrated_test_probs = scaler.transform(test_logits)

    y_test_true = np.array([class_to_idx[l] for l in test_df["class_label"]])
    test_metrics = compute_comprehensive_metrics(y_test_true, calibrated_test_probs, unique_classes)

    logger.info("FINAL TEST RESULTS (%s):", chosen_arch)
    logger.info("  Accuracy:     %.4f", test_metrics["accuracy"])
    logger.info("  Macro-F1:     %.4f", test_metrics["macro_f1"])
    logger.info("  Weighted-F1:  %.4f", test_metrics["weighted_f1"])
    logger.info("  Top-3 Acc:    %.4f", test_metrics["top_k_accuracy"])

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # 10. Visualizations & Confused Pairs
    y_test_pred = np.argmax(calibrated_test_probs, axis=1)
    cm = confusion_matrix(y_test_true, y_test_pred, labels=np.arange(num_classes))
    plot_confusion_matrix_heatmap(cm, unique_classes, output_dir / "confusion_matrix.png")

    confused_pairs = extract_top_confused_pairs(cm, unique_classes, top_n=20)
    pd.DataFrame(confused_pairs).to_csv(output_dir / "most_confused_pairs.csv", index=False)

    # 11. Export Production Artifacts
    metadata = {
        "architecture": chosen_arch,
        "input_dimensions": list(image_size) + [3],
        "num_classes": num_classes,
        "temperature": scaler.temperature,
        "uncertainty_threshold": config["calibration"]["default_uncertainty_threshold"],
        "ece_calibrated": ece_calibrated,
        "ece_uncalibrated": ece_uncalibrated,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "top_3_accuracy": test_metrics["top_k_accuracy"],
        },
        "gpu_info": gpu_info,
        "library_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": tf.__version__ if HAS_TF else "None",
        },
        "git_commit": get_git_commit_hash(),
        "selection_reason": selection_reason,
        "training_date": datetime.now(timezone.utc).isoformat(),
    }

    export_disease_detection_artifacts(
        model=champion_model,
        class_names_meta=class_names_meta,
        metadata=metadata,
        output_artifacts_dir=artifacts_dir,
    )

    if args.backup_dir:
        b_dir = Path(args.backup_dir)
        try:
            b_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copytree(artifacts_dir, b_dir / "artifacts", dirs_exist_ok=True)
            shutil.copytree(output_dir, b_dir / "reports", dirs_exist_ok=True)
            logger.info("Successfully mirrored final artifacts and reports to backup directory: %s", b_dir)
        except Exception as e:
            logger.warning("Could not sync final artifacts to backup directory: %s", e)

    logger.info("Pipeline Complete! All artifacts exported to %s", artifacts_dir)


if __name__ == "__main__":
    main()
