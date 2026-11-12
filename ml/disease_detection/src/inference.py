import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("ml.disease_detection.inference")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from ml.disease_detection.src.gradcam import generate_gradcam_heatmap, overlay_gradcam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


class PlantDiseasePredictor:
    """Production inference engine for plant disease classification.
    Designed for zero-training-code FastAPI integration.
    """

    def __init__(self, artifact_dir: Path | str):
        self.artifact_dir = Path(artifact_dir)

        model_path = self.artifact_dir / "disease_model.keras"
        class_names_path = self.artifact_dir / "class_names.json"
        metadata_path = self.artifact_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        if not class_names_path.exists():
            raise FileNotFoundError(f"Class names mapping not found at: {class_names_path}")

        # 1. Load class definitions
        with open(class_names_path, "r", encoding="utf-8") as f:
            self.classes_meta = json.load(f)
        self.class_names = [item["class_label"] for item in self.classes_meta]

        # 2. Load metadata (temperature, uncertainty threshold, image dimensions)
        self.metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.uncertainty_threshold = float(self.metadata.get("uncertainty_threshold", 0.65))
        self.target_size = tuple(self.metadata.get("input_dimensions", [224, 224])[:2])

        # 3. Load Keras model
        if HAS_TF:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Loaded Keras model from %s (Architecture: %s, Classes: %d)",
                        model_path, self.metadata.get("architecture", "Unknown"), len(self.class_names))
        else:
            self.model = None
            logger.warning("TensorFlow not installed. Model object not initialized.")

    def preprocess_image(self, image_input: Union[bytes, Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """Convert path, bytes, PIL Image, or numpy array into raw RGB float32 tensor (1, 224, 224, 3) in [0, 255]."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_input = np.stack([image_input] * 3, axis=-1)
            elif image_input.shape[2] == 4:
                image_input = image_input[:, :, :3]
            pil_img = Image.fromarray(image_input.astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        resized = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        return arr

    def predict(
        self,
        image_input: Union[bytes, Image.Image, np.ndarray],
        top_k: int = 3,
        return_gradcam: bool = False,
    ) -> Dict[str, Any]:
        """Classify plant disease from input image.

        Returns structured dictionary:
        {
            "predicted_crop": str,
            "predicted_disease": str,
            "predicted_class": str,
            "confidence": float,
            "uncertain": bool,
            "top_predictions": List[Dict[str, Any]],
            "gradcam_base64": Optional[str]
        }
        """
        img_arr = self.preprocess_image(image_input)
        batch = np.expand_dims(img_arr, axis=0)

        if self.model is None:
            raise RuntimeError("Model is not loaded. Ensure TensorFlow is installed.")

        # Forward pass (raw predictions in [0, 1])
        raw_probs = self.model.predict(batch, verbose=0)[0]

        # Apply temperature scaling
        # Invert softmax to logits approx, apply temperature
        clipped_probs = np.clip(raw_probs, 1e-12, 1.0 - 1e-12)
        logits = np.log(clipped_probs)
        scaled_logits = logits / max(1e-3, self.temperature)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        # Ranked indices
        top_indices = np.argsort(calibrated_probs)[::-1][:top_k]
        top_predictions = []
        for idx in top_indices:
            meta = self.classes_meta[idx]
            top_predictions.append({
                "class_index": int(idx),
                "class_label": meta["class_label"],
                "crop_name": meta["crop_name"],
                "disease_name": meta["disease_name"],
                "confidence": float(calibrated_probs[idx]),
            })

        best = top_predictions[0]
        is_uncertain = bool(best["confidence"] < self.uncertainty_threshold)

        result: Dict[str, Any] = {
            "predicted_crop": best["crop_name"],
            "predicted_disease": best["disease_name"],
            "predicted_class": best["class_label"],
            "confidence": best["confidence"],
            "uncertain": is_uncertain,
            "top_predictions": top_predictions,
            "calibrated": bool(self.temperature != 1.0),
            "temperature_used": self.temperature,
            "uncertainty_threshold": self.uncertainty_threshold,
        }

        if return_gradcam and HAS_GRADCAM and self.model is not None:
            try:
                heatmap = generate_gradcam_heatmap(self.model, img_arr, target_class_idx=int(top_indices[0]))
                overlay = overlay_gradcam_on_image(img_arr, heatmap)
                buf = io.BytesIO()
                Image.fromarray(overlay).save(buf, format="JPEG")
                result["gradcam_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                logger.warning("Could not generate Grad-CAM: %s", e)
                result["gradcam_base64"] = None
        else:
            result["gradcam_base64"] = None

        return result


_cached_predictor: Optional[PlantDiseasePredictor] = None


def load_model(artifact_dir: Path | str) -> PlantDiseasePredictor:
    """Module-level loader caching predictor instance for FastAPI."""
    global _cached_predictor
    if _cached_predictor is None or str(_cached_predictor.artifact_dir) != str(artifact_dir):
        _cached_predictor = PlantDiseasePredictor(artifact_dir)
    return _cached_predictor


def predict_image(
    image_bytes: bytes,
    top_k: int = 3,
    return_gradcam: bool = False,
    artifact_dir: Path | str = "artifacts/disease_detection",
) -> Dict[str, Any]:
    """Module-level convenience prediction helper for FastAPI routes."""
    predictor = load_model(artifact_dir)
    return predictor.predict(image_bytes, top_k=top_k, return_gradcam=return_gradcam)
