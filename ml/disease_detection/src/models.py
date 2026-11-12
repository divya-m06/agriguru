import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ml.disease_detection.models")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
    HAS_TF = True
except ImportError:
    HAS_TF = False


def build_augmentation_layer(config: Optional[Dict[str, Any]] = None) -> Any:
    """Build Keras data augmentation sequential block applied strictly during training."""
    if not HAS_TF:
        raise ImportError("TensorFlow required for data augmentation layers.")

    aug_cfg = config or {}
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(aug_cfg.get("random_rotation", 0.15)),
        layers.RandomZoom(aug_cfg.get("random_zoom", 0.15)),
        layers.RandomBrightness(aug_cfg.get("random_brightness", 0.10)),
        layers.RandomContrast(aug_cfg.get("random_contrast", 0.10)),
    ], name="data_augmentation")


@tf.keras.utils.register_keras_serializable(package="agri") if HAS_TF else (lambda cls: cls)
class ResNet50PreprocessLayer(layers.Layer if HAS_TF else object):
    """Architecture-specific preprocessing for ResNet50.
    Converts raw RGB [0, 255] images into zero-centered BGR format with ImageNet mean subtraction:
    [103.939, 116.779, 123.68].
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Reverse RGB to BGR
        bgr = tf.reverse(inputs, axis=[-1])
        mean = tf.constant([103.939, 116.779, 123.68], dtype=inputs.dtype)
        return bgr - mean

    def get_config(self):
        return super().get_config()


def get_architecture_preprocessor(arch_name: str) -> Any:
    """Return the Keras layer for architecture-specific input preprocessing.
    Input to the layer is always raw RGB float32 [0, 255].
    """
    if not HAS_TF:
        raise ImportError("TensorFlow required for architecture preprocessor.")

    arch_lower = arch_name.lower()
    if "mobilenet" in arch_lower:
        # MobileNetV2 expects [-1, 1]
        return layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")
    elif "efficientnet" in arch_lower:
        # EfficientNetB0 has internal scaling built-in; accepts [0, 255] directly
        return layers.Rescaling(scale=1.0, name="efficientnet_passthrough")
    elif "resnet" in arch_lower:
        # ResNet50 expects BGR with ImageNet mean subtraction
        return ResNet50PreprocessLayer(name="resnet50_preprocess")
    else:
        return layers.Rescaling(scale=1.0 / 255.0, name="standard_rescale")


def build_disease_model(
    architecture: str,
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    augmentation_config: Optional[Dict[str, Any]] = None,
    weights: str = "imagenet",
) -> Any:
    """Build transfer learning plant disease classification model.

    Pipeline:
        Raw RGB [0, 255] -> Augmentation (training only) -> Architecture Preprocessing -> Backbone -> GAP -> BN -> Dropout -> Dense(Softmax, float32)
    """
    if not HAS_TF:
        raise ImportError("TensorFlow required to build disease models.")

    inputs = layers.Input(shape=input_shape, name="image_input", dtype="float32")

    # 1. Augmentation block (active only in training mode)
    x = build_augmentation_layer(augmentation_config)(inputs)

    # 2. Architecture Preprocessing (embedded inside model so FastAPI receives raw RGB)
    x = get_architecture_preprocessor(architecture)(x)

    # 3. Pretrained Backbone
    arch_lower = architecture.lower()
    if "mobilenet" in arch_lower:
        backbone = MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
    elif "efficientnet" in arch_lower:
        backbone = EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights)
    elif "resnet" in arch_lower:
        backbone = ResNet50(input_shape=input_shape, include_top=False, weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: MobileNetV2, EfficientNetB0, ResNet50.")

    backbone.trainable = False  # Start Phase 1 frozen
    x = backbone(x)

    # 4. Classification Head
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # Output layer: float32 for numerical stability under mixed precision
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"plant_disease_{arch_lower}")
    return model, backbone


def set_backbone_trainable(
    backbone: Any,
    unfreeze_percent: float = 0.35,
    keep_batchnorm_frozen: bool = True,
) -> int:
    """Unfreeze the top percentage of backbone layers for Phase 2 fine-tuning.
    Keeps BatchNormalization layers frozen to prevent batch statistics degradation.
    """
    backbone.trainable = True
    total_layers = len(backbone.layers)
    unfreeze_count = int(total_layers * unfreeze_percent)
    unfreeze_from_idx = total_layers - unfreeze_count

    unfrozen_counter = 0
    for idx, layer in enumerate(backbone.layers):
        if idx < unfreeze_from_idx:
            layer.trainable = False
        else:
            if keep_batchnorm_frozen and isinstance(layer, (layers.BatchNormalization,)):
                layer.trainable = False
            else:
                layer.trainable = True
                unfrozen_counter += 1

    logger.info("Unfroze top %.1f%% of backbone layers (%d of %d layers unfrozen; BatchNorm frozen=%s).",
                unfreeze_percent * 100, unfrozen_counter, total_layers, keep_batchnorm_frozen)
    return unfrozen_counter
