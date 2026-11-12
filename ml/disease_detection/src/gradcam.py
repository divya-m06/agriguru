import logging
from typing import Any, Optional, Tuple
import matplotlib.cm as cm
import numpy as np

logger = logging.getLogger("ml.disease_detection.gradcam")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def find_target_conv_layer(model: Any) -> Tuple[Any, str]:
    """Inspect model and find the deepest suitable 4D convolutional activation layer,
    supporting both flat models and nested backbone submodels.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow required for Grad-CAM layer resolution.")

    # 1. Check if model has a nested backbone submodel (e.g. mobilenetv2, efficientnetb0, resnet50)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers") and len(layer.layers) > 5:
            # Nested submodel
            submodel = layer
            for sublayer in reversed(submodel.layers):
                out_shape = getattr(sublayer, "output_shape", None)
                if out_shape and len(out_shape) == 4 and "conv" in sublayer.name.lower() or "top_activation" in sublayer.name or "out_relu" in sublayer.name:
                    return submodel, sublayer.name

    # 2. Check flat model layers
    for layer in reversed(model.layers):
        out_shape = getattr(layer, "output_shape", None)
        if out_shape and len(out_shape) == 4:
            return model, layer.name

    raise ValueError(f"Could not automatically locate a 4D convolutional feature map layer in model: {model.name}")


def generate_gradcam_heatmap(
    model: Any,
    img_array: np.ndarray,
    target_class_idx: Optional[int] = None,
    conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generate Grad-CAM activation heatmap for an input image (224, 224, 3) in [0, 255].

    Returns:
        2D float32 numpy array with shape (224, 224) normalized to [0, 1].
    """
    if not HAS_TF:
        raise ImportError("TensorFlow required to compute Grad-CAM heatmaps.")

    if img_array.ndim == 3:
        img_tensor = np.expand_dims(img_array, axis=0)
    else:
        img_tensor = img_array

    # Build gradient model mapping input to target conv output and final predictions
    submodel, found_layer_name = find_target_conv_layer(model)
    layer_name = conv_layer_name or found_layer_name

    if submodel != model and hasattr(submodel, "get_layer"):
        # Nested submodel architecture (e.g. transfer learning backbone)
        target_sublayer = submodel.get_layer(layer_name)
        sub_extractor = tf.keras.models.Model(
            inputs=submodel.inputs,
            outputs=[target_sublayer.output, submodel.output]
        )

        # Trace from outer model inputs through layers prior to submodel
        x = model.inputs[0] if isinstance(model.inputs, list) else model.inputs
        for layer in model.layers:
            if layer == submodel:
                break
            x = layer(x)

        conv_out, sub_out = sub_extractor(x)

        # Trace through downstream head layers
        head_out = sub_out
        passed_submodel = False
        for layer in model.layers:
            if passed_submodel:
                head_out = layer(head_out)
            elif layer == submodel:
                passed_submodel = True

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_out, head_out],
        )
    else:
        # Flat model architecture
        target_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

    with tf.GradientTape() as tape:
        inputs_cast = tf.cast(img_tensor, tf.float32)
        conv_outputs, predictions = grad_model(inputs_cast)
        if target_class_idx is None:
            target_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, target_class_idx]

    # Compute gradients of class score with respect to feature map
    grads = tape.gradient(loss, conv_outputs)
    # Global average pooling of gradients gives feature map importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU on weighted map
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap_np = heatmap.numpy()

    # Resize heatmap to match image spatial resolution (224, 224)
    heatmap_resized = tf.image.resize(
        heatmap_np[..., np.newaxis],
        (img_array.shape[0] if img_array.ndim == 3 else img_array.shape[1],
         img_array.shape[1] if img_array.ndim == 3 else img_array.shape[2]),
    ).numpy().squeeze()

    return np.clip(heatmap_resized, 0.0, 1.0)


def overlay_gradcam_on_image(
    img_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend Grad-CAM heatmap over original RGB leaf image.

    Args:
        img_array: RGB image array shape (H, W, 3), range [0, 255].
        heatmap: 2D heatmap shape (H, W), range [0, 1].
        alpha: Blend factor for overlay.
        colormap: Matplotlib colormap name.

    Returns:
        Blended RGB uint8 image array shape (H, W, 3).
    """
    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)  # (H, W, 4) in [0, 1]
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.float32)

    base_img = img_array.astype(np.float32)
    overlay = (1.0 - alpha) * base_img + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
