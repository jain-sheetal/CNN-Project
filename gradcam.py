"""
Grad-CAM: Gradient-Weighted Class Activation Mapping
Reference: Selvaraju et al., ICCV 2017 (https://arxiv.org/abs/1610.02391)

Industry-grade Explainable AI (XAI) utility designed for:
  - EfficientNetV2, ResNet, MobileNet, and Sequential CNN architectures
  - Binary & Multi-class classification
  - Sub-model and nested functional graph introspection
  - High-fidelity heatmap overlays with OpenCV colormaps

Author: Senior Computer Vision Engineer
"""

import os
import sys
import argparse
import logging
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GradCAM")


def find_last_conv_layer(model: tf.keras.Model) -> Tuple[Optional[tf.keras.layers.Layer], Optional[tf.keras.Model]]:
    """
    Introspects the model architecture backwards to locate the last 4D convolutional layer.
    Handles both flat models and nested functional models (e.g., Transfer Learning backbones).
    
    Returns:
        (target_layer, parent_model): The target layer and its immediate parent model.
    """
    # 1. First, search top-level layers in reverse order
    for layer in reversed(model.layers):
        # If this layer is an inner Model (like EfficientNetV2 backbone)
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if is_candidate_conv_layer(inner_layer):
                    logger.info(f"Located convolutional layer '{inner_layer.name}' inside sub-model '{layer.name}'.")
                    return inner_layer, layer
        elif is_candidate_conv_layer(layer):
            logger.info(f"Located top-level convolutional layer '{layer.name}'.")
            return layer, model

    # 2. Specific search for known EfficientNet names as fallback
    for name in ["top_conv", "top_activation", "conv5_block3_out", "out_relu"]:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    target = layer.get_layer(name)
                    return target, layer
                except ValueError:
                    continue
            if layer.name == name:
                return layer, model

    raise ValueError(f"Could not automatically identify a 4D convolutional layer in model '{model.name}'.")


def is_candidate_conv_layer(layer: tf.keras.layers.Layer) -> bool:
    """Checks if a layer is a valid target convolutional feature map generator."""
    # Check layer type
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D)):
        return True
    
    # Check output shape rank (must be 4D: [batch, height, width, channels])
    try:
        output_shape = getattr(layer, "output_shape", None)
        if output_shape and len(output_shape) == 4:
            # Exclude input or reshape layers if not conv
            if not isinstance(layer, (tf.keras.layers.InputLayer, tf.keras.layers.ZeroPadding2D)):
                return True
    except Exception:
        pass

    # Check name heuristics (e.g. 'top_conv', 'conv', 'features')
    name = layer.name.lower()
    if any(k in name for k in ["conv", "activation"]) and not any(k in name for k in ["bn", "batch", "norm", "dropout"]):
        try:
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return True
        except Exception:
            pass

    return False


def build_gradcam_model(
    model: tf.keras.Model,
    target_layer_name: Optional[str] = None
) -> Tuple[tf.keras.Model, str]:
    """
    Constructs a dual-output computational model mapping model inputs to:
      [target_layer_activations, final_model_predictions]
    """
    if target_layer_name:
        target_layer = None
        for layer in model.layers:
            if layer.name == target_layer_name:
                target_layer = layer
                break
            if isinstance(layer, tf.keras.Model):
                try:
                    target_layer = layer.get_layer(target_layer_name)
                    break
                except ValueError:
                    pass
        if target_layer is None:
            logger.warning(f"Layer '{target_layer_name}' not found. Falling back to auto-detection.")
            target_layer, _ = find_last_conv_layer(model)
    else:
        target_layer, _ = find_last_conv_layer(model)

    resolved_layer_name = target_layer.name
    logger.info(f"Using layer '{resolved_layer_name}' for Grad-CAM computation.")

    # In modern Keras (v3 / TF 2.16+), try building direct multi-output model
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
            name="gradcam_extractor"
        )
        return grad_model, resolved_layer_name
    except Exception as e:
        logger.info(f"Direct graph extraction yielded: {e}. Building functional wrapper...")

    # Fallback for nested models:
    # If target_layer is inside an inner model (e.g. base_model)
    sub_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                _ = layer.get_layer(resolved_layer_name)
                sub_model = layer
                break
            except ValueError:
                continue

    if sub_model is not None:
        sub_grad = tf.keras.Model(
            inputs=sub_model.inputs,
            outputs=[target_layer.output, sub_model.output]
        )
        
        # Build head model from the remaining layers
        head_inputs = tf.keras.Input(shape=sub_model.output_shape[1:])
        x = head_inputs
        found_sub = False
        for l in model.layers:
            if l == sub_model:
                found_sub = True
                continue
            if found_sub:
                x = l(x)
        head_model = tf.keras.Model(inputs=head_inputs, outputs=x)

        # Composite grad model
        comp_inputs = model.inputs
        conv_feats, sub_out = sub_grad(comp_inputs)
        final_preds = head_model(sub_out)
        return tf.keras.Model(inputs=comp_inputs, outputs=[conv_feats, final_preds]), resolved_layer_name

    raise RuntimeError(f"Unable to construct Grad-CAM extraction model for layer '{resolved_layer_name}'.")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    target_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> Tuple[np.ndarray, float, str, str]:
    """
    Computes the 2D Grad-CAM attention heatmap for a single image array.

    Args:
        model: Trained Keras model
        img_array: Preprocessed image tensor [1, H, W, 3] with values scaled appropriately
        target_layer_name: Optional specific conv layer name
        pred_index: Optional target class index (0 for Cat, 1 for Dog). If None, uses argmax/threshold.

    Returns:
        (heatmap, score, pred_class, layer_name)
    """
    grad_model, layer_name = build_gradcam_model(model, target_layer_name)

    # Cast input to float32 tensor
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        # Handle binary classification (Sigmoid output shape [batch, 1])
        if predictions.shape[-1] == 1:
            raw_score = float(predictions[0, 0].numpy())
            if pred_index is None:
                # 1 = Dog, 0 = Cat
                pred_class_idx = 1 if raw_score >= 0.5 else 0
            else:
                pred_class_idx = pred_index
            
            # Formulate targeted loss
            if pred_class_idx == 1:
                loss = predictions[:, 0]
                pred_class = "Dog"
                confidence = raw_score
            else:
                loss = 1.0 - predictions[:, 0]
                pred_class = "Cat"
                confidence = 1.0 - raw_score
        else:
            # Multi-class softmax output
            probs = tf.nn.softmax(predictions, axis=-1)
            if pred_index is None:
                pred_class_idx = int(tf.argmax(predictions[0]).numpy())
            else:
                pred_class_idx = pred_index
            loss = predictions[:, pred_class_idx]
            pred_class = f"Class_{pred_class_idx}"
            confidence = float(probs[0, pred_class_idx].numpy())

    # Calculate gradients of the chosen class score w.r.t. convolutional activations
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Failed to compute gradients. Ensure layer has trainable/active connections.")

    # Global Average Pooling of gradients: Importance weights alpha_k
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of feature map activations
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to retain only positive influence features (Selvaraju et al., 2017)
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize heatmap strictly to [0.0, 1.0]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy(), float(confidence), pred_class, layer_name


def overlay_gradcam(
    original_img: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Superimposes the 2D Grad-CAM heatmap onto the original RGB image.

    Args:
        original_img: Original image (RGB format, shape [H, W, 3], uint8 [0, 255])
        heatmap: 2D float array in range [0, 1]
        alpha: Blending weight for heatmap (0 = only original, 1 = only heatmap)
        colormap: OpenCV colormap enum (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS)

    Returns:
        (superimposed_rgb, colored_heatmap_rgb): Both in uint8 RGB format [H, W, 3]
    """
    # Ensure original_img is uint8 numpy array in RGB
    if isinstance(original_img, Image.Image):
        original_np = np.array(original_img.convert("RGB"), dtype=np.uint8)
    else:
        original_np = np.asarray(original_img, dtype=np.uint8)

    height, width = original_np.shape[:2]

    # Resize heatmap to match original image dimensions with bicubic interpolation
    resized_heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Clip and scale to [0, 255]
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * resized_heatmap)

    # Apply OpenCV Colormap (Outputs BGR)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    # blended = alpha * heatmap + (1 - alpha) * original
    blended = cv2.addWeighted(colored_rgb, alpha, original_np, 1.0 - alpha, 0)

    return blended, colored_rgb


def explain_image(
    model: tf.keras.Model,
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    target_layer_name: Optional[str] = None,
    alpha: float = 0.45,
    colormap_name: str = "JET"
) -> Dict[str, Any]:
    """
    End-to-end Explainable AI pipeline function:
    Takes an input image, runs inference, computes Grad-CAM, and generates visualizations.

    Returns:
        Dictionary containing:
          - 'original_rgb': Original image [H, W, 3]
          - 'superimposed_rgb': Blended overlay [H, W, 3]
          - 'heatmap_rgb': Colored heatmap [H, W, 3]
          - 'raw_heatmap': 2D float numpy array [0, 1]
          - 'predicted_class': 'Dog' or 'Cat'
          - 'confidence': float (0.0 to 1.0)
          - 'layer_used': str
    """
    # Map colormap name to OpenCV enum
    cmap_map = {
        "JET": cv2.COLORMAP_JET,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "TURBO": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    }
    cv_colormap = cmap_map.get(colormap_name.upper(), cv2.COLORMAP_JET)

    # 1. Load image
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    original_rgb = np.array(pil_img)

    # 2. Resize and prepare input tensor for EfficientNetV2
    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized_pil, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)

    # 3. Compute Grad-CAM
    raw_heatmap, confidence, pred_class, layer_used = compute_gradcam_heatmap(
        model=model,
        img_array=img_tensor,
        target_layer_name=target_layer_name
    )

    # 4. Overlay onto original high-resolution image
    superimposed_rgb, heatmap_rgb = overlay_gradcam(
        original_img=original_rgb,
        heatmap=raw_heatmap,
        alpha=alpha,
        colormap=cv_colormap
    )

    return {
        "original_rgb": original_rgb,
        "superimposed_rgb": superimposed_rgb,
        "heatmap_rgb": heatmap_rgb,
        "raw_heatmap": raw_heatmap,
        "predicted_class": pred_class,
        "confidence": confidence,
        "layer_used": layer_used
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Visual Attention Heatmap Generator")
    parser.add_argument("--image", type=str, required=True, help="Path to input test image")
    parser.add_argument("--model", type=str, default="best_model.keras", help="Path to saved model")
    parser.add_argument("--output", type=str, default="gradcam_output.jpg", help="Path to save result")
    parser.add_argument("--colormap", type=str, default="JET", choices=["JET", "VIRIDIS", "INFERNO", "PLASMA"], help="Colormap")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity overlay (0.0 to 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)

    print(f"Loading model '{args.model}'...")
    loaded_model = tf.keras.models.load_model(args.model, compile=False)

    print(f"Computing Grad-CAM for '{args.image}'...")
    result = explain_image(
        model=loaded_model,
        image_input=args.image,
        alpha=args.alpha,
        colormap_name=args.colormap
    )

    print(f"Prediction: {result['predicted_class']} (Confidence: {result['confidence']*100:.2f}%)")
    print(f"Attention Layer: {result['layer_used']}")

    output_bgr = cv2.cvtColor(result["superimposed_rgb"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_bgr)
    print(f"Grad-CAM visual overlay saved to '{args.output}'.")
