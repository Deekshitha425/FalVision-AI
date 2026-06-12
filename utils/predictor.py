"""
utils/predictor.py
Model loading, inference, and GradCAM for FalVision AI.

Model architecture (verified):
  MobileNetV2 backbone → out_relu (7×7×1280) → GAP → BN → Dropout → Dense(3, softmax)

Class index order (alphabetical, as assigned by ImageDataGenerator):
  0 → bad_quality   → Bad Quality
  1 → good_quality  → Good Quality
  2 → mixed_quality → Mixed Quality
"""

import numpy as np
import streamlit as st
from PIL import Image
import datetime
import io

# ── Class registry ────────────────────────────────────────────────────────────
CLASS_NAMES    = ["bad_quality", "good_quality", "mixed_quality"]
QUALITY_LABELS = {
    "bad_quality":   "Bad Quality",
    "good_quality":  "Good Quality",
    "mixed_quality": "Mixed Quality",
}

GRADCAM_LAYER = "out_relu"   # last spatial activation: (7, 7, 1280)
IMG_SIZE      = (224, 224)

# Fruit list for the manual selector in the UI
FRUIT_LIST = [
    "Apple", "Banana", "Cherry", "Fig", "Grape", "Guava", "Kiwi",
    "Lemon", "Mango", "Melon", "Orange", "Papaya", "Peach", "Pear",
    "Pineapple", "Plum", "Pomegranate", "Strawberry", "Tomato", "Watermelon",
]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load and cache the Keras model. Runs once per session."""
    import tensorflow as tf
    import os
    try:
        # Build absolute path relative to this file's location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.join(base_dir, model_path)
        model = tf.keras.models.load_model(abs_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


def preprocess_image(image_file) -> np.ndarray:
    """
    Open an uploaded file-like object, resize to 224×224,
    normalise to [0, 1], return shape (1, 224, 224, 3).
    """
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(model, uploaded_file, fruit_name: str) -> dict:
    """
    Run full prediction pipeline.

    Args:
        model         : loaded Keras model
        uploaded_file : Streamlit UploadedFile
        fruit_name    : user-selected fruit name string

    Returns dict with keys:
        fruit_name, quality_label, confidence, probs, timestamp
    """
    import tensorflow as tf

    uploaded_file.seek(0)
    img_array = preprocess_image(uploaded_file)

    preds     = model.predict(img_array, verbose=0)[0]   # (3,)
    pred_idx  = int(np.argmax(preds))
    raw_class = CLASS_NAMES[pred_idx]
    quality_label = QUALITY_LABELS[raw_class]
    confidence    = float(preds[pred_idx]) * 100

    probs = {
        QUALITY_LABELS[CLASS_NAMES[i]]: float(preds[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "fruit_name":    fruit_name,
        "quality_label": quality_label,
        "confidence":    confidence,
        "probs":         probs,
        "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_gradcam(model, uploaded_file, pred_class_idx: int) -> Image.Image:
    """
    Generate a GradCAM heatmap overlay for the predicted class.

    Args:
        model          : loaded Keras model
        uploaded_file  : Streamlit UploadedFile
        pred_class_idx : index of predicted class (0, 1, or 2)

    Returns:
        PIL Image — original image blended with green heatmap overlay
    """
    import tensorflow as tf

    uploaded_file.seek(0)
    orig_img = Image.open(uploaded_file).convert("RGB")
    orig_arr = np.array(orig_img.resize(IMG_SIZE), dtype=np.float32) / 255.0

    img_tensor = np.expand_dims(orig_arr, axis=0)  # (1, 224, 224, 3)

    # Build GradCAM sub-model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(GRADCAM_LAYER).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, pred_class_idx]

    # Gradients of loss w.r.t. conv feature maps
    grads       = tape.gradient(loss, conv_outputs)          # (1, 7, 7, 1280)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))     # (1280,)

    # Weight channels by importance
    conv_outputs = conv_outputs[0]                            # (7, 7, 1280)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]   # (7, 7, 1)
    heatmap = tf.squeeze(heatmap)                            # (7, 7)

    # Normalise to [0, 1]
    heatmap = tf.nn.relu(heatmap)
    heatmap_np = heatmap.numpy()
    if heatmap_np.max() > 0:
        heatmap_np = heatmap_np / heatmap_np.max()

    # Resize heatmap to 224×224
    heatmap_img = Image.fromarray(np.uint8(heatmap_np * 255)).resize(
        IMG_SIZE, Image.BILINEAR
    )
    heatmap_arr = np.array(heatmap_img, dtype=np.float32) / 255.0  # (224, 224)

    # Apply green-tinted colormap (green = high activation)
    colormap = np.zeros((224, 224, 3), dtype=np.float32)
    colormap[:, :, 0] = heatmap_arr * 0.9   # Red channel (low)
    colormap[:, :, 1] = heatmap_arr          # Green channel (high = bright green)
    colormap[:, :, 2] = heatmap_arr * 0.1   # Blue channel (low)

    # Blend with original image
    blended = orig_arr * 0.55 + colormap * 0.45
    blended = np.clip(blended, 0, 1)

    return Image.fromarray(np.uint8(blended * 255))
