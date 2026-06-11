"""
utils/predictor.py
Model loading and inference utilities for FalVision AI.

Model: MobileNetV2 transfer learning
Input: (1, 224, 224, 3) — float32 in [0, 1]
Output: softmax over 3 classes (alphabetical folder order):
    index 0 → bad_quality
    index 1 → good_quality
    index 2 → mixed_quality
"""

import numpy as np
import streamlit as st
from PIL import Image
import datetime

# ── Class registry (must match ImageDataGenerator alphabetical sort) ─────────
CLASS_NAMES = ["bad_quality", "good_quality", "mixed_quality"]

QUALITY_LABELS = {
    "bad_quality":   "Bad Quality",
    "good_quality":  "Good Quality",
    "mixed_quality": "Mixed Quality",
}

# Ordered display list for probability bars (same index order as model output)
DISPLAY_ORDER = [
    ("Good Quality",  1),   # index 1
    ("Mixed Quality", 2),   # index 2
    ("Bad Quality",   0),   # index 0
]

# Fruit keyword → display name (inferred from filename)
FRUIT_KEYWORDS = {
    "apple":       "Apple",
    "mango":       "Mango",
    "banana":      "Banana",
    "orange":      "Orange",
    "grape":       "Grape",
    "strawberry":  "Strawberry",
    "tomato":      "Tomato",
    "lemon":       "Lemon",
    "peach":       "Peach",
    "pear":        "Pear",
    "plum":        "Plum",
    "cherry":      "Cherry",
    "watermelon":  "Watermelon",
    "kiwi":        "Kiwi",
    "papaya":      "Papaya",
    "guava":       "Guava",
    "pomegranate": "Pomegranate",
    "pineapple":   "Pineapple",
    "fig":         "Fig",
    "melon":       "Melon",
}

IMG_SIZE = (224, 224)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load and cache the Keras model. Called once per session."""
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from '{model_path}': {e}")
        st.stop()


def preprocess_image(image_file) -> np.ndarray:
    """
    Open an uploaded file-like object, resize to 224×224,
    normalise to [0, 1], and return a batch of shape (1, 224, 224, 3).
    """
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def infer_fruit_name(filename: str) -> str:
    """Guess fruit display name from image filename."""
    name_lower = filename.lower()
    for keyword, display_name in FRUIT_KEYWORDS.items():
        if keyword in name_lower:
            return display_name
    return "Unknown Fruit"


def predict_image(model, uploaded_file) -> dict:
    """
    Full prediction pipeline.

    Args:
        model:         Loaded Keras model.
        uploaded_file: Streamlit UploadedFile object.

    Returns dict:
        fruit_name    – display name inferred from filename
        quality_label – "Good Quality" | "Mixed Quality" | "Bad Quality"
        confidence    – float 0–100 (top-class probability × 100)
        probs         – dict {label: probability} for all 3 classes
        timestamp     – formatted string
    """
    # Seek to start in case file was already read (e.g. for display)
    uploaded_file.seek(0)
    img_array = preprocess_image(uploaded_file)

    # Raw softmax output — shape (3,)
    preds = model.predict(img_array, verbose=0)[0]

    pred_idx      = int(np.argmax(preds))
    raw_class     = CLASS_NAMES[pred_idx]          # e.g. "good_quality"
    quality_label = QUALITY_LABELS[raw_class]      # e.g. "Good Quality"
    confidence    = float(preds[pred_idx]) * 100   # 0–100

    # Build probability dict keyed by friendly label
    probs = {
        QUALITY_LABELS[CLASS_NAMES[i]]: float(preds[i])
        for i in range(len(CLASS_NAMES))
    }

    fruit_name = infer_fruit_name(uploaded_file.name)

    return {
        "fruit_name":    fruit_name,
        "quality_label": quality_label,
        "confidence":    confidence,
        "probs":         probs,   # {"Good Quality": 0.92, "Bad Quality": 0.05, "Mixed Quality": 0.03}
        "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
