"""
preprocessing.py — Image preprocessing pipeline for FalVision AI.
"""

import numpy as np
from PIL import Image


IMG_HEIGHT = 224
IMG_WIDTH  = 224


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Resize, normalise, and batch a PIL image for MobileNetV2 inference.

    Args:
        pil_image: PIL.Image object (any mode).
    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32, values in [0, 1].
    """
    img = pil_image.convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
