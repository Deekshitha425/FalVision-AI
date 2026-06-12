"""
predictor.py — Model loading and inference for FalVision AI.
Compatible with TensorFlow 2.21 + Keras 3.x (standalone)
"""

import time
import numpy as np
import streamlit as st
from utils.preprocessing import preprocess_image

CLASS_NAMES = [
    "Bad Quality_Fruits",
    "Good Quality_Fruits",
    "Mixed Qualit_Fruits",
]

CLASS_META = {
    "Bad Quality_Fruits": {
        "label":   "⛔ Bad Quality",
        "type":    "error",
        "emoji":   "🍂",
        "summary": "This fruit has been identified as poor quality.",
    },
    "Good Quality_Fruits": {
        "label":   "✅ Good Quality",
        "type":    "success",
        "emoji":   "🍎",
        "summary": "This fruit meets quality standards.",
    },
    "Mixed Qualit_Fruits": {
        "label":   "⚠️ Mixed Quality",
        "type":    "warning",
        "emoji":   "🍊",
        "summary": "This fruit shows mixed quality indicators.",
    },
}

RECOMMENDATIONS = {
    "success": [
        "✅ Ready for immediate sale or retail display.",
        "📦 Suitable for standard shelf-life packaging.",
        "🚚 Can be transported through normal cold chain.",
    ],
    "warning": [
        "🔍 Inspect batch carefully before distribution.",
        "⏰ Prioritise sale within 1-2 days.",
        "🔄 Consider sorting — some units may still be sellable.",
    ],
    "error": [
        "🚫 Do not distribute — remove from supply chain.",
        "🔬 Inspect nearby batch for contamination spread.",
        "🗑️ Compost or dispose of safely.",
    ],
}

QUALITY_TIPS = {
    "success": {
        "storage":  "Follow standard cold-chain: 2-8 degrees C for most fruits.",
        "handling": "Single-layer crating recommended to avoid bruising.",
        "note":     "Batch cleared for distribution. Document lot number.",
    },
    "warning": {
        "storage":  "Reduce storage time — move to front of stock rotation.",
        "handling": "Separate mixed-quality units from premium stock.",
        "note":     "Re-inspect within 24 hours before dispatch decision.",
    },
    "error": {
        "storage":  "Do not refrigerate with good stock — risk of spread.",
        "handling": "Use gloves; bag separately before disposal.",
        "note":     "Log rejection in quality management system.",
    },
}


@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "model/falvision_model.keras"):
    """Load model using standalone keras (required for TF 2.16+ / Keras 3.x)."""
    try:
        import keras
        model = keras.models.load_model(model_path)
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


def predict_image(model, pil_image):
    arr     = preprocess_image(pil_image)
    t0      = time.time()
    probs   = model.predict(arr, verbose=0)[0]
    elapsed = round((time.time() - t0) * 1000, 1)

    idx        = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100
    class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else CLASS_NAMES[0]
    meta       = CLASS_META[class_name]

    return {
        "class_name":      class_name,
        "quality_label":   meta["label"],
        "quality_type":    meta["type"],
        "quality_emoji":   meta["emoji"],
        "quality_summary": meta["summary"],
        "confidence":      confidence,
        "all_probs": {
            CLASS_NAMES[i]: float(probs[i]) * 100
            for i in range(min(len(probs), len(CLASS_NAMES)))
        },
        "prediction_time": elapsed,
        "recommendations": RECOMMENDATIONS[meta["type"]],
        "tips":            QUALITY_TIPS[meta["type"]],
    }
