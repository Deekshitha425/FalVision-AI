"""
analytics.py — Session history and analytics helpers for FalVision AI.
"""

import streamlit as st
from datetime import datetime


def init_history():
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0


def log_prediction(filename: str, result: dict):
    init_history()
    record = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename":   filename,
        "fruit":      result["quality_label"],   # reuse fruit key for compatibility
        "quality":    result["quality_label"],
        "confidence": round(result["confidence"], 2),
        "time_ms":    result["prediction_time"],
    }
    st.session_state.prediction_history.insert(0, record)
    st.session_state.total_predictions += 1


def get_analytics_summary():
    init_history()
    history = st.session_state.prediction_history

    if not history:
        return {"total": 0, "avg_confidence": 0.0, "top_fruit": "—", "quality_counts": {}}

    avg_conf = sum(r["confidence"] for r in history) / len(history)

    quality_counts = {}
    for r in history:
        q = r["quality"]
        quality_counts[q] = quality_counts.get(q, 0) + 1

    top_fruit = max(quality_counts, key=quality_counts.get)

    return {
        "total":          len(history),
        "avg_confidence": round(avg_conf, 1),
        "top_fruit":      top_fruit,
        "quality_counts": quality_counts,
    }
