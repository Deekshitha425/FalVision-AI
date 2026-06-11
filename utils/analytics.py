"""
utils/analytics.py
Session-state analytics for FalVision AI.
"""
import streamlit as st


def init_session():
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "last_result" not in st.session_state:
        pass  # set only after first prediction


def add_prediction(image_name, fruit_name, quality, confidence, timestamp):
    st.session_state.prediction_history.append({
        "image":      image_name,
        "fruit":      fruit_name,
        "quality":    quality,
        "confidence": round(confidence, 1),
        "timestamp":  timestamp,
    })


def get_analytics() -> dict:
    history = st.session_state.get("prediction_history", [])
    if not history:
        return {
            "total": 0, "avg_confidence": 0.0,
            "good_count": 0, "bad_count": 0, "mixed_count": 0,
            "quality_dist": {}, "top_fruit": "—",
        }

    total      = len(history)
    avg_conf   = sum(r["confidence"] for r in history) / total
    good_count  = sum(1 for r in history if r["quality"] == "Good Quality")
    bad_count   = sum(1 for r in history if r["quality"] == "Bad Quality")
    mixed_count = sum(1 for r in history if r["quality"] == "Mixed Quality")

    quality_dist: dict = {}
    fruit_count:  dict = {}
    for r in history:
        quality_dist[r["quality"]] = quality_dist.get(r["quality"], 0) + 1
        fruit_count[r["fruit"]]    = fruit_count.get(r["fruit"], 0) + 1

    top_fruit = max(fruit_count, key=fruit_count.get) if fruit_count else "—"

    return {
        "total": total, "avg_confidence": avg_conf,
        "good_count": good_count, "bad_count": bad_count, "mixed_count": mixed_count,
        "quality_dist": quality_dist, "top_fruit": top_fruit,
    }
