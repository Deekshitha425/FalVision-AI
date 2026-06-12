"""
app.py — FalVision AI · Fruit Quality Detection Platform
Run: streamlit run app.py
"""

import io
import os
import sys

import streamlit as st
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from utils.predictor import (
    load_model, predict_image, CLASS_NAMES,
    make_gradcam_heatmap, overlay_gradcam,
)
from utils.analytics import init_history, log_prediction, get_analytics_summary

# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FalVision AI — Fruit Quality Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_history()
if "page" not in st.session_state:
    st.session_state.page = "detection"


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 .5rem;'>
        <div class='sidebar-logo'>🌿 FalVision AI</div>
        <div class='sidebar-badge'>AgriTech Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    for key, (icon, label) in {
        "detection": ("🔬", "Fruit Detection"),
        "analytics": ("📊", "Analytics"),
        "about":     ("ℹ️",  "About"),
    }.items():
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem;color:rgba(255,255,255,.6);margin-bottom:.4rem;
                font-weight:700;letter-spacing:.05em;text-transform:uppercase;'>
        Model Details
    </div>
    """, unsafe_allow_html=True)

    for label, val in [
        ("Architecture", "MobileNetV2"),
        ("Framework",    "TensorFlow"),
        ("Method",       "Transfer Learning"),
        ("Input size",   "224 × 224 px"),
        ("Classes",      "Good / Bad / Mixed"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:.35rem 0;
                    border-bottom:1px solid rgba(255,255,255,.07);font-size:.82rem;'>
            <span style='color:rgba(255,255,255,.5);'>{label}</span>
            <span style='font-weight:600;'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem;color:rgba(255,255,255,.4);line-height:1.6;'>
        Upload any fruit image.<br>
        The AI returns quality status<br>
        instantly — no labels needed.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Page: Detection
# ═══════════════════════════════════════════════════════════════
def page_detection():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-eyebrow'>🌾 AI-Powered Quality Analysis</div>
        <h1 class='hero-title'>Is your fruit <span>good to go?</span></h1>
        <p class='hero-sub'>
            Upload any fruit image. The model classifies it as
            Good Quality, Bad Quality, or Mixed Quality — instantly.
            No labels, no setup.
        </p>
        <div class='hero-stats'>
            <div class='hero-stat'>
                <span class='hero-stat-val'>3</span>
                <span class='hero-stat-label'>Quality classes</span>
            </div>
            <div class='hero-stat'>
                <span class='hero-stat-val'>MobileNetV2</span>
                <span class='hero-stat-label'>Architecture</span>
            </div>
            <div class='hero-stat'>
                <span class='hero-stat-val'>19,526</span>
                <span class='hero-stat-label'>Training images</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading model…"):
        model = load_model("model/falvision_model.keras")

    if model is None:
        st.error("Model not found. Place `falvision_model.keras` in the `model/` folder.")
        return

    col_upload, col_result = st.columns([1, 1.2], gap="large")

    # ── Upload column ──────────────────────────────────────────
    with col_upload:
        st.markdown("<div class='section-heading'>📤 Upload Image</div>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop a fruit image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            img = Image.open(io.BytesIO(uploaded.read()))
            st.image(img, use_container_width=True)
            st.markdown(f"""
            <div style='font-size:.8rem;color:var(--gray-500);margin-top:.4rem;
                        display:flex;gap:1.2rem;'>
                <span>📄 {uploaded.name}</span>
                <span>📐 {img.width}×{img.height}px</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Result column ──────────────────────────────────────────
    with col_result:
        if not uploaded:
            st.markdown("""
            <div class='glass-card' style='text-align:center;padding:3rem 2rem;margin-top:2.5rem;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>🍑</div>
                <div style='font-weight:700;font-size:1.1rem;color:var(--green-800);
                            margin-bottom:.5rem;'>Ready to analyse</div>
                <div style='font-size:.88rem;color:var(--gray-500);line-height:1.6;'>
                    Upload a fruit image and the model will return
                    <strong>Good</strong>, <strong>Bad</strong>, or
                    <strong>Mixed</strong> quality — with confidence score
                    and actionable recommendations.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='section-heading'>🧠 Quality Result</div>",
                        unsafe_allow_html=True)

            with st.spinner("Analysing image…"):
                result = predict_image(model, img)

            log_prediction(uploaded.name, result)

            # Border colour per quality
            border_colors = {
                "success": "var(--green-600)",
                "warning": "#d97706",
                "error":   "#dc2626",
            }
            border = border_colors.get(result["quality_type"], "var(--green-600)")

            st.markdown(f"""
            <div class='result-wrap' style='border-left-color:{border};'>
                <div style='display:flex;align-items:center;gap:.8rem;margin-bottom:1rem;'>
                    <span style='font-size:3rem;'>{result['quality_emoji']}</span>
                    <div>
                        <div class='result-fruit'>{result['quality_label']}</div>
                        <div style='font-size:.88rem;color:var(--gray-500);margin-top:.2rem;'>
                            {result['quality_summary']}
                        </div>
                    </div>
                </div>

                <div class='conf-bar-wrap'>
                    <div class='conf-bar-label'>
                        <span>Confidence</span>
                        <span>{result['confidence']:.1f}%</span>
                    </div>
                    <div class='conf-bar-track'>
                        <div class='conf-bar-fill' style='width:{result['confidence']:.1f}%; background:linear-gradient(90deg,{border},{border});'></div>
                    </div>
                </div>

                <div style='font-size:.8rem;color:var(--gray-500);margin-top:.6rem;'>
                    ⚡ {result['prediction_time']} ms inference time
                </div>
            </div>
            """, unsafe_allow_html=True)

            # All 3 class scores
            st.markdown("<div class='section-heading'>📊 All Class Scores</div>",
                        unsafe_allow_html=True)

            display_names = {
                "Bad Quality_Fruits":   ("⛔ Bad Quality",   "#dc2626"),
                "Good Quality_Fruits":  ("✅ Good Quality",  "var(--green-600)"),
                "Mixed Qualit_Fruits":  ("⚠️ Mixed Quality", "#d97706"),
            }
            sorted_probs = sorted(result["all_probs"].items(),
                                  key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                name, color = display_names.get(cls, (cls, "var(--gray-500)"))
                is_top = cls == result["class_name"]
                st.markdown(f"""
                <div style='margin-bottom:.65rem;'>
                    <div style='display:flex;justify-content:space-between;font-size:.85rem;
                                font-weight:{"700" if is_top else "400"};
                                color:{"var(--green-900)" if is_top else "var(--gray-500)"};
                                margin-bottom:.25rem;'>
                        <span>{name}</span><span>{prob:.1f}%</span>
                    </div>
                    <div style='height:8px;background:var(--gray-100);
                                border-radius:99px;overflow:hidden;'>
                        <div style='width:{max(prob,0.5):.1f}%;height:100%;
                                    background:{color};border-radius:99px;
                                    transition:width .5s ease;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Grad-CAM ────────────────────────────────────────
            st.markdown("<div class='section-heading'>🔥 Grad-CAM — Where the Model Looked</div>",
                        unsafe_allow_html=True)
            try:
                from utils.preprocessing import preprocess_image

                img_array = preprocess_image(img)
                pred_idx  = CLASS_NAMES.index(result["class_name"])

                heatmap = make_gradcam_heatmap(img_array, model, pred_idx)
                overlay = overlay_gradcam(img_array[0], heatmap)

                # Colorize heatmap for display (JET colormap)
                import cv2
                heatmap_uint8   = np.uint8(255 * heatmap)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                gc_col1, gc_col2 = st.columns(2)
                with gc_col1:
                    st.image(heatmap_colored, caption="Activation Heatmap",
                             use_container_width=True)
                with gc_col2:
                    st.image(overlay, caption="Overlay on Original",
                             use_container_width=True)

                st.caption("🔴 Red/yellow regions show where the model focused most "
                           "when making this prediction.")
            except Exception as e:
                st.info(f"Grad-CAM unavailable: {e}")

    # ── Recommendations + Tips (shown below once result exists) ──
    if uploaded and "result" in dir() and result:
        st.markdown("<div class='section-heading'>💡 Recommendations</div>",
                    unsafe_allow_html=True)
        rec_cols = st.columns(3)
        for col, rec in zip(rec_cols, result["recommendations"]):
            with col:
                st.markdown(f"<div class='rec-chip'>{rec}</div>",
                            unsafe_allow_html=True)

        st.markdown("<div class='section-heading'>📋 Handling & Storage Guidelines</div>",
                    unsafe_allow_html=True)
        tips = result["tips"]
        st.markdown(f"""
        <div class='info-grid'>
            <div class='info-tile'>
                <div class='info-tile-label'>❄️ Storage</div>
                <div class='info-tile-text'>{tips['storage']}</div>
            </div>
            <div class='info-tile'>
                <div class='info-tile-label'>🤲 Handling</div>
                <div class='info-tile-text'>{tips['handling']}</div>
            </div>
            <div class='info-tile'>
                <div class='info-tile-label'>📝 Action Note</div>
                <div class='info-tile-text'>{tips['note']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Page: Analytics
# ═══════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("""
    <div class='hero-wrap' style='padding:2rem 2.5rem;'>
        <div class='hero-eyebrow'>📊 Session Dashboard</div>
        <h1 class='hero-title' style='font-size:2rem;'>Prediction <span>Analytics</span></h1>
        <p class='hero-sub' style='margin-bottom:0;'>
            Real-time overview of all quality checks run in this session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    summary = get_analytics_summary()
    history = st.session_state.prediction_history

    st.markdown(f"""
    <div class='metric-grid'>
        <div class='metric-card'>
            <div class='metric-card-icon'>🔬</div>
            <div class='metric-card-value'>{summary['total']}</div>
            <div class='metric-card-label'>Total Checks</div>
        </div>
        <div class='metric-card'>
            <div class='metric-card-icon'>🎯</div>
            <div class='metric-card-value'>{summary['avg_confidence']}%</div>
            <div class='metric-card-label'>Avg Confidence</div>
        </div>
        <div class='metric-card'>
            <div class='metric-card-icon'>🏆</div>
            <div class='metric-card-value' style='font-size:1rem;'>{summary['top_fruit']}</div>
            <div class='metric-card-label'>Most Common Result</div>
        </div>
        <div class='metric-card'>
            <div class='metric-card-icon'>📈</div>
            <div class='metric-card-value'>{len(summary['quality_counts'])}</div>
            <div class='metric-card-label'>Quality Levels Seen</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:3rem;'>
            <div style='font-size:2.5rem;margin-bottom:.8rem;'>📭</div>
            <div style='font-weight:700;color:var(--green-800);'>No checks yet</div>
            <div style='font-size:.88rem;color:var(--gray-500);margin-top:.4rem;'>
                Upload a fruit image in Fruit Detection to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Quality distribution
    if summary["quality_counts"]:
        st.markdown("<div class='section-heading'>📈 Quality Distribution</div>",
                    unsafe_allow_html=True)
        q_cols = st.columns(len(summary["quality_counts"]))
        for col, (quality, count) in zip(q_cols, summary["quality_counts"].items()):
            pct = round(count / summary["total"] * 100)
            with col:
                st.markdown(f"""
                <div class='glass-card' style='text-align:center;'>
                    <div style='font-size:1.5rem;font-weight:800;color:var(--green-700);'>{count}</div>
                    <div style='font-size:.78rem;color:var(--gray-500);margin-top:.2rem;'>{quality}</div>
                    <div style='font-size:.7rem;color:var(--green-600);font-weight:600;'>{pct}%</div>
                </div>
                """, unsafe_allow_html=True)

    # History table
    st.markdown("<div class='section-heading'>🗂️ Prediction History</div>",
                unsafe_allow_html=True)
    rows = ""
    for r in history:
        rows += f"""
        <tr>
            <td>{r['timestamp']}</td>
            <td style='max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                title='{r['filename']}'>{r['filename']}</td>
            <td>{r['quality']}</td>
            <td>
                <div style='display:flex;align-items:center;gap:.5rem;'>
                    <div style='flex:1;height:6px;background:var(--gray-100);
                                border-radius:99px;overflow:hidden;'>
                        <div style='width:{r['confidence']}%;height:100%;
                                    background:var(--green-500);border-radius:99px;'></div>
                    </div>
                    <span style='font-weight:600;font-size:.82rem;'>{r['confidence']}%</span>
                </div>
            </td>
            <td>{r['time_ms']} ms</td>
        </tr>
        """
    st.markdown(f"""
    <div style='overflow-x:auto;border-radius:var(--radius);box-shadow:var(--shadow-sm);'>
        <table class='hist-table'>
            <thead>
                <tr>
                    <th>Timestamp</th><th>File</th><th>Quality Result</th>
                    <th>Confidence</th><th>Time</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear History", type="secondary"):
        st.session_state.prediction_history = []
        st.session_state.total_predictions  = 0
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# Page: About
# ═══════════════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div class='hero-wrap' style='padding:2rem 2.5rem;'>
        <div class='hero-eyebrow'>ℹ️ Project Overview</div>
        <h1 class='hero-title' style='font-size:2rem;'>About <span>FalVision AI</span></h1>
        <p class='hero-sub' style='margin-bottom:0;'>
            End-to-end ML engineering — from raw dataset to deployed web application.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-heading' style='margin-top:0;'>🎯 What It Does</div>
            <p style='font-size:.9rem;color:var(--gray-700);line-height:1.7;margin:0;'>
                FalVision AI takes any fruit image and classifies it into one of three
                quality tiers — <strong>Good</strong>, <strong>Bad</strong>, or
                <strong>Mixed</strong> — using a fine-tuned MobileNetV2 model trained
                on 19,526 images across three quality classes.
                It is designed to support farmers, distributors, and QA teams in making
                fast, data-driven decisions about fruit batches.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card' style='margin-top:1rem;'>
            <div class='section-heading' style='margin-top:0;'>🏗️ Model Architecture</div>
            <div style='font-size:.88rem;color:var(--gray-700);line-height:1.9;'>
                <b>Base:</b> MobileNetV2 (ImageNet weights)<br>
                <b>Phase 1:</b> Head-only training — base frozen, LR = 1e-3<br>
                <b>Phase 2:</b> Fine-tune layers 100+ — LR = 1e-5<br>
                <b>Head:</b> GAP → BatchNorm → Dropout(0.5) → Dense(256) → Softmax(3)<br>
                <b>Regularisation:</b> L2 + Dropout<br>
                <b>Callbacks:</b> EarlyStopping, ReduceLROnPlateau, ModelCheckpoint<br>
                <b>Class weights:</b> Balanced (Mixed class heavily upweighted at 6.055×)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class='glass-card'>
            <div class='section-heading' style='margin-top:0;'>📦 Dataset</div>
        """, unsafe_allow_html=True)
        for cls, count, weight in [
            ("✅ Good Quality_Fruits",  "majority", "0.558×"),
            ("⛔ Bad Quality_Fruits",   "moderate", "0.959×"),
            ("⚠️ Mixed Qualit_Fruits",  "minority", "6.055×"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:.5rem 0;border-bottom:1px solid var(--gray-100);
                        font-size:.85rem;'>
                <span>{cls}</span>
                <span style='color:var(--gray-500);font-size:.78rem;'>
                    {count} · weight {weight}
                </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
            <div style='font-size:.78rem;color:var(--gray-500);margin-top:.6rem;'>
                Total: 19,526 images · 80/20 train-val split
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='glass-card' style='margin-top:1rem;'>
            <div class='section-heading' style='margin-top:0;'>🛠️ Tech Stack</div>
        """, unsafe_allow_html=True)
        for icon, name, desc in [
            ("🧠", "TensorFlow / Keras", "Model training & inference"),
            ("🌐", "Streamlit",          "Web application"),
            ("🖼️", "Pillow",            "Image preprocessing"),
            ("📊", "NumPy",             "Numerical ops"),
            ("🎨", "Custom CSS",        "AgriTech theme"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.8rem;padding:.5rem 0;
                        border-bottom:1px solid var(--gray-100);'>
                <span style='font-size:1.2rem;'>{icon}</span>
                <div>
                    <div style='font-weight:600;font-size:.85rem;color:var(--green-800);'>{name}</div>
                    <div style='font-size:.76rem;color:var(--gray-500);'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-heading'>📁 Project Structure</div>",
                unsafe_allow_html=True)
    st.code("""
FalVisionAI/
├── app.py                  # Main Streamlit application
├── style.css               # Premium AgriTech theme
├── requirements.txt
├── README.md
├── model/
│   └── falvision_model.keras
└── utils/
    ├── predictor.py        # Inference + class metadata
    ├── preprocessing.py    # Image pipeline
    └── analytics.py        # Session history
    """, language="text")


# ═══════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════
page = st.session_state.get("page", "detection")
if page == "detection":
    page_detection()
elif page == "analytics":
    page_analytics()
elif page == "about":
    page_about()
