import streamlit as st
import time
import io
from utils.predictor import load_model, predict_image, FRUIT_LIST, generate_gradcam, CLASS_NAMES
from utils.analytics import init_session, add_prediction, get_analytics
from utils.fruit_info import get_fruit_info, get_ai_insights

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="फल Vision AI — Fruit Quality Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
init_session()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="hindi-fal">फल</span><span class="brand-vision">Vision AI</span>
    </div>
    <p class="sidebar-tagline">AI-Powered Fruit Quality Detection</p>
    <hr class="sidebar-divider"/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🍎 Fruit Quality Detection", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="sidebar-section">
        <h4>🤖 Model Info</h4>
        <div class="info-chip">Architecture: MobileNetV2</div>
        <div class="info-chip">Framework: TensorFlow</div>
        <div class="info-chip">Method: Transfer Learning</div>
        <div class="info-chip">Input: 224 × 224 px</div>
        <div class="info-chip">Classes: 3 Quality Grades</div>
        <div class="info-chip">Layers: 158 total</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>📌 About</h4>
        <p class="sidebar-body">
        <strong>फल Vision AI</strong> uses deep learning to classify
        fruit quality instantly — helping farmers, distributors and
        agri-businesses make smarter decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    analytics = get_analytics()
    if analytics["total"] > 0:
        st.markdown(f"""
        <div class="sidebar-section">
            <h4>📊 Session Stats</h4>
            <div class="info-chip">Predictions: {analytics['total']}</div>
            <div class="info-chip">Avg Confidence: {analytics['avg_confidence']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-badge">🌿 Powered by Deep Learning &amp; Transfer Learning</div>
            <h1 class="hero-title">
                <span class="hindi-fal-hero">फल</span>Vision AI
            </h1>
            <p class="hero-subtitle">AI-Powered Fruit Quality Detection for Smarter Farming</p>
            <p class="hero-desc">
                Analyse fruit quality instantly using advanced computer vision technology
                designed for farmers, distributors, and agricultural businesses.
            </p>
            <div class="hero-pills">
                <div class="hero-pill">🎯 MobileNetV2</div>
                <div class="hero-pill">⚡ Real-Time Inference</div>
                <div class="hero-pill">🌾 AgriTech Grade AI</div>
                <div class="hero-pill">🔬 Transfer Learning</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    features = [
        ("🔬", "Deep Learning Core",
         "MobileNetV2 pretrained on ImageNet, fine-tuned with 2-phase transfer learning for fruit quality classification."),
        ("⚡", "Instant Inference",
         "Get quality predictions in milliseconds — ready for real-time conveyor belt or field inspection workflows."),
        ("📈", "Smart Analytics",
         "Track prediction history, confidence trends and quality distribution across all your session assessments."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quality classes
    st.markdown("""
    <div class="section-header">
        <h2>Quality Classifications</h2>
        <p>The model outputs one of three quality grades</p>
    </div>
    """, unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)
    classes = [
        (q1, "✅", "Good Quality",  "#22c55e", "Uniform colour, firm texture, no visible defects. Cleared for retail sale and direct distribution."),
        (q2, "⚠️", "Mixed Quality", "#f59e0b", "Inconsistent specimens detected. Requires manual sorting before dispatch to market."),
        (q3, "❌", "Bad Quality",   "#ef4444", "Significant bruising, discolouration or spoilage detected. Not suitable for retail distribution."),
    ]
    for col, emoji, label, color, desc in classes:
        with col:
            st.markdown(f"""
            <div class="quality-class-card" style="border-top: 4px solid {color}">
                <div class="qc-emoji">{emoji}</div>
                <h4 style="color:{color}">{label}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="section-header">
        <h2>How It Works</h2>
        <p>Three steps to fruit quality intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    steps = [
        (s1, "01", "📸", "Upload Image",
         "Drag-and-drop or browse a clear JPG/PNG photo of the fruit you want to inspect."),
        (s2, "02", "🧠", "AI Analysis",
         "MobileNetV2 preprocesses the 224×224 image and outputs softmax class probabilities."),
        (s3, "03", "📋", "Get Insights",
         "Receive quality grade, confidence score, per-class probabilities, and actionable recommendations."),
    ]
    for col, num, icon, title, desc in steps:
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cta-bar">
        ← Select <strong>Fruit Quality Detection</strong> from the sidebar to begin
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FRUIT QUALITY DETECTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🍎 Fruit Quality Detection":

    st.markdown("""
    <div class="page-header">
        <h1>🍎 Fruit Quality Detection</h1>
        <p>Select fruit type, upload an image, and receive an instant quality assessment powered by MobileNetV2</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model (cached)
    with st.spinner("⚙️ Loading AI model…"):
        model = load_model("model/falvision_model.keras")

    st.markdown('<div class="detect-success">✅ Model loaded — MobileNetV2 (158 layers, 3 quality classes)</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_up, col_res = st.columns([1, 1], gap="large")

    # ── Upload column ────────────────────────────────────────────────────────
    with col_up:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🍎 Select Fruit Type")
        st.markdown("""
        <p style="font-size:0.82rem;color:#6b7280;margin:-0.5rem 0 0.75rem 0">
        The model detects quality (Good / Mixed / Bad). Select the fruit you are inspecting.
        </p>
        """, unsafe_allow_html=True)

        selected_fruit = st.selectbox(
            "Fruit type",
            FRUIT_LIST,
            label_visibility="collapsed",
        )

        st.markdown("#### 📤 Upload Fruit Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            uploaded_file.seek(0)
            st.image(uploaded_file, caption=uploaded_file.name,
                     use_container_width=True)
            st.markdown(f"""
            <div class="file-meta">
                <span>📄 {uploaded_file.name}</span>
                <span>{uploaded_file.size / 1024:.1f} KB</span>
                <span>{uploaded_file.type}</span>
            </div>
            """, unsafe_allow_html=True)
            analyse = st.button("🔍 Analyse Quality", use_container_width=True,
                                type="primary")
        else:
            st.markdown("""
            <div class="upload-placeholder">
                <div class="upload-icon">🌿</div>
                <p>Supported: JPG · JPEG · PNG</p>
                <p class="upload-hint">Use a clear, well-lit photo for best accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            analyse = False

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Result column ────────────────────────────────────────────────────────
    with col_res:
        if uploaded_file and analyse:
            t0 = time.time()
            with st.spinner("🧠 Analysing quality…"):
                result = predict_image(model, uploaded_file, selected_fruit)
            elapsed_ms = (time.time() - t0) * 1000

            add_prediction(
                image_name=uploaded_file.name,
                fruit_name=result["fruit_name"],
                quality=result["quality_label"],
                confidence=result["confidence"],
                timestamp=result["timestamp"],
            )

            quality_cfg = {
                "Good Quality":  ("#22c55e", "✅", "Cleared for sale"),
                "Mixed Quality": ("#f59e0b", "⚠️", "Inspection required"),
                "Bad Quality":   ("#ef4444", "❌", "Not market-ready"),
            }
            color, emoji, verdict = quality_cfg.get(
                result["quality_label"], ("#6b7280", "🔍", "—"))

            st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid {color}">
                <div class="result-header">
                    <span class="result-emoji">{emoji}</span>
                    <div class="result-text">
                        <h2 class="result-fruit">{result['fruit_name']}</h2>
                        <div class="quality-badge" style="background:{color}18;color:{color};border:1px solid {color}44">
                            {result['quality_label']}
                        </div>
                        <p class="verdict-line">{verdict}</p>
                    </div>
                </div>
                <div class="result-metrics">
                    <div class="metric-box">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value" style="color:{color}">{result['confidence']:.1f}%</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Inference</span>
                        <span class="metric-value">{elapsed_ms:.0f} ms</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Model</span>
                        <span class="metric-value">MobileNetV2</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Time</span>
                        <span class="metric-value">{result['timestamp'][11:]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence meter
            st.markdown(f"""
            <div class="conf-meter-label">
                <span>Confidence Score</span>
                <span style="color:{color};font-weight:700">{result['confidence']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(result["confidence"] / 100)

            # Per-class probability bars
            st.markdown('<div class="prob-section"><h5>Class Probabilities</h5>', unsafe_allow_html=True)
            bar_colors = {
                "Good Quality":  "#22c55e",
                "Mixed Quality": "#f59e0b",
                "Bad Quality":   "#ef4444",
            }
            for lbl in ["Good Quality", "Mixed Quality", "Bad Quality"]:
                pct = result["probs"][lbl] * 100
                bc  = bar_colors[lbl]
                st.markdown(f"""
                <div class="prob-row">
                    <span class="prob-label">{lbl}</span>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{pct:.1f}%;background:{bc}"></div>
                    </div>
                    <span class="prob-pct" style="color:{bc}">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Store result + file bytes for GradCAM below
            uploaded_file.seek(0)
            st.session_state["last_result"]     = result
            st.session_state["last_img_bytes"]  = uploaded_file.read()
            st.session_state["last_pred_idx"]   = CLASS_NAMES.index(
                [k for k, v in {"bad_quality": "Bad Quality",
                                 "good_quality": "Good Quality",
                                 "mixed_quality": "Mixed Quality"}.items()
                 if v == result["quality_label"]][0]
            )

        elif not uploaded_file:
            st.markdown("""
            <div class="empty-result">
                <div class="empty-icon">🍃</div>
                <h3>Awaiting Image</h3>
                <p>Select a fruit, upload an image, and click <strong>Analyse Quality</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── GradCAM section ───────────────────────────────────────────────────────
    if "last_result" in st.session_state and "last_img_bytes" in st.session_state:
        result = st.session_state["last_result"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-divider">🔬 GradCAM — AI Attention Heatmap</div>',
                    unsafe_allow_html=True)

        col_orig, col_cam = st.columns(2, gap="large")

        with col_orig:
            st.markdown('<div class="card"><h4>📸 Original Image</h4>', unsafe_allow_html=True)
            orig_bytes = io.BytesIO(st.session_state["last_img_bytes"])
            st.image(orig_bytes, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_cam:
            st.markdown('<div class="card"><h4>🌡️ GradCAM Heatmap</h4>', unsafe_allow_html=True)
            st.markdown("""
            <p style="font-size:0.8rem;color:#6b7280;margin:-0.5rem 0 0.75rem 0">
            Bright green regions = areas the model focused on most to make its prediction.
            </p>
            """, unsafe_allow_html=True)
            with st.spinner("Generating GradCAM…"):
                cam_file = io.BytesIO(st.session_state["last_img_bytes"])
                cam_file.name = "img.jpg"
                gradcam_img = generate_gradcam(
                    model, cam_file,
                    st.session_state["last_pred_idx"]
                )
            st.image(gradcam_img, use_container_width=True)
            st.markdown(f"""
            <div class="gradcam-label" style="background:{
                '#22c55e' if result['quality_label']=='Good Quality'
                else '#f59e0b' if result['quality_label']=='Mixed Quality'
                else '#ef4444'}18;
                border:1px solid {'#22c55e' if result['quality_label']=='Good Quality'
                else '#f59e0b' if result['quality_label']=='Mixed Quality'
                else '#ef4444'}44;
                border-radius:8px;padding:0.5rem 0.75rem;margin-top:0.5rem;
                font-size:0.82rem;font-weight:600;color:#374151">
                Visualising decision for: <strong>{result['quality_label']}</strong>
                ({result['confidence']:.1f}% confidence)
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── AI Insights + Fruit Info ──────────────────────────────────────────────
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-divider">💡 AI Insights &amp; Fruit Information</div>',
                    unsafe_allow_html=True)

        col_ins, col_info = st.columns(2, gap="large")

        with col_ins:
            ins = get_ai_insights(result["quality_label"])
            tips_html = "".join(f"<li>{t}</li>" for t in ins["tips"])
            st.markdown(f"""
            <div class="card">
                <h4>💡 AI Recommendations</h4>
                <div class="insight-block">
                    <span class="insight-icon">{ins['icon']}</span>
                    <div>
                        <p class="insight-main">{ins['main']}</p>
                        <p class="insight-sub">{ins['sub']}</p>
                    </div>
                </div>
                <ul class="insight-list">{tips_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        with col_info:
            fi = get_fruit_info(result["fruit_name"])
            st.markdown(f"""
            <div class="card">
                <h4>🍎 {result['fruit_name']} — Quick Reference</h4>
                <div class="fi-grid">
                    <div class="fi-item">
                        <span class="fi-label">🥗 Nutrition</span>
                        <span class="fi-val">{fi['nutrition']}</span>
                    </div>
                    <div class="fi-item">
                        <span class="fi-label">📦 Storage</span>
                        <span class="fi-val">{fi['storage']}</span>
                    </div>
                    <div class="fi-item">
                        <span class="fi-label">🤲 Handling</span>
                        <span class="fi-val">{fi['handling']}</span>
                    </div>
                    <div class="fi-item">
                        <span class="fi-label">🕐 Shelf Life</span>
                        <span class="fi-val">{fi['shelf_life']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown("""
    <div class="page-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>Session-level insights across all your fruit quality predictions</p>
    </div>
    """, unsafe_allow_html=True)

    analytics = get_analytics()

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "🔍", "Total Predictions",  str(analytics["total"])),
        (k2, "🎯", "Avg Confidence",     f"{analytics['avg_confidence']:.1f}%"),
        (k3, "✅", "Good Quality",        str(analytics["good_count"])),
        (k4, "❌", "Bad / Mixed",         str(analytics["bad_count"] +
                                             analytics.get("mixed_count", 0))),
    ]
    for col, icon, label, val in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if analytics["total"] > 0:
        import pandas as pd

        col_ch, col_tb = st.columns([1, 1], gap="large")

        with col_ch:
            st.markdown('<div class="card"><h4>Quality Distribution</h4>', unsafe_allow_html=True)
            dist_df = pd.DataFrame(
                list(analytics["quality_dist"].items()),
                columns=["Quality", "Count"]
            )
            st.bar_chart(dist_df.set_index("Quality"), color="#22c55e", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Top fruit
            st.markdown(f"""
            <div class="card">
                <h4>🏆 Most Detected Fruit</h4>
                <div class="top-fruit">{analytics['top_fruit']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_tb:
            st.markdown('<div class="card"><h4>Prediction History</h4>', unsafe_allow_html=True)
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence %", min_value=0, max_value=100,
                        format="%.1f%%"
                    ),
                    "image":     st.column_config.TextColumn("Image"),
                    "fruit":     st.column_config.TextColumn("Fruit"),
                    "quality":   st.column_config.TextColumn("Quality"),
                    "timestamp": st.column_config.TextColumn("Time"),
                },
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Session History", type="secondary"):
            st.session_state.prediction_history = []
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.rerun()

    else:
        st.markdown("""
        <div class="empty-result" style="margin-top:2rem">
            <div class="empty-icon">📊</div>
            <h3>No Predictions Yet</h3>
            <p>Run predictions in <strong>Fruit Detection</strong> to populate this dashboard.</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div class="page-header">
        <h1>ℹ️ About <span class="hindi-fal">फल</span>Vision AI</h1>
        <p>Deep learning meets real-world agriculture</p>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("""
        <div class="card">
            <h4>🎯 Mission</h4>
            <p>
            <strong>फल Vision AI</strong> bridges the gap between computer vision research
            and real-world agricultural quality control. Using MobileNetV2 with two-phase
            transfer learning, the system classifies fruit quality in milliseconds — empowering
            farmers, packers, and distributors to make data-driven decisions at scale.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h4>🔧 Tech Stack</h4>
            <div class="tech-grid">
                <div class="tech-chip">🧠 TensorFlow 2.x</div>
                <div class="tech-chip">📦 MobileNetV2</div>
                <div class="tech-chip">🎨 Streamlit</div>
                <div class="tech-chip">🐍 Python 3.10+</div>
                <div class="tech-chip">🖼️ Pillow</div>
                <div class="tech-chip">🔢 NumPy</div>
                <div class="tech-chip">📊 Pandas</div>
                <div class="tech-chip">🤗 ImageNet Weights</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="card">
            <h4>📐 Model Architecture</h4>
            <ul class="about-list">
                <li><strong>Base:</strong> MobileNetV2 — ImageNet pretrained</li>
                <li><strong>GAP:</strong> GlobalAveragePooling2D (1280-d vector)</li>
                <li><strong>BN:</strong> BatchNormalization</li>
                <li><strong>Dropout:</strong> rate = 0.3</li>
                <li><strong>Head:</strong> Dense(3, activation='softmax')</li>
                <li><strong>Total layers:</strong> 158</li>
                <li><strong>Input:</strong> 224 × 224 × 3 (float32, [0,1])</li>
                <li><strong>Output:</strong> 3-class softmax probabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h4>🏋️ Training Details</h4>
            <ul class="about-list">
                <li><strong>Phase 1:</strong> Head-only, base frozen — LR 1e-3</li>
                <li><strong>Phase 2:</strong> Fine-tune layers ≥100 — LR 1e-5</li>
                <li><strong>Augmentation:</strong> Flip, rotation ±15°, zoom 0.2, brightness [0.8–1.2]</li>
                <li><strong>Loss:</strong> Categorical cross-entropy + class weights</li>
                <li><strong>Callbacks:</strong> EarlyStopping, ReduceLROnPlateau, ModelCheckpoint</li>
                <li><strong>Dataset:</strong> ryandpark/fruit-quality-classification (Kaggle)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>📊 Output Classes</h4>
        <div class="class-list">
            <div class="class-row" style="border-left:4px solid #22c55e">
                ✅ <strong>Good Quality</strong> (index 1) — Uniform colour, firm texture, no visible defects. Ready for sale.
            </div>
            <div class="class-row" style="border-left:4px solid #f59e0b">
                ⚠️ <strong>Mixed Quality</strong> (index 2) — Inconsistent specimens. Requires manual sorting before dispatch.
            </div>
            <div class="class-row" style="border-left:4px solid #ef4444">
                ❌ <strong>Bad Quality</strong> (index 0) — Significant bruising or spoilage. Not suitable for retail.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Built with 💚 for AgriTech Innovation ·
        <strong>फल Vision AI</strong> ·
        Powered by TensorFlow &amp; Streamlit
    </div>
    """, unsafe_allow_html=True)
