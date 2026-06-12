# फल Vision AI 🌿

> **AI-Powered Fruit Quality Detection for Smarter Farming**

A production-quality AgriTech web application built with TensorFlow and Streamlit that classifies fruit quality in real-time using MobileNetV2 transfer learning.

---

## 📸 Features

| Feature | Description |
|---|---|
| 🔬 Deep Learning | MobileNetV2 backbone, 2-phase transfer learning |
| ⚡ Real-Time | Sub-100ms inference per image |
| 📊 Analytics | Session-level prediction history & distribution charts |
| 💡 AI Insights | Dynamic quality-based recommendations |
| 🍎 Fruit Info | Nutrition, storage & handling per detected fruit |
| 🎨 Premium UI | Glassmorphism cards, animated progress bars, green AgriTech theme |

---

## 🏗️ Project Structure

```
FalVisionAI/
├── app.py                  # Main Streamlit application
├── style.css               # Complete custom CSS design system
├── requirements.txt
├── README.md
├── model/
│   └── falvision_model.keras   # Trained MobileNetV2 model
├── utils/
│   ├── __init__.py
│   ├── predictor.py        # Model loading & inference pipeline
│   ├── analytics.py        # Session-state analytics
│   └── fruit_info.py       # Fruit data & AI insight generation
└── assets/                 # Static assets (optional icons/banners)
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (ImageNet pretrained) |
| Head | GAP → BatchNorm → Dropout(0.3) → Dense(3, softmax) |
| Total Layers | 158 |
| Input Shape | (224, 224, 3) — float32 normalised to [0, 1] |
| Output Classes | 3 |
| Training | 2-phase transfer learning |

### Class Labels (alphabetical / index order)

| Index | Raw Class | Display Label |
|---|---|---|
| 0 | `bad_quality` | ❌ Bad Quality |
| 1 | `good_quality` | ✅ Good Quality |
| 2 | `mixed_quality` | ⚠️ Mixed Quality |

---

## 🚀 Local Setup

```bash
# 1. Clone / download the project
cd FalVisionAI

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---



---

*Built with 💚 for AgriTech Innovation · Powered by TensorFlow & Streamlit*
