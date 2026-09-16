"""
app_onnx.py  —  Wildfire Detection  |  Standalone Streamlit App
================================================================
Copy these two files to the other PC:
    fire_detector.onnx
    app_onnx.py

Install (one-time):
    pip install streamlit onnxruntime pillow numpy

Run:
    streamlit run app_onnx.py
"""

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Wildfire Sentinel",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Dark background ── */
  .stApp { background: #0d1117; }
  section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

  /* ── Header banner ── */
  .hero {
    background: linear-gradient(135deg, #ff4500 0%, #ff8c00 50%, #ffd700 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    border-radius: 16px;
  }
  .hero-title {
    font-size: 2.4rem; font-weight: 800;
    color: #fff; text-shadow: 0 2px 12px rgba(0,0,0,0.3);
    margin: 0; position: relative;
  }
  .hero-sub {
    font-size: 1rem; color: rgba(255,255,255,0.85);
    margin-top: 6px; font-weight: 400; position: relative;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; }
  .metric-value { font-size: 2rem; font-weight: 700; margin-top: 4px; }

  /* ── Result banners ── */
  .result-fire {
    background: linear-gradient(135deg, #3d0000, #7a1a00);
    border: 1px solid #ff4500;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
  }
  .result-safe {
    background: linear-gradient(135deg, #003d1a, #005a25);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
  }
  .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }
  .result-sub   { font-size: 0.9rem; color: rgba(255,255,255,0.75); }

  /* ── Probability bar ── */
  .prob-bar-wrap {
    background: #21262d;
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    margin: 10px 0 4px;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
  }

  /* ── History table ── */
  .hist-row {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 8px;
  }
  .hist-label { flex: 1; font-size: 0.85rem; color: #c9d1d9; }
  .hist-badge-fire { background: #ff4500; color: #fff; border-radius: 6px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
  .hist-badge-safe { background: #22c55e; color: #fff; border-radius: 6px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
  .hist-prob { color: #8b949e; font-size: 0.82rem; min-width: 52px; text-align: right; }

  /* ── Sidebar labels ── */
  .sidebar-section { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: .1em; margin: 20px 0 6px; }

  /* Hide default header decoration */
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_SIZE     = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MODEL  = os.path.join(os.path.dirname(__file__), "fire_detector.onnx")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {name, prob, is_fire}
if "session" not in st.session_state:
    st.session_state.session = None
if "model_path" not in st.session_state:
    st.session_state.model_path = DEFAULT_MODEL

# ── Helper: load ONNX session ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime not installed. Run:  pip install onnxruntime"
    if not os.path.exists(model_path):
        return None, f"Model not found at: {model_path}"
    providers = ort.get_available_providers()
    prov = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=prov)
    device = "GPU (CUDA)" if "CUDA" in prov[0] else "CPU"
    return sess, device

# ── Helper: preprocess ────────────────────────────────────────────────────────
def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    left  = (img.width  - INPUT_SIZE) // 2
    upper = (img.height - INPUT_SIZE) // 2
    img = img.crop((left, upper, left + INPUT_SIZE, upper + INPUT_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

# ── Helper: run inference ─────────────────────────────────────────────────────
def run_inference(session, pil_image: Image.Image, threshold: float):
    inp = preprocess(pil_image)
    name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    logit = session.run(None, {name: inp})[0][0, 0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return prob, prob >= threshold, elapsed_ms

# ── Helper: probability bar HTML ─────────────────────────────────────────────
def prob_bar(prob: float) -> str:
    pct = prob * 100
    color = f"hsl({int((1-prob)*120)}, 90%, 50%)"   # green → red
    return f"""
    <div class="prob-bar-wrap">
      <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8b949e;">
      <span>0%</span><span style="font-weight:600;color:{color}">{pct:.1f}%</span><span>100%</span>
    </div>
    """

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)

    model_path = st.text_input(
        "ONNX model path",
        value=st.session_state.model_path,
        label_visibility="collapsed",
        placeholder="Path to fire_detector.onnx",
    )
    st.session_state.model_path = model_path

    session, device_info = load_session(model_path)

    if session:
        st.success(f"Model loaded  |  {device_info}")
    else:
        st.error(device_info)

    st.markdown('<div class="sidebar-section">Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Fire threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability above this is classified as FIRE")

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Single Image", "Batch Analysis"], label_visibility="collapsed")

    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#8b949e;line-height:1.6'>
    <b>Dependencies</b><br>
    pip install streamlit onnxruntime pillow numpy<br><br>
    <b>Run</b><br>
    streamlit run app_onnx.py
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Wildfire Sentinel</div>
  <div class="hero-sub">AI-powered fire detection from satellite & aerial imagery &mdash; ONNX inference, no GPU required</div>
</div>
""", unsafe_allow_html=True)

# Guard: model must be loaded
if not session:
    st.warning("Point the sidebar to your `fire_detector.onnx` file to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: SINGLE IMAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Single Image":
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload Image")
        uploaded = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, use_container_width=True, caption=uploaded.name)

            run_btn = st.button("Run Detection", type="primary", use_container_width=True)
        else:
            st.markdown("""
            <div style="
              border:2px dashed #30363d;border-radius:12px;
              padding:60px 20px;text-align:center;color:#8b949e;
            ">
              <div style="font-size:2.5rem;margin-bottom:8px">🛰️</div>
              <div style="font-weight:500">Upload satellite or aerial imagery</div>
              <div style="font-size:0.82rem;margin-top:4px">JPG, PNG, BMP, WebP</div>
            </div>
            """, unsafe_allow_html=True)
            run_btn = False

    with col_result:
        st.markdown("#### Detection Result")

        if uploaded and run_btn:
            with st.spinner("Analysing..."):
                prob, is_fire, ms = run_inference(session, pil_img, threshold)

            # Save to history
            st.session_state.history.insert(0, {
                "name": uploaded.name,
                "prob": prob,
                "is_fire": is_fire,
            })

            if is_fire:
                st.markdown(f"""
                <div class="result-fire">
                  <div class="result-title">FIRE DETECTED</div>
                  <div class="result-sub">The model flagged this image as containing active fire or smoke.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                  <div class="result-title">No Fire Detected</div>
                  <div class="result-sub">The model found no evidence of active fire in this image.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**Fire Probability**")
            st.markdown(prob_bar(prob), unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">Probability</div>
                  <div class="metric-value" style="color:{'#ff4500' if is_fire else '#22c55e'}">{prob:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">Inference</div>
                  <div class="metric-value" style="color:#58a6ff">{ms:.0f}<span style="font-size:1rem"> ms</span></div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">Threshold</div>
                  <div class="metric-value" style="color:#8b949e">{threshold:.0%}</div>
                </div>""", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div style="
              border:1px solid #30363d;border-radius:12px;
              padding:60px 20px;text-align:center;color:#8b949e;
            ">
              <div style="font-size:2rem;margin-bottom:8px">📊</div>
              <div>Results will appear here after analysis</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Session history ──
    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### Analysis History")
        for item in st.session_state.history[:10]:
            badge = '<span class="hist-badge-fire">FIRE</span>' if item["is_fire"] else '<span class="hist-badge-safe">SAFE</span>'
            st.markdown(f"""
            <div class="hist-row">
              <div class="hist-label">{item['name']}</div>
              {badge}
              <div class="hist-prob">{item['prob']:.1%}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("#### Batch Image Analysis")
    st.markdown("Upload multiple images at once and get a summary table with fire risk for each.")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button(f"Analyse all {len(uploaded_files)} image(s)", type="primary"):
            results = []
            progress = st.progress(0, text="Starting...")
            for i, f in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files), text=f"Analysing {f.name}...")
                try:
                    img = Image.open(f)
                    prob, is_fire, ms = run_inference(session, img, threshold)
                    results.append({"name": f.name, "prob": prob, "is_fire": is_fire, "ms": ms})
                    st.session_state.history.insert(0, {"name": f.name, "prob": prob, "is_fire": is_fire})
                except Exception as e:
                    results.append({"name": f.name, "prob": None, "is_fire": None, "ms": None, "error": str(e)})
            progress.empty()

            # Summary stats
            valid = [r for r in results if r["prob"] is not None]
            fire_count = sum(1 for r in valid if r["is_fire"])

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Total Images</div>
                  <div class="metric-value" style="color:#c9d1d9">{len(results)}</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Fire Detected</div>
                  <div class="metric-value" style="color:#ff4500">{fire_count}</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Clear</div>
                  <div class="metric-value" style="color:#22c55e">{len(valid)-fire_count}</div>
                </div>""", unsafe_allow_html=True)
            with s4:
                avg_prob = np.mean([r["prob"] for r in valid]) if valid else 0
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Avg Probability</div>
                  <div class="metric-value" style="color:#58a6ff">{avg_prob:.1%}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Results grid
            for r in sorted(results, key=lambda x: x.get("prob") or 0, reverse=True):
                if r.get("error"):
                    st.error(f"**{r['name']}** — Error: {r['error']}")
                    continue
                label = "FIRE" if r["is_fire"] else "SAFE"
                badge_cls = "hist-badge-fire" if r["is_fire"] else "hist-badge-safe"
                st.markdown(f"""
                <div class="hist-row">
                  <div class="hist-label" style="font-size:0.9rem;font-weight:500">{r['name']}</div>
                  <span class="{badge_cls}">{label}</span>
                  <div class="hist-prob" style="font-size:0.88rem">{r['prob']:.1%}</div>
                  <div style="color:#8b949e;font-size:0.78rem;min-width:60px;text-align:right">{r['ms']:.0f} ms</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
          border:2px dashed #30363d;border-radius:12px;
          padding:80px 20px;text-align:center;color:#8b949e;
        ">
          <div style="font-size:3rem;margin-bottom:10px">📁</div>
          <div style="font-weight:500;font-size:1.05rem">Drop multiple images above</div>
          <div style="font-size:0.82rem;margin-top:6px">Sorted by risk level, highest first</div>
        </div>
        """, unsafe_allow_html=True)
