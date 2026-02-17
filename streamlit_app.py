"""
🎓 Student Attention Monitor — Streamlit App
Connects to the FastAPI backend running on Colab via ngrok.

Run locally:
    pip install streamlit requests opencv-python Pillow plotly pandas
    streamlit run streamlit_app.py
"""

import base64, time, io, json
from datetime import datetime
from collections import deque

import requests
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Attention Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e2e8f0; font-size: 2rem; margin: 0; }
    .main-header p  { color: #94a3b8; margin: 0.3rem 0 0; }

    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4f46e5;
        margin-bottom: 0.5rem;
    }
    .metric-card.green  { border-left-color: #22c55e; }
    .metric-card.red    { border-left-color: #ef4444; }
    .metric-card.yellow { border-left-color: #f59e0b; }
    .metric-card.blue   { border-left-color: #3b82f6; }

    .metric-val  { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
    .metric-label{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }

    .face-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .dot {
        width: 12px; height: 12px; border-radius: 50%;
        display: inline-block; margin-right: 6px;
    }
    .dot-green  { background: #22c55e; }
    .dot-yellow { background: #f59e0b; }
    .dot-red    { background: #ef4444; }
    .dot-gray   { background: #64748b; }

    .status-banner {
        text-align: center; padding: 0.5rem;
        border-radius: 8px; font-weight: 700;
        font-size: 1.1rem; margin-bottom: 0.5rem;
    }
    .banner-attentive  { background: #14532d; color: #86efac; }
    .banner-distracted { background: #450a0a; color: #fca5a5; }
    .banner-partial    { background: #451a03; color: #fcd34d; }
    .banner-unknown    { background: #1e293b; color: #94a3b8; }

    .api-status-ok  { color: #22c55e; font-weight: 600; }
    .api-status-err { color: #ef4444; font-weight: 600; }

    div[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=120)   # last 120 frames
if "running" not in st.session_state:
    st.session_state.running = False
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "attentive": 0, "distracted": 0, "partial": 0}

# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════
def img_to_b64(img: np.ndarray) -> str:
    """OpenCV BGR → base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def call_api(api_url: str, img_b64: str) -> dict | None:
    try:
        r = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json={"image": img_b64},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def check_health(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return r.json() if r.ok else None
    except:
        return None

def attention_color(att: str) -> str:
    return {"ATTENTIVE": "#22c55e", "PARTIAL": "#f59e0b",
            "DISTRACTED": "#ef4444", "UNKNOWN": "#64748b"}.get(att, "#64748b")

def attention_emoji(att: str) -> str:
    return {"ATTENTIVE": "✅", "PARTIAL": "⚠️",
            "DISTRACTED": "❌", "UNKNOWN": "❓"}.get(att, "❓")

# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_url = st.text_input(
        "🔗 API URL (ngrok)",
        value=st.session_state.get("api_url", "https://xxxx-xx-xx-xx-xx.ngrok-free.app"),
        help="Paste the ngrok URL from your Colab backend here",
    )
    st.session_state["api_url"] = api_url

    # Health check button
    if st.button("🏥 Check API Health"):
        h = check_health(api_url)
        if h:
            st.markdown(f'<span class="api-status-ok">✅ API Online</span>', unsafe_allow_html=True)
            st.json(h)
        else:
            st.markdown(f'<span class="api-status-err">❌ API Offline</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📷 Input Source")
    source = st.radio("Source", ["Webcam", "Upload Image", "Upload Video"],
                      label_visibility="collapsed")

    st.divider()
    st.markdown("### 🎯 Attention Thresholds")
    alert_threshold = st.slider(
        "Alert if attention rate below (%)",
        min_value=10, max_value=90, value=50, step=5,
    )

    st.divider()
    st.markdown("### ℹ️ Direction Map")
    st.markdown("""
| Direction | Status |
|-----------|--------|
| Top* | ✅ Attentive |
| Middle* | ⚠️ Partial |
| Bottom* | ❌ Distracted |
    """)

    st.divider()
    if st.button("🗑️ Reset Stats"):
        st.session_state.stats = {"total": 0, "attentive": 0, "distracted": 0, "partial": 0}
        st.session_state.history.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🎓 Student Attention Monitor</h1>
  <p>Real-time gaze detection powered by GazeNet8 (8-direction MobileNetV2)</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  LAYOUT — two columns
# ═══════════════════════════════════════════════════════════════════════
col_feed, col_stats = st.columns([3, 2], gap="large")

# ── RIGHT: Stats panel ────────────────────────────────────────────────
with col_stats:
    st.markdown("### 📊 Session Statistics")

    m1, m2 = st.columns(2)
    metric_attentive  = m1.empty()
    metric_distracted = m2.empty()
    m3, m4 = st.columns(2)
    metric_partial    = m3.empty()
    metric_rate       = m4.empty()

    st.divider()
    st.markdown("### 👤 Per-Face Results")
    face_list_placeholder = st.empty()

    st.divider()
    st.markdown("### 📈 Attention Over Time")
    chart_placeholder = st.empty()

    st.divider()
    st.markdown("### 🎯 Gaze Distribution")
    pie_placeholder = st.empty()

def render_metrics():
    s = st.session_state.stats
    total = s["total"] or 1
    rate  = s["attentive"] / total

    with metric_attentive:
        st.markdown(f"""
        <div class="metric-card green">
          <div class="metric-val">{s['attentive']}</div>
          <div class="metric-label">✅ Attentive</div>
        </div>""", unsafe_allow_html=True)

    with metric_distracted:
        st.markdown(f"""
        <div class="metric-card red">
          <div class="metric-val">{s['distracted']}</div>
          <div class="metric-label">❌ Distracted</div>
        </div>""", unsafe_allow_html=True)

    with metric_partial:
        st.markdown(f"""
        <div class="metric-card yellow">
          <div class="metric-val">{s['partial']}</div>
          <div class="metric-label">⚠️ Partial</div>
        </div>""", unsafe_allow_html=True)

    color_cls = "green" if rate >= 0.7 else ("yellow" if rate >= 0.4 else "red")
    with metric_rate:
        st.markdown(f"""
        <div class="metric-card {color_cls}">
          <div class="metric-val">{rate*100:.0f}%</div>
          <div class="metric-label">📊 Attention Rate</div>
        </div>""", unsafe_allow_html=True)

def render_face_list(faces):
    if not faces:
        face_list_placeholder.markdown("*No faces detected*")
        return
    html = ""
    for f in faces:
        dot = {"ATTENTIVE": "dot-green", "PARTIAL": "dot-yellow",
               "DISTRACTED": "dot-red"}.get(f["attention"], "dot-gray")
        emoji = attention_emoji(f["attention"])
        html += f"""
        <div class="face-card">
          <span class="dot {dot}"></span>
          <span style="color:#e2e8f0">
            <b>Face #{f['face_id']}</b> — {emoji} {f['attention']}
            <br><small style="color:#94a3b8">
              Gaze: {f['direction']} ({f['confidence']*100:.0f}%)
              &nbsp;|&nbsp;
              L: {f['left_eye']['direction'] if f.get('left_eye') else 'N/A'}
              &nbsp;
              R: {f['right_eye']['direction'] if f.get('right_eye') else 'N/A'}
            </small>
          </span>
        </div>"""
    face_list_placeholder.markdown(html, unsafe_allow_html=True)

def render_chart():
    if len(st.session_state.history) < 2:
        return
    df = pd.DataFrame(list(st.session_state.history))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["attention_rate"] * 100,
        mode="lines", fill="tozeroy",
        line=dict(color="#4f46e5", width=2),
        fillcolor="rgba(79,70,229,0.15)",
        name="Attention %",
    ))
    fig.add_hline(y=alert_threshold, line_dash="dash",
                  line_color="#ef4444", annotation_text=f"Alert ({alert_threshold}%)")
    fig.update_layout(
        height=180, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,105], gridcolor="#334155", color="#94a3b8"),
        xaxis=dict(showgrid=False, color="#94a3b8"),
        font_color="#e2e8f0", showlegend=False,
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

def render_pie():
    s = st.session_state.stats
    total = s["attentive"] + s["distracted"] + s["partial"]
    if total == 0:
        return
    fig = px.pie(
        values=[s["attentive"], s["partial"], s["distracted"]],
        names=["Attentive", "Partial", "Distracted"],
        color_discrete_sequence=["#22c55e", "#f59e0b", "#ef4444"],
        hole=0.55,
    )
    fig.update_layout(
        height=200, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    pie_placeholder.plotly_chart(fig, use_container_width=True)

def process_api_response(resp: dict):
    """Update session state from one API response."""
    if "error" in resp:
        return

    s  = st.session_state.stats
    at = resp.get("attentive_count", 0)
    di = resp.get("distracted_count", 0)
    pa = resp.get("partial_count", 0)
    s["total"]      += resp.get("total_faces", 0)
    s["attentive"]  += at
    s["distracted"] += di
    s["partial"]    += pa

    st.session_state.history.append({
        "time":           datetime.now().strftime("%H:%M:%S"),
        "attention_rate": resp.get("attention_rate", 0),
        "attentive":      at,
        "distracted":     di,
        "partial":        pa,
        "total":          resp.get("total_faces", 0),
    })

# ═══════════════════════════════════════════════════════════════════════
#  LEFT: Video feed
# ═══════════════════════════════════════════════════════════════════════
with col_feed:

    # ── WEBCAM MODE ───────────────────────────────────────────────────
    if source == "Webcam":
        st.markdown("### 🎥 Live Webcam Feed")

        cam_col1, cam_col2 = st.columns(2)
        start_btn = cam_col1.button("▶️ Start", use_container_width=True)
        stop_btn  = cam_col2.button("⏹️ Stop",  use_container_width=True)

        if start_btn:
            st.session_state.running = True
        if stop_btn:
            st.session_state.running = False

        frame_placeholder  = st.empty()
        banner_placeholder = st.empty()
        fps_placeholder    = st.empty()

        if st.session_state.running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam. Try 'Upload Image' mode instead.")
                st.session_state.running = False
            else:
                t_prev = time.time()
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    b64 = img_to_b64(frame)
                    resp = call_api(api_url, b64)

                    if resp and "error" not in resp:
                        process_api_response(resp)

                        # Annotated frame
                        if resp.get("annotated_frame"):
                            ann_img = b64_to_pil(resp["annotated_frame"])
                            frame_placeholder.image(ann_img, use_container_width=True)
                        else:
                            frame_placeholder.image(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                use_container_width=True,
                            )

                        # Attention banner
                        rate = resp.get("attention_rate", 0)
                        if resp["total_faces"] == 0:
                            banner_cls, banner_txt = "banner-unknown", "❓ No faces detected"
                        elif rate >= 0.7:
                            banner_cls, banner_txt = "banner-attentive", f"✅ Class is ATTENTIVE — {rate*100:.0f}%"
                        elif rate >= 0.4:
                            banner_cls, banner_txt = "banner-partial",   f"⚠️ Partial Attention — {rate*100:.0f}%"
                        else:
                            banner_cls, banner_txt = "banner-distracted", f"❌ Class is DISTRACTED — {rate*100:.0f}%"

                        banner_placeholder.markdown(
                            f'<div class="status-banner {banner_cls}">{banner_txt}</div>',
                            unsafe_allow_html=True,
                        )

                        # FPS
                        t_now = time.time()
                        fps_placeholder.caption(f"⚡ {1/(t_now - t_prev):.1f} FPS")
                        t_prev = t_now

                        # Update right-panel
                        render_metrics()
                        render_face_list(resp.get("faces", []))
                        render_chart()
                        render_pie()

                        # Alert
                        if (resp["total_faces"] > 0 and
                                rate * 100 < alert_threshold):
                            st.toast(f"⚠️ Attention dropped below {alert_threshold}%!", icon="⚠️")

                    else:
                        err = resp.get("error", "Unknown error") if resp else "No response"
                        frame_placeholder.error(f"API Error: {err}")

                cap.release()
        else:
            st.info("👆 Click **▶️ Start** to begin live monitoring")

    # ── UPLOAD IMAGE MODE ─────────────────────────────────────────────
    elif source == "Upload Image":
        st.markdown("### 🖼️ Upload Image")
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)

            if analyze_btn:
                with st.spinner("Analyzing..."):
                    b64  = img_to_b64(img)
                    resp = call_api(api_url, b64)

                if resp and "error" not in resp:
                    process_api_response(resp)

                    disp_col1, disp_col2 = st.columns(2)
                    disp_col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                    caption="Original", use_container_width=True)
                    if resp.get("annotated_frame"):
                        ann = b64_to_pil(resp["annotated_frame"])
                        disp_col2.image(ann, caption="Annotated", use_container_width=True)

                    # Summary
                    rate = resp.get("attention_rate", 0)
                    if resp["total_faces"] == 0:
                        st.warning("❓ No faces detected in this image.")
                    else:
                        att_cls = ("banner-attentive" if rate >= 0.7 else
                                   "banner-partial"   if rate >= 0.4 else
                                   "banner-distracted")
                        st.markdown(
                            f'<div class="status-banner {att_cls}">'
                            f'Attentive: {resp["attentive_count"]}/{resp["total_faces"]} '
                            f'({rate*100:.0f}%)</div>',
                            unsafe_allow_html=True,
                        )
                        render_metrics()
                        render_face_list(resp.get("faces", []))
                        render_pie()

                    # Raw JSON expander
                    with st.expander("🔍 Raw API Response"):
                        st.json(resp)
                else:
                    err = resp.get("error", "Unknown") if resp else "No response"
                    st.error(f"❌ API Error: {err}")
            else:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption="Preview", use_container_width=True)

    # ── UPLOAD VIDEO MODE ─────────────────────────────────────────────
    elif source == "Upload Video":
        st.markdown("### 🎬 Upload Video")
        uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
        frame_step   = st.slider("Analyze every N frames", 1, 30, 5)

        if uploaded_vid:
            tmp_path = f"/tmp/uploaded_video_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_vid.read())

            analyze_btn = st.button("🔍 Analyze Video", use_container_width=True)

            if analyze_btn:
                cap        = cv2.VideoCapture(tmp_path)
                total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress   = st.progress(0)
                v_frame_ph = st.empty()
                v_banner   = st.empty()
                frame_idx  = 0

                with st.spinner(f"Processing {total_fr} frames (every {frame_step})..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_idx % frame_step == 0:
                            b64  = img_to_b64(frame)
                            resp = call_api(api_url, b64)

                            if resp and "error" not in resp:
                                process_api_response(resp)

                                if resp.get("annotated_frame"):
                                    v_frame_ph.image(b64_to_pil(resp["annotated_frame"]),
                                                     use_container_width=True)

                                rate = resp.get("attention_rate", 0)
                                cls  = ("banner-attentive" if rate >= 0.7 else
                                        "banner-partial"   if rate >= 0.4 else
                                        "banner-distracted")
                                v_banner.markdown(
                                    f'<div class="status-banner {cls}">'
                                    f'Frame {frame_idx}/{total_fr} — '
                                    f'Attentive: {resp["attentive_count"]}/{resp["total_faces"]} '
                                    f'({rate*100:.0f}%)</div>',
                                    unsafe_allow_html=True,
                                )
                                render_metrics()
                                render_face_list(resp.get("faces", []))
                                render_chart()
                                render_pie()

                        progress.progress(min(frame_idx / max(total_fr, 1), 1.0))
                        frame_idx += 1

                cap.release()
                st.success("✅ Video analysis complete!")

                # Download report
                history_data = list(st.session_state.history)
                if history_data:
                    df = pd.DataFrame(history_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Attention Report (CSV)",
                        data=csv,
                        file_name=f"attention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

# ── Initial render of right panel ─────────────────────────────────────
render_metrics()
render_chart()
render_pie()
