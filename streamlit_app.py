"""
🎓 Student Attention Monitor — Streamlit Cloud (Real-Time Video)
Uses streamlit-webrtc for live camera + supports image/video upload.
Improvements v2:
  - Mirrored (selfie-view) live camera
  - Annotated frame overlaid on live feed in real-time
  - HUD overlay (attention rate bar burned into video)
  - Auto-refreshing stats panel (2s loop while live)
  - Lower default API interval (10 frames) for more responsiveness
  - Per-eye confidence now shown in face cards
  - b64_to_arr helper avoids double PIL conversion
"""
import base64, time, io, threading
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

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="Student Attention Monitor",
                   page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════
st.markdown("""<style>
.main-header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
  padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.2rem;text-align:center}
.main-header h1{color:#e2e8f0;font-size:1.9rem;margin:0}
.main-header p{color:#94a3b8;margin:.3rem 0 0;font-size:.9rem}
.mcard{background:#1e293b;border-radius:10px;padding:.9rem 1.1rem;
  border-left:4px solid #4f46e5;margin-bottom:.5rem}
.mcard.g{border-left-color:#22c55e}.mcard.r{border-left-color:#ef4444}
.mcard.y{border-left-color:#f59e0b}
.mval{font-size:1.9rem;font-weight:700;color:#e2e8f0}
.mlbl{font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.fcard{background:#1e293b;border-radius:8px;padding:.7rem 1rem;margin-bottom:.4rem}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block;margin-right:6px}
.dg{background:#22c55e}.dy{background:#f59e0b}.dr{background:#ef4444}.dgr{background:#64748b}
.sbanner{text-align:center;padding:.55rem;border-radius:8px;font-weight:700;
  font-size:1rem;margin-bottom:.5rem}
.att{background:#14532d;color:#86efac}.dis{background:#450a0a;color:#fca5a5}
.par{background:#451a03;color:#fcd34d}.unk{background:#1e293b;color:#94a3b8}
div[data-testid="stSidebar"]{background:#0f172a}
.live-badge{display:inline-block;background:#ef4444;color:white;border-radius:20px;
  padding:2px 10px;font-size:.75rem;font-weight:700;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════
DEFAULTS = {
    "history":    deque(maxlen=180),
    "stats":      dict(total=0, attentive=0, distracted=0, partial=0),
    "api_url":    "https://xxxx-xx.ngrok-free.app",
    "last_resp":  None,
    "last_frame": None,
    "lock":       threading.Lock(),
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
def arr_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()

def pil_to_b64(p: Image.Image) -> str:
    buf = io.BytesIO()
    p.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_pil(b: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b)))

def b64_to_arr(b: str) -> np.ndarray:
    """Base64 JPEG/PNG → BGR numpy array (no double PIL conversion)."""
    data = np.frombuffer(base64.b64decode(b), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def call_api(b64: str) -> dict:
    url = st.session_state.api_url.rstrip("/")
    try:
        r = requests.post(f"{url}/predict", json={"image": b64}, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Timeout (>8s)"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect — check ngrok URL"}
    except Exception as e:
        return {"error": str(e)}

def health():
    try:
        r = requests.get(f"{st.session_state.api_url.rstrip('/')}/health", timeout=5)
        return r.json() if r.ok else None
    except:
        return None

def emoji(a):
    return {"ATTENTIVE": "✅", "PARTIAL": "⚠️", "DISTRACTED": "❌"}.get(a, "❓")

def update_stats(resp: dict):
    if not resp or "error" in resp:
        return
    s = st.session_state.stats
    s["total"]      += resp.get("total_faces", 0)
    s["attentive"]  += resp.get("attentive_count", 0)
    s["distracted"] += resp.get("distracted_count", 0)
    s["partial"]    += resp.get("partial_count", 0)
    st.session_state.history.append({
        "time":           datetime.now().strftime("%H:%M:%S"),
        "attention_rate": resp.get("attention_rate", 0),
        "attentive":      resp.get("attentive_count", 0),
        "distracted":     resp.get("distracted_count", 0),
        "total":          resp.get("total_faces", 0),
    })
    st.session_state.last_resp = resp

def banner_html(resp: dict) -> str:
    rate = resp.get("attention_rate", 0)
    tot  = resp.get("total_faces", 0)
    att  = resp.get("attentive_count", 0)
    if tot == 0:
        return '<div class="sbanner unk">❓ No faces detected</div>'
    cls = "att" if rate >= 0.7 else ("par" if rate >= 0.4 else "dis")
    ico = "✅" if rate >= 0.7 else ("⚠️" if rate >= 0.4 else "❌")
    lbl = "ATTENTIVE" if rate >= 0.7 else ("Partial Attention" if rate >= 0.4 else "DISTRACTED")
    return f'<div class="sbanner {cls}">{ico} {lbl} — {att}/{tot} ({rate*100:.0f}%)</div>'

# ══════════════════════════════════════════════════════════
#  RENDER HELPERS
# ══════════════════════════════════════════════════════════
def draw_metrics():
    s    = st.session_state.stats
    rate = s["attentive"] / max(s["total"], 1)
    rc   = "g" if rate >= 0.7 else ("y" if rate >= 0.4 else "r")
    c1, c2, c3, c4 = st.columns(4)
    for col, clr, val, lbl in [
        (c1, "g", s["attentive"],     "✅ Attentive"),
        (c2, "r", s["distracted"],    "❌ Distracted"),
        (c3, "y", s["partial"],       "⚠️ Partial"),
        (c4, rc,  f"{rate*100:.0f}%", "📊 Attention Rate"),
    ]:
        col.markdown(
            f'<div class="mcard {clr}"><div class="mval">{val}</div>'
            f'<div class="mlbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

def draw_faces(ph, faces):
    if not faces:
        ph.markdown("*No faces detected*")
        return
    html = ""
    for f in faces:
        dot = {"ATTENTIVE": "dg", "PARTIAL": "dy", "DISTRACTED": "dr"}.get(
            f.get("attention", ""), "dgr")
        le  = (f.get("left_eye")  or {}).get("direction", "N/A")
        re  = (f.get("right_eye") or {}).get("direction", "N/A")
        lc  = (f.get("left_eye")  or {}).get("confidence", 0)
        rc  = (f.get("right_eye") or {}).get("confidence", 0)
        crop  = f.get("crop_type", "")
        cicon = {"iris": "🎯", "contour": "🔲", "none": "❓"}.get(crop, "")
        meth  = f.get("method", "")
        micon = {"agree": "🤝", "disagree_best": "⚖️",
                 "single_eye": "👁", "none": "—"}.get(meth, "")
        html += (
            f'<div class="fcard"><span class="dot {dot}"></span>'
            f'<span style="color:#e2e8f0"><b>Face #{f["face_id"]}</b> '
            f'{emoji(f["attention"])} {f["attention"]}<br>'
            f'<small style="color:#94a3b8">'
            f'Gaze: <b>{f["direction"]}</b> ({f["confidence"]*100:.0f}%) '
            f'| L: {le} ({lc*100:.0f}%) &nbsp; R: {re} ({rc*100:.0f}%)<br>'
            f'Crop: {cicon} {crop} &nbsp;|&nbsp; Fusion: {micon} {meth}'
            f'</small></span></div>'
        )
    ph.markdown(html, unsafe_allow_html=True)

def draw_chart(ph, threshold):
    hist = list(st.session_state.history)
    if len(hist) < 2:
        ph.caption("Chart appears after a few frames...")
        return
    df  = pd.DataFrame(hist)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["attention_rate"] * 100,
        mode="lines", fill="tozeroy",
        line=dict(color="#4f46e5", width=2),
        fillcolor="rgba(79,70,229,0.15)"
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Alert ({threshold}%)")
    fig.update_layout(
        height=175, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 105], gridcolor="#334155", color="#94a3b8"),
        xaxis=dict(showgrid=False, color="#94a3b8"),
        font_color="#e2e8f0", showlegend=False
    )
    ph.plotly_chart(fig, use_container_width=True)

def draw_pie(ph):
    s     = st.session_state.stats
    total = s["attentive"] + s["distracted"] + s["partial"]
    if total == 0:
        ph.caption("Pie appears after analysis...")
        return
    fig = px.pie(
        values=[s["attentive"], s["partial"], s["distracted"]],
        names=["Attentive", "Partial", "Distracted"],
        color_discrete_sequence=["#22c55e", "#f59e0b", "#ef4444"],
        hole=0.55
    )
    fig.update_layout(
        height=190, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        showlegend=True, legend=dict(orientation="h", y=-0.15)
    )
    ph.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  HUD OVERLAY  — burns live attention info into video frame
# ══════════════════════════════════════════════════════════
def draw_hud(frame_bgr: np.ndarray, resp: dict | None) -> np.ndarray:
    """Overlay a semi-transparent attention HUD on the frame."""
    out  = frame_bgr.copy()
    h, w = out.shape[:2]

    if not resp or "error" in resp:
        # Minimal HUD when no result yet
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 34), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        cv2.putText(out, "Waiting for API...", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA)
        return out

    rate  = resp.get("attention_rate", 0)
    att   = resp.get("attentive_count", 0)
    tot   = resp.get("total_faces", 0)
    color = (0, 210, 0) if rate >= 0.7 else ((0, 190, 255) if rate >= 0.4 else (0, 0, 220))

    # Semi-transparent top bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    label = f"Attention: {rate*100:.0f}%  |  Faces: {att}/{tot}"
    cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, color,    2, cv2.LINE_AA)
    cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (240,240,240), 1, cv2.LINE_AA)

    # Bottom attention progress bar
    bar_h = 6
    cv2.rectangle(out, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    cv2.rectangle(out, (0, h - bar_h), (int(w * rate), h), color, -1)

    return out

# ══════════════════════════════════════════════════════════
#  WEBRTC VIDEO PROCESSOR  (real-time, mirrored, HUD)
# ══════════════════════════════════════════════════════════
class AttentionVideoProcessor:
    """
    Each webcam frame is:
    1. Flipped horizontally → natural selfie mirror view
    2. Every api_interval frames → sent to FastAPI in a background thread
    3. Last annotated frame from API is blended with HUD and returned
    """
    def __init__(self):
        self.frame_count  = 0
        self.api_interval = 10        # ~3×/sec at 30fps  (was 15)
        self.last_overlay = None      # last annotated BGR from API
        self.last_resp    = None
        self.lock         = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        # ── 1. Mirror for selfie / natural classroom view ─────────────
        img_bgr = cv2.flip(img_bgr, 1)

        self.frame_count += 1

        # ── 2. Fire API call in background thread ─────────────────────
        if self.frame_count % self.api_interval == 0:
            threading.Thread(
                target=self._call_api,
                args=(img_bgr.copy(),),
                daemon=True
            ).start()

        # ── 3. Compose output frame ───────────────────────────────────
        with self.lock:
            overlay = self.last_overlay
            resp    = self.last_resp

        if overlay is not None and overlay.shape == img_bgr.shape:
            # Use API-annotated frame (already has bounding boxes + labels)
            base = overlay
        else:
            base = img_bgr

        out = draw_hud(base, resp)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def _call_api(self, img_bgr: np.ndarray):
        """Background thread: call API, store result."""
        b64  = arr_to_b64(img_bgr)
        resp = call_api(b64)
        if resp and "error" not in resp:
            update_stats(resp)
            ann_bgr = None
            if resp.get("annotated_frame"):
                ann_bgr = b64_to_arr(resp["annotated_frame"])
                if ann_bgr is not None:
                    # Mirror annotated frame to match selfie view
                    ann_bgr = cv2.flip(ann_bgr, 1)
            with self.lock:
                self.last_resp    = resp
                if ann_bgr is not None:
                    self.last_overlay = ann_bgr
            # Push to session state so sidebar stats panel can read it
            st.session_state.last_resp = resp

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_url = st.text_input(
        "🔗 Colab ngrok URL",
        value=st.session_state.api_url,
        placeholder="https://xxxx-xx.ngrok-free.app",
        help="Copy from Cell 9 in Colab"
    )
    st.session_state.api_url = api_url.strip()

    if st.button("🏥 Test Connection", use_container_width=True):
        with st.spinner("Checking..."):
            h = health()
        if h:
            st.success(f"✅ Online | {h.get('model','?')} | {h.get('device','?')}")
        else:
            st.error("❌ Cannot reach API")

    st.divider()
    alert_threshold = st.slider("⚠️ Alert below (%)", 10, 90, 50, 5)
    api_interval    = st.slider(
        "🔄 API call every N frames", 5, 60, 10, 5,
        help="Lower = more real-time but heavier. 10 ≈ 3×/sec at 30fps"
    )

    st.divider()
    st.markdown("### 📖 Direction → Attention")
    st.markdown("""| Direction | Status |
|-----------|--------|
| Top L/C/R | ✅ Attentive |
| Middle L/R | ⚠️ Partial |
| Bottom L/C/R | ❌ Distracted |""")

    st.divider()
    if st.button("🗑️ Reset Stats", use_container_width=True):
        st.session_state.stats     = dict(total=0, attentive=0, distracted=0, partial=0)
        st.session_state.history   = deque(maxlen=180)
        st.session_state.last_resp = None
        st.rerun()

# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""<div class="main-header">
  <h1>🎓 Student Attention Monitor</h1>
  <p>Real-time gaze detection — GazeNet8 (8-direction) via FastAPI + ngrok</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab_live, tab_snap, tab_img, tab_vid = st.tabs([
    "📹 Live Camera (Real-Time)",
    "📷 Camera Snapshot",
    "🖼️ Upload Image",
    "🎬 Upload Video",
])

# ─────────────────────────────────────────────────────────
#  TAB 1 — LIVE CAMERA (WebRTC real-time)
# ─────────────────────────────────────────────────────────
with tab_live:
    cl, cr = st.columns([3, 2], gap="large")

    with cl:
        st.markdown("### 📹 Live Camera Feed")
        st.info("🪞 Camera is **mirrored** (selfie view). Bounding boxes + HUD update automatically.")

        RTC_CONFIG = RTCConfiguration({"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]})

        ctx = webrtc_streamer(
            key="attention-detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=AttentionVideoProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False
            },
            async_processing=True,
        )

        # Sync sidebar slider to processor
        if ctx.video_processor:
            ctx.video_processor.api_interval = api_interval

        if ctx.state.playing:
            st.markdown(
                '<span class="live-badge">🔴 LIVE</span> &nbsp; Analyzing in real-time',
                unsafe_allow_html=True
            )
        else:
            st.caption("Click **START** above to begin live monitoring")

    with cr:
        st.markdown("### 📊 Live Session Statistics")

        # ── Placeholders for auto-refreshed stats ─────────────────────
        banner_ph  = st.empty()
        metrics_ph = st.empty()

        resp = st.session_state.last_resp
        if resp and "error" not in resp:
            banner_ph.markdown(banner_html(resp), unsafe_allow_html=True)
            if resp["total_faces"] > 0 and resp.get("attention_rate", 0) * 100 < alert_threshold:
                st.toast(f"⚠️ Attention below {alert_threshold}%!", icon="⚠️")

        with metrics_ph.container():
            draw_metrics()

        st.divider()
        st.markdown("### 👤 Per-Face Results")
        face_ph = st.empty()
        draw_faces(face_ph, resp.get("faces", []) if resp else [])

        st.divider()
        st.markdown("### 📈 Attention Over Time")
        chart_ph = st.empty()
        draw_chart(chart_ph, alert_threshold)

        st.divider()
        st.markdown("### 🎯 Gaze Distribution")
        pie_ph = st.empty()
        draw_pie(pie_ph)

        # ── Auto-refresh controls ──────────────────────────────────────
        col_btn, col_tog = st.columns(2)
        if col_btn.button("🔄 Refresh Stats", use_container_width=True):
            st.rerun()

        auto_refresh = col_tog.toggle("⚡ Auto Refresh", value=True)
        if auto_refresh and ctx.state.playing:
            # Rerun every 2 seconds to pull latest last_resp into the UI
            time.sleep(2)
            st.rerun()

# ─────────────────────────────────────────────────────────
#  TAB 2 — SNAPSHOT
# ─────────────────────────────────────────────────────────
with tab_snap:
    cl2, cr2 = st.columns([3, 2], gap="large")
    with cl2:
        st.markdown("### 📷 Camera Snapshot")
        st.info("📌 Take a photo → automatically sent to the API.")
        cam_img    = st.camera_input("", label_visibility="collapsed")
        banner_ph2 = st.empty()
        frame_ph2  = st.empty()
        if cam_img:
            pil_img = Image.open(cam_img)
            b64     = pil_to_b64(pil_img)
            with st.spinner("Analyzing..."):
                resp = call_api(b64)
            if resp and "error" not in resp:
                update_stats(resp)
                banner_ph2.markdown(banner_html(resp), unsafe_allow_html=True)
                if resp.get("annotated_frame"):
                    frame_ph2.image(b64_to_pil(resp["annotated_frame"]),
                                    caption="Annotated", use_container_width=True)
                rate = resp.get("attention_rate", 0)
                if resp["total_faces"] > 0 and rate * 100 < alert_threshold:
                    st.toast(f"⚠️ Attention below {alert_threshold}%!", icon="⚠️")
                with st.expander("🔍 Raw JSON"):
                    st.json(resp)
            else:
                st.error(f"❌ {resp.get('error','?') if resp else 'No response'}")
                st.caption("Check that Colab Cell 9 (ngrok) is active and URL is in sidebar.")
    with cr2:
        st.markdown("### 📊 Statistics")
        draw_metrics()
        st.divider()
        st.markdown("### 📈 Attention Over Time")
        draw_chart(st.empty(), alert_threshold)
        st.divider()
        st.markdown("### 🎯 Gaze Distribution")
        draw_pie(st.empty())
        resp = st.session_state.last_resp
        if resp and resp.get("faces"):
            st.divider()
            st.markdown("### 👤 Per-Face Results")
            draw_faces(st.empty(), resp["faces"])

# ─────────────────────────────────────────────────────────
#  TAB 3 — UPLOAD IMAGE
# ─────────────────────────────────────────────────────────
with tab_img:
    cl3, cr3 = st.columns([3, 2], gap="large")
    with cl3:
        st.markdown("### 🖼️ Upload Image")
        up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if up:
            fb  = np.frombuffer(up.read(), dtype=np.uint8)
            img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            pc, ac = st.columns(2)
            pc.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Original", use_container_width=True)
            aph = ac.empty()
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing..."):
                    resp = call_api(arr_to_b64(img))
                if resp and "error" not in resp:
                    update_stats(resp)
                    st.markdown(banner_html(resp), unsafe_allow_html=True)
                    if resp.get("annotated_frame"):
                        aph.image(b64_to_pil(resp["annotated_frame"]),
                                  caption="Annotated", use_container_width=True)
                    with st.expander("🔍 Raw JSON"):
                        st.json(resp)
                else:
                    st.error(f"❌ {resp.get('error','?') if resp else 'No response'}")
    with cr3:
        st.markdown("### 📊 Statistics")
        draw_metrics()
        st.divider()
        st.markdown("### 🎯 Gaze Distribution")
        draw_pie(st.empty())
        resp = st.session_state.last_resp
        if resp and resp.get("faces"):
            st.divider()
            st.markdown("### 👤 Per-Face Results")
            draw_faces(st.empty(), resp["faces"])

# ─────────────────────────────────────────────────────────
#  TAB 4 — UPLOAD VIDEO
# ─────────────────────────────────────────────────────────
with tab_vid:
    cl4, cr4 = st.columns([3, 2], gap="large")
    with cl4:
        st.markdown("### 🎬 Upload Video")
        upv   = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
        fstep = st.slider("Analyze every N frames", 1, 30, 5)
        if upv:
            tmp = f"/tmp/vid_{int(time.time())}.mp4"
            with open(tmp, "wb") as fh:
                fh.write(upv.read())
            if st.button("🔍 Analyze Video", use_container_width=True):
                cap      = cv2.VideoCapture(tmp)
                total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prog     = st.progress(0, text="Starting...")
                vf = st.empty()
                vb = st.empty()
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % fstep == 0:
                        resp = call_api(arr_to_b64(frame))
                        if resp and "error" not in resp:
                            update_stats(resp)
                            if resp.get("annotated_frame"):
                                vf.image(b64_to_pil(resp["annotated_frame"]),
                                         use_container_width=True)
                            vb.markdown(banner_html(resp), unsafe_allow_html=True)
                    prog.progress(
                        min(idx / max(total_fr, 1), 1.0),
                        text=f"Frame {idx}/{total_fr}"
                    )
                    idx += 1
                cap.release()
                st.success(f"✅ Done — {idx} frames processed.")
                hist = list(st.session_state.history)
                if hist:
                    csv = pd.DataFrame(hist).to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV Report", data=csv,
                        file_name=f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", use_container_width=True
                    )
    with cr4:
        st.markdown("### 📊 Statistics")
        draw_metrics()
        st.divider()
        st.markdown("### 📈 Attention Over Time")
        draw_chart(st.empty(), alert_threshold)
        st.divider()
        st.markdown("### 🎯 Gaze Distribution")
        draw_pie(st.empty())
