"""
Student Attention Monitor - Streamlit Cloud App
"""
import base64, time, io
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

st.set_page_config(page_title="Student Attention Monitor", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main-header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
  padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;text-align:center}
.main-header h1{color:#e2e8f0;font-size:2rem;margin:0}
.main-header p{color:#94a3b8;margin:.3rem 0 0}
.mcard{background:#1e293b;border-radius:10px;padding:1rem 1.2rem;
  border-left:4px solid #4f46e5;margin-bottom:.5rem}
.mcard.g{border-left-color:#22c55e}.mcard.r{border-left-color:#ef4444}
.mcard.y{border-left-color:#f59e0b}
.mval{font-size:2rem;font-weight:700;color:#e2e8f0}
.mlbl{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.fcard{background:#1e293b;border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem}
.dot{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:6px}
.dg{background:#22c55e}.dy{background:#f59e0b}.dr{background:#ef4444}.dgr{background:#64748b}
.sbanner{text-align:center;padding:.6rem;border-radius:8px;font-weight:700;
  font-size:1.1rem;margin-bottom:.5rem}
.att{background:#14532d;color:#86efac}.dis{background:#450a0a;color:#fca5a5}
.par{background:#451a03;color:#fcd34d}.unk{background:#1e293b;color:#94a3b8}
div[data-testid="stSidebar"]{background:#0f172a}
</style>""", unsafe_allow_html=True)

# session state
for k,v in [("history",deque(maxlen=120)),("stats",dict(total=0,attentive=0,distracted=0,partial=0)),
            ("api_url","https://xxxx-xx.ngrok-free.app"),("last_resp",None)]:
    if k not in st.session_state: st.session_state[k]=v

# helpers
def pil_to_b64(p):
    buf=io.BytesIO(); p.convert("RGB").save(buf,format="JPEG",quality=82)
    return base64.b64encode(buf.getvalue()).decode()
def arr_to_b64(img):
    _,buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,82])
    return base64.b64encode(buf.tobytes()).decode()
def b64_to_pil(b): return Image.open(io.BytesIO(base64.b64decode(b)))
def call_api(b64):
    url=st.session_state.api_url.rstrip("/")
    try:
        r=requests.post(f"{url}/predict",json={"image":b64},timeout=15)
        r.raise_for_status(); return r.json()
    except requests.exceptions.Timeout: return {"error":"Timed out — is Colab still running?"}
    except requests.exceptions.ConnectionError: return {"error":"Cannot connect — check the ngrok URL."}
    except Exception as e: return {"error":str(e)}
def health():
    try:
        r=requests.get(f"{st.session_state.api_url.rstrip('/')}/health",timeout=6)
        return r.json() if r.ok else None
    except: return None
def emoji(a): return {"ATTENTIVE":"✅","PARTIAL":"⚠️","DISTRACTED":"❌"}.get(a,"❓")
def update(resp):
    if not resp or "error" in resp: return
    s=st.session_state.stats
    s["total"]+=resp.get("total_faces",0); s["attentive"]+=resp.get("attentive_count",0)
    s["distracted"]+=resp.get("distracted_count",0); s["partial"]+=resp.get("partial_count",0)
    st.session_state.history.append({"time":datetime.now().strftime("%H:%M:%S"),
        "attention_rate":resp.get("attention_rate",0),"attentive":resp.get("attentive_count",0),
        "distracted":resp.get("distracted_count",0),"total":resp.get("total_faces",0)})
    st.session_state.last_resp=resp
def banner(resp):
    rate=resp.get("attention_rate",0); tot=resp.get("total_faces",0); att=resp.get("attentive_count",0)
    if tot==0: return '<div class="sbanner unk">❓ No faces detected</div>'
    cls="att" if rate>=0.7 else("par" if rate>=0.4 else "dis")
    ico="✅" if rate>=0.7 else("⚠️" if rate>=0.4 else "❌")
    lbl="ATTENTIVE" if rate>=0.7 else("Partial Attention" if rate>=0.4 else "DISTRACTED")
    return f'<div class="sbanner {cls}">{ico} {lbl} — {att}/{tot} ({rate*100:.0f}%)</div>'

def draw_metrics(alert_threshold):
    s=st.session_state.stats; rate=s["attentive"]/max(s["total"],1)
    rc="g" if rate>=0.7 else("y" if rate>=0.4 else "r")
    c1,c2,c3,c4=st.columns(4)
    for col,clr,val,lbl in [(c1,"g",s["attentive"],"✅ Attentive"),(c2,"r",s["distracted"],"❌ Distracted"),
                             (c3,"y",s["partial"],"⚠️ Partial"),(c4,rc,f"{rate*100:.0f}%","📊 Attention Rate")]:
        col.markdown(f'<div class="mcard {clr}"><div class="mval">{val}</div>'
                     f'<div class="mlbl">{lbl}</div></div>',unsafe_allow_html=True)

def draw_faces(ph, faces):
    if not faces: ph.markdown("*No faces detected*"); return
    html=""
    for f in faces:
        dot={"ATTENTIVE":"dg","PARTIAL":"dy","DISTRACTED":"dr"}.get(f.get("attention",""),"dgr")
        le=(f.get("left_eye") or {}).get("direction","N/A"); re=(f.get("right_eye") or {}).get("direction","N/A")
        html+=(f'<div class="fcard"><span class="dot {dot}"></span>'
               f'<span style="color:#e2e8f0"><b>Face #{f["face_id"]}</b> {emoji(f["attention"])} {f["attention"]}<br>'
               f'<small style="color:#94a3b8">Gaze:<b>{f["direction"]}</b>({f["confidence"]*100:.0f}%) L:{le} R:{re}</small></span></div>')
    ph.markdown(html,unsafe_allow_html=True)

def draw_chart(ph, threshold):
    hist=list(st.session_state.history)
    if len(hist)<2: ph.caption("Chart appears after a few frames..."); return
    df=pd.DataFrame(hist)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(df))),y=df["attention_rate"]*100,mode="lines",
        fill="tozeroy",line=dict(color="#4f46e5",width=2),fillcolor="rgba(79,70,229,0.15)"))
    fig.add_hline(y=threshold,line_dash="dash",line_color="#ef4444",annotation_text=f"Alert({threshold}%)")
    fig.update_layout(height=180,margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,105],gridcolor="#334155",color="#94a3b8"),
        xaxis=dict(showgrid=False,color="#94a3b8"),font_color="#e2e8f0",showlegend=False)
    ph.plotly_chart(fig,use_container_width=True)

def draw_pie(ph):
    s=st.session_state.stats; total=s["attentive"]+s["distracted"]+s["partial"]
    if total==0: ph.caption("Pie appears after analysis..."); return
    fig=px.pie(values=[s["attentive"],s["partial"],s["distracted"]],
        names=["Attentive","Partial","Distracted"],
        color_discrete_sequence=["#22c55e","#f59e0b","#ef4444"],hole=0.55)
    fig.update_layout(height=200,margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0",
        showlegend=True,legend=dict(orientation="h",y=-0.15))
    ph.plotly_chart(fig,use_container_width=True)

# sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_url=st.text_input("🔗 Colab ngrok URL",value=st.session_state.api_url,
        placeholder="https://xxxx-xx.ngrok-free.app",
        help="Copy from Cell 9 output in your Colab notebook")
    st.session_state.api_url=api_url.strip()
    if st.button("🏥 Test Connection",use_container_width=True):
        with st.spinner("Checking..."): h=health()
        if h: st.success(f"✅ Online | {h.get('model','?')} | {h.get('device','?')}")
        else: st.error("❌ Cannot reach API — check URL")
    st.divider()
    alert_threshold=st.slider("⚠️ Alert below (%)",10,90,50,5)
    st.divider()
    st.markdown("### 📖 Direction → Attention")
    st.markdown("""| Direction | Status |\n|-----------|--------|\n| Top L/C/R | ✅ Attentive |\n| Middle L/R | ⚠️ Partial |\n| Bottom L/C/R | ❌ Distracted |""")
    st.divider()
    if st.button("🗑️ Reset Stats",use_container_width=True):
        st.session_state.stats=dict(total=0,attentive=0,distracted=0,partial=0)
        st.session_state.history=deque(maxlen=120); st.session_state.last_resp=None; st.rerun()

# header
st.markdown("""<div class="main-header"><h1>🎓 Student Attention Monitor</h1>
<p>Real-time gaze detection — GazeNet8 (8-direction MobileNetV2) via FastAPI + ngrok</p></div>""",
unsafe_allow_html=True)

tab_cam,tab_img,tab_vid=st.tabs(["📷 Camera Snapshot","🖼️ Upload Image","🎬 Upload Video"])

# ── TAB 1: CAMERA (st.camera_input — Streamlit Cloud safe) ────────────
with tab_cam:
    cl,cr=st.columns([3,2],gap="large")
    with cl:
        st.markdown("### 📷 Camera Snapshot")
        st.info("📌 Click **Take Photo** — automatically sent to the API.")
        cam_img=st.camera_input("",label_visibility="collapsed")
        banner_ph=st.empty(); frame_ph=st.empty()
        if cam_img is not None:
            pil_img=Image.open(cam_img); b64=pil_to_b64(pil_img)
            with st.spinner("Analyzing..."): resp=call_api(b64)
            if resp and "error" not in resp:
                update(resp); banner_ph.markdown(banner(resp),unsafe_allow_html=True)
                if resp.get("annotated_frame"):
                    frame_ph.image(b64_to_pil(resp["annotated_frame"]),caption="Annotated",use_container_width=True)
                if resp["total_faces"]>0 and resp.get("attention_rate",0)*100<alert_threshold:
                    st.toast(f"⚠️ Attention below {alert_threshold}%!",icon="⚠️")
                with st.expander("🔍 Raw JSON"): st.json(resp)
            else:
                st.error(f"❌ {resp.get('error','?') if resp else 'No response'}")
                st.caption("Make sure Colab Cell 9 (ngrok) is running and the URL is in the sidebar.")
    with cr:
        st.markdown("### 📊 Session Statistics"); draw_metrics(alert_threshold)
        st.divider(); st.markdown("### 👤 Per-Face Results")
        fp1=st.empty(); resp=st.session_state.last_resp
        draw_faces(fp1,resp.get("faces",[]) if resp else [])
        st.divider(); st.markdown("### 📈 Attention Over Time"); draw_chart(st.empty(),alert_threshold)
        st.divider(); st.markdown("### 🎯 Gaze Distribution"); draw_pie(st.empty())

# ── TAB 2: UPLOAD IMAGE ────────────────────────────────────────────────
with tab_img:
    cl2,cr2=st.columns([3,2],gap="large")
    with cl2:
        st.markdown("### 🖼️ Upload Image")
        up=st.file_uploader("Choose an image",type=["jpg","jpeg","png"])
        if up:
            fb=np.frombuffer(up.read(),dtype=np.uint8); img=cv2.imdecode(fb,cv2.IMREAD_COLOR)
            pc,ac=st.columns(2)
            pc.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),caption="Original",use_container_width=True)
            aph=ac.empty()
            if st.button("🔍 Analyze Image",use_container_width=True):
                with st.spinner("Analyzing..."): resp=call_api(arr_to_b64(img))
                if resp and "error" not in resp:
                    update(resp); st.markdown(banner(resp),unsafe_allow_html=True)
                    if resp.get("annotated_frame"):
                        aph.image(b64_to_pil(resp["annotated_frame"]),caption="Annotated",use_container_width=True)
                    with st.expander("🔍 Raw JSON"): st.json(resp)
                else: st.error(f"❌ {resp.get('error','?') if resp else 'No response'}")
    with cr2:
        st.markdown("### 📊 Statistics"); draw_metrics(alert_threshold)
        st.divider(); st.markdown("### 🎯 Gaze Distribution"); draw_pie(st.empty())
        resp2=st.session_state.last_resp
        if resp2 and resp2.get("faces"):
            st.divider(); st.markdown("### 👤 Per-Face Results"); draw_faces(st.empty(),resp2["faces"])

# ── TAB 3: UPLOAD VIDEO ────────────────────────────────────────────────
with tab_vid:
    cl3,cr3=st.columns([3,2],gap="large")
    with cl3:
        st.markdown("### 🎬 Upload Video")
        upv=st.file_uploader("Choose a video",type=["mp4","avi","mov"])
        fstep=st.slider("Analyze every N frames",1,30,5)
        if upv:
            tmp=f"/tmp/vid_{int(time.time())}.mp4"
            with open(tmp,"wb") as fh: fh.write(upv.read())
            if st.button("🔍 Analyze Video",use_container_width=True):
                cap=cv2.VideoCapture(tmp); total_fr=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prog=st.progress(0,text="Starting..."); vf=st.empty(); vb=st.empty(); idx=0
                while True:
                    ret,frame=cap.read()
                    if not ret: break
                    if idx%fstep==0:
                        resp=call_api(arr_to_b64(frame))
                        if resp and "error" not in resp:
                            update(resp)
                            if resp.get("annotated_frame"): vf.image(b64_to_pil(resp["annotated_frame"]),use_container_width=True)
                            vb.markdown(banner(resp),unsafe_allow_html=True)
                    prog.progress(min(idx/max(total_fr,1),1.0),text=f"Frame {idx}/{total_fr}"); idx+=1
                cap.release(); st.success(f"✅ Done — {idx} frames processed.")
                hist=list(st.session_state.history)
                if hist:
                    csv=pd.DataFrame(hist).to_csv(index=False)
                    st.download_button("📥 Download CSV Report",data=csv,
                        file_name=f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",use_container_width=True)
    with cr3:
        st.markdown("### 📊 Statistics"); draw_metrics(alert_threshold)
        st.divider(); st.markdown("### 📈 Attention Over Time"); draw_chart(st.empty(),alert_threshold)
        st.divider(); st.markdown("### 🎯 Gaze Distribution"); draw_pie(st.empty())
