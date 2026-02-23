"""
🎓 Student Attention Monitor v4
States: ATTENTIVE · DISTRACTED · SLEEP
Backend: FP16 + TorchScript + fast preprocessing
"""
import base64, time, io, threading, queue
from datetime import datetime
from collections import deque

import requests
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(
    page_title="Attention Monitor",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html,body,[class*="css"]{font-family:'Syne',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:.8rem;}
.stApp{background:#080b12;}
section[data-testid="stSidebar"]{background:#0c0f18!important;border-right:1px solid #1a1f2e;}

/* top bar */
.topbar{display:flex;align-items:center;justify-content:space-between;
  background:#0c0f18;border:1px solid #1a1f2e;border-radius:12px;
  padding:.9rem 1.4rem;margin-bottom:1rem;}
.topbar-title{font-size:1.25rem;font-weight:800;color:#f0f4ff;letter-spacing:-.02em;}
.topbar-sub{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#3a4255;margin-top:.15rem;}
.topbar-states{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#3a4255;}

/* 3-state grid */
.sgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-bottom:.8rem;}
.scard{background:#0c0f18;border-radius:10px;padding:1rem 1.1rem;
  border:1px solid #1a1f2e;position:relative;overflow:hidden;}
.scard::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.scard.att::before{background:#22c55e;}
.scard.dis::before{background:#f97316;}
.scard.slp::before{background:#a78bfa;}
.sn{font-family:'JetBrains Mono',monospace;font-size:2.2rem;font-weight:700;line-height:1;margin-bottom:.2rem;}
.scard.att .sn{color:#22c55e;}
.scard.dis .sn{color:#f97316;}
.scard.slp .sn{color:#a78bfa;}
.sl{font-size:.65rem;color:#3a4255;text-transform:uppercase;letter-spacing:.1em;font-family:'JetBrains Mono',monospace;}

/* rate bar */
.rbar{background:#0c0f18;border:1px solid #1a1f2e;border-radius:10px;padding:.85rem 1.1rem;margin-bottom:.75rem;}
.rl{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#3a4255;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.45rem;}
.rt{height:7px;background:#1a1f2e;border-radius:4px;overflow:hidden;}
.rf{height:100%;border-radius:4px;transition:width .45s ease;}

/* banner */
.banner{text-align:center;padding:.55rem;border-radius:8px;
  font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.85rem;
  margin-bottom:.65rem;letter-spacing:.05em;}
.banner.att{background:#052e16;color:#22c55e;border:1px solid #14532d;}
.banner.dis{background:#1c0900;color:#f97316;border:1px solid #7c2d12;}
.banner.slp{background:#1e1038;color:#c4b5fd;border:1px solid #5b21b6;}
.banner.unk{background:#0c0f18;color:#3a4255;border:1px solid #1a1f2e;}

/* face rows */
.frow{display:flex;align-items:center;gap:.7rem;
  background:#0c0f18;border:1px solid #1a1f2e;border-radius:8px;
  padding:.55rem .9rem;margin-bottom:.35rem;}
.fid{font-family:'JetBrains Mono',monospace;color:#3a4255;font-size:.72rem;min-width:2rem;}
.fbadge{font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:700;
  padding:.2rem .6rem;border-radius:4px;white-space:nowrap;}
.fbadge.att{background:#052e16;color:#22c55e;border:1px solid #14532d;}
.fbadge.dis{background:#1c0900;color:#f97316;border:1px solid #7c2d12;}
.fbadge.slp{background:#1e1038;color:#c4b5fd;border:1px solid #5b21b6;}
.fbadge.unk{background:#111;color:#3a4255;border:1px solid #1a1f2e;}
.fconf{font-family:'JetBrains Mono',monospace;color:#3a4255;font-size:.7rem;margin-left:auto;}
.fcrop{font-family:'JetBrains Mono',monospace;color:#3a4255;font-size:.65rem;}

/* live badge */
.ldot{width:8px;height:8px;border-radius:50%;background:#ef4444;
  display:inline-block;margin-right:5px;animation:blink 1.2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}

/* sidebar labels */
.sblbl{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#3a4255;
  text-transform:uppercase;letter-spacing:.09em;margin-bottom:.3rem;}

/* inference ms badge */
.msbadge{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#3a4255;
  background:#0c0f18;border:1px solid #1a1f2e;border-radius:4px;
  padding:.1rem .4rem;display:inline-block;margin-left:.5rem;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════
SLEEP_SEC = 5.0   # eyes closed ≥ this → SLEEP

# Module-level timer — NOT in session_state (avoids thread issues)
_eye_closed_since: dict = {}

# ══════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════
if "history"    not in st.session_state: st.session_state.history    = deque(maxlen=300)
if "stats"      not in st.session_state: st.session_state.stats      = dict(total=0,attentive=0,distracted=0,sleep=0)
if "api_url"    not in st.session_state: st.session_state.api_url    = "https://xxxx-xx.ngrok-free.app"
if "last_resp"  not in st.session_state: st.session_state.last_resp  = None
if "_stats_key" not in st.session_state: st.session_state._stats_key = ""

# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
def b64_to_arr(b):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b),np.uint8),cv2.IMREAD_COLOR)

def b64_to_pil(b):
    return Image.open(io.BytesIO(base64.b64decode(b)))

def pil_to_b64(p):
    buf=io.BytesIO(); p.convert("RGB").save(buf,"JPEG",quality=80)
    return base64.b64encode(buf.getvalue()).decode()

def arr_to_b64(img):
    _,buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,80])
    return base64.b64encode(buf.tobytes()).decode()

def call_api(b64, url=None):
    url=(url or st.session_state.api_url).rstrip("/")
    try:
        r=requests.post(f"{url}/predict",json={"image":b64},timeout=10)
        r.raise_for_status(); return r.json()
    except requests.exceptions.Timeout:         return {"error":"Timeout (>10s)"}
    except requests.exceptions.ConnectionError: return {"error":"Cannot connect — check ngrok URL"}
    except Exception as e:                      return {"error":str(e)}

def health_check():
    try:
        r=requests.get(f"{st.session_state.api_url.rstrip('/')}/health",timeout=5)
        return r.json() if r.ok else None
    except: return None

# ── 3-state rules (frontend) ──────────────────────────────────────────────
def _is_closed(f):
    return (f.get("attention")=="SLEEP" or
            (f.get("left_eye") is None and
             f.get("right_eye") is None and
             f.get("crop_type","")=="none"))

def enrich(resp):
    """Apply SLEEP timer + recompute counts. Never touches session_state."""
    if not resp or "error" in resp: return resp
    now=time.time(); out=[]
    for f in resp.get("faces",[]):
        f=dict(f); fid=f.get("face_id",0)
        # Backend already marks SLEEP — trust it; add frontend timer as fallback
        if f.get("attention")=="SLEEP":
            out.append(f); continue
        if _is_closed(f):
            if fid not in _eye_closed_since: _eye_closed_since[fid]=now
            if now-_eye_closed_since[fid]>=SLEEP_SEC:
                f["attention"]="SLEEP"
        else:
            _eye_closed_since.pop(fid,None)
        out.append(f)
    # Cleanup stale
    active={f.get("face_id") for f in resp.get("faces",[])}
    for k in list(_eye_closed_since):
        if k not in active: _eye_closed_since.pop(k)
    total=len(out)
    att=sum(1 for f in out if f.get("attention")=="ATTENTIVE")
    dis=sum(1 for f in out if f.get("attention")=="DISTRACTED")
    slp=sum(1 for f in out if f.get("attention")=="SLEEP")
    rate=round(att/total,3) if total else 0.0
    return {**resp,"faces":out,"total_faces":total,
            "attentive_count":att,"distracted_count":dis,
            "sleep_count":slp,"attention_rate":rate}

def update_stats(resp):
    if not resp or "error" in resp: return
    key=f"{resp.get('attention_rate',0)}_{resp.get('total_faces',0)}_{datetime.now().strftime('%H:%M:%S')}"
    if st.session_state._stats_key==key: return
    st.session_state._stats_key=key
    s=st.session_state.stats
    s["total"]      += resp.get("total_faces",0)
    s["attentive"]  += resp.get("attentive_count",0)
    s["distracted"] += resp.get("distracted_count",0)
    s["sleep"]      += resp.get("sleep_count",0)
    st.session_state.history.append({
        "time":           datetime.now().strftime("%H:%M:%S"),
        "attention_rate": resp.get("attention_rate",0),
        "attentive":      resp.get("attentive_count",0),
        "distracted":     resp.get("distracted_count",0),
        "sleep":          resp.get("sleep_count",0),
        "total":          resp.get("total_faces",0),
    })
    st.session_state.last_resp=resp

# ══════════════════════════════════════════════════════════
#  RENDER COMPONENTS
# ══════════════════════════════════════════════════════════
def _cls(a): return {"ATTENTIVE":"att","DISTRACTED":"dis","SLEEP":"slp"}.get(a,"unk")
def _ico(a): return {"ATTENTIVE":"✓","DISTRACTED":"!","SLEEP":"zz"}.get(a,"?")

def render_banner(resp):
    if not resp or "error" in resp:
        st.markdown('<div class="banner unk">— waiting for analysis —</div>',unsafe_allow_html=True); return
    rate=resp.get("attention_rate",0); tot=resp.get("total_faces",0)
    att=resp.get("attentive_count",0); slp=resp.get("sleep_count",0)
    ms=resp.get("inference_ms",0)
    ms_tag=f'<span class="msbadge">{ms:.0f}ms</span>' if ms else ""
    if tot==0:
        st.markdown('<div class="banner unk">❓ No faces detected</div>',unsafe_allow_html=True); return
    if slp>0:   cls,ico,lbl="slp","zz",f"{slp} SLEEPING"
    elif rate>=0.7: cls,ico,lbl="att","✓",f"{att}/{tot} ATTENTIVE  {rate*100:.0f}%"
    elif rate>=0.4: cls,ico,lbl="dis","~",f"Partial — {att}/{tot}  {rate*100:.0f}%"
    else:           cls,ico,lbl="dis","✗",f"DISTRACTED — {att}/{tot}  {rate*100:.0f}%"
    st.markdown(f'<div class="banner {cls}">{ico}  {lbl}{ms_tag}</div>',unsafe_allow_html=True)

def render_cards():
    s=st.session_state.stats; total=max(s["total"],1)
    rate=s["attentive"]/total*100
    color="#22c55e" if rate>=70 else ("#f97316" if rate>=40 else "#ef4444")
    st.markdown(f"""
<div class="sgrid">
  <div class="scard att"><div class="sn">{s['attentive']}</div><div class="sl">✓ Attentive</div></div>
  <div class="scard dis"><div class="sn">{s['distracted']}</div><div class="sl">! Distracted</div></div>
  <div class="scard slp"><div class="sn">{s['sleep']}</div><div class="sl">zz Sleep</div></div>
</div>
<div class="rbar">
  <div class="rl">Attention Rate — {rate:.0f}%</div>
  <div class="rt"><div class="rf" style="width:{min(rate,100):.1f}%;background:{color}"></div></div>
</div>""",unsafe_allow_html=True)

def render_faces(faces):
    if not faces:
        st.markdown('<p style="color:#3a4255;font-size:.8rem;font-family:JetBrains Mono">No faces detected</p>',
                    unsafe_allow_html=True); return
    html=""
    for f in faces:
        a    = f.get("attention","UNKNOWN")
        c    = _cls(a); ico = _ico(a)
        conf = f.get("confidence", 0)
        # State label mapping
        lbl  = {"ATTENTIVE":"Attentive","DISTRACTED":"Distracted","SLEEP":"Sleeping"}.get(a, a)
        html += f'''<div class="frow">
  <span class="fid">#{f['face_id']}</span>
  <span class="fbadge {c}">{ico}&nbsp;{lbl}</span>
  <span class="fconf">{conf*100:.0f}%</span>
</div>'''
    st.markdown(html, unsafe_allow_html=True)

def render_chart(key, threshold):
    hist=list(st.session_state.history)
    if len(hist)<2: st.markdown('<p style="color:#3a4255;font-size:.75rem;font-family:JetBrains Mono">Chart after a few readings...</p>',unsafe_allow_html=True); return
    df=pd.DataFrame(hist)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(df))),y=df["attention_rate"]*100,
        mode="lines",fill="tozeroy",name="Attentive",
        line=dict(color="#22c55e",width=2),fillcolor="rgba(34,197,94,.1)"))
    if "sleep" in df.columns and df["sleep"].sum()>0:
        slp_pct=(df["sleep"]/df["total"].clip(lower=1))*100
        fig.add_trace(go.Scatter(x=list(range(len(df))),y=slp_pct,
            mode="lines",fill="tozeroy",name="Sleep",
            line=dict(color="#a78bfa",width=1.5,dash="dot"),
            fillcolor="rgba(167,139,250,.07)"))
    fig.add_hline(y=threshold,line_dash="dash",line_color="#ef4444",
                  annotation_text=f"Alert {threshold}%",annotation_font_color="#ef4444")
    fig.update_layout(height=175,margin=dict(l=0,r=0,t=8,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,105],gridcolor="#1a1f2e",color="#3a4255",
                   tickfont=dict(family="JetBrains Mono",size=9)),
        xaxis=dict(showgrid=False,color="#3a4255"),
        font=dict(color="#f0f4ff",family="JetBrains Mono",size=9),
        legend=dict(orientation="h",y=1.1,font=dict(size=9)),showlegend=True)
    st.plotly_chart(fig,use_container_width=True,key=key)

# ══════════════════════════════════════════════════════════
#  HUD — burned into video
# ══════════════════════════════════════════════════════════
def draw_hud(frame, resp):
    out=frame.copy(); h,w=out.shape[:2]
    ov=out.copy(); cv2.rectangle(ov,(0,0),(w,42),(8,11,18),-1)
    cv2.addWeighted(ov,0.8,out,0.2,0,out)
    if not resp or "error" in resp:
        cv2.putText(out,"connecting to API...",(12,27),cv2.FONT_HERSHEY_SIMPLEX,0.55,(58,66,85),1,cv2.LINE_AA)
        return out
    rate=resp.get("attention_rate",0); att=resp.get("attentive_count",0)
    slp=resp.get("sleep_count",0); tot=resp.get("total_faces",0)
    ms=resp.get("inference_ms",0)
    if slp>0:     state,color=f"SLEEP x{slp}",(160,60,200)
    elif rate>=.7: state,color="ATTENTIVE",(0,200,80)
    elif rate>=.4: state,color="PARTIAL",(0,160,240)
    else:          state,color="DISTRACTED",(0,80,230)
    txt=f"{state}  {att}/{tot}  {rate*100:.0f}%"
    if ms: txt+=f"  {ms:.0f}ms"
    cv2.putText(out,txt,(12,27),cv2.FONT_HERSHEY_DUPLEX,0.6,color,2,cv2.LINE_AA)
    cv2.putText(out,txt,(12,27),cv2.FONT_HERSHEY_DUPLEX,0.6,(220,225,235),1,cv2.LINE_AA)
    bh=5; cv2.rectangle(out,(0,h-bh),(w,h),(18,22,34),-1)
    cv2.rectangle(out,(0,h-bh),(int(w*rate),h),color,-1)
    return out

# ══════════════════════════════════════════════════════════
#  WEBRTC PROCESSOR
# ══════════════════════════════════════════════════════════
class AttentionVideoProcessor:
    _shared_resp: dict={}
    _shared_lock=threading.Lock()

    def __init__(self):
        self.api_interval=10; self.api_url=""
        self.last_overlay=None; self.last_resp=None; self.last_error=""
        self.lock=threading.Lock()
        self._frame_q=queue.Queue(maxsize=1)
        self._frame_count=0; self._stop_evt=threading.Event()
        threading.Thread(target=self._worker,daemon=True).start()

    def recv(self,frame):
        img=frame.to_ndarray(format="bgr24")
        self._frame_count+=1
        if self._frame_count%self.api_interval==0:
            try: self._frame_q.put_nowait(img.copy())
            except queue.Full: pass
        with self.lock:
            overlay=self.last_overlay; resp=self.last_resp
        if overlay is not None:
            h,w=img.shape[:2]
            if overlay.shape[:2]!=(h,w):
                overlay=cv2.resize(overlay,(w,h),interpolation=cv2.INTER_LINEAR)
            base=overlay
        else: base=img
        return av.VideoFrame.from_ndarray(draw_hud(base,resp),format="bgr24")

    def _worker(self):
        while not self._stop_evt.is_set():
            try: img=self._frame_q.get(timeout=1.0)
            except queue.Empty: continue
            url=self.api_url
            if not url or "xxxx" in url: continue
            self._call_api(img,url)

    def _call_api(self,img,url):
        try:
            _,buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,80])
            b64=base64.b64encode(buf.tobytes()).decode()
            try:
                r=requests.post(f"{url.rstrip('/')}/predict",json={"image":b64},timeout=10)
                r.raise_for_status(); resp=r.json()
            except requests.exceptions.Timeout: resp={"error":"Timeout"}
            except requests.exceptions.ConnectionError: resp={"error":"Cannot connect"}
            except Exception as e: resp={"error":str(e)}
            if resp and "error" not in resp:
                ann=b64_to_arr(resp["annotated_frame"]) if resp.get("annotated_frame") else None
                with self.lock:
                    self.last_resp=resp; self.last_error=""
                    if ann is not None: self.last_overlay=ann
                with AttentionVideoProcessor._shared_lock:
                    AttentionVideoProcessor._shared_resp=resp
            else:
                err=(resp or {}).get("error","Unknown")
                with self.lock: self.last_error=err
                with AttentionVideoProcessor._shared_lock:
                    AttentionVideoProcessor._shared_resp={"error":err}
        except Exception as e:
            with self.lock: self.last_error=str(e)

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sblbl">API Connection</p>',unsafe_allow_html=True)
    api_url=st.text_input("url",value=st.session_state.api_url,
                           label_visibility="collapsed",
                           placeholder="https://xxxx-xx.ngrok-free.app")
    st.session_state.api_url=api_url.strip()

    c1,c2=st.columns(2)
    if c1.button("Test",use_container_width=True):
        with st.spinner("..."):
            h=health_check()
        if h:
            fp="FP16✓" if h.get("fp16") else "FP32"
            st.success(f"Online · {fp} · v{h.get('version','?')}")
        else: st.error("Offline")

    st.divider()
    st.markdown('<p class="sblbl">Settings</p>',unsafe_allow_html=True)
    alert_threshold=st.slider("Alert below (%)",10,90,50,5)
    api_interval=st.slider("Analyze every N frames",5,60,10,5,
                            help="Lower = more real-time, heavier load")

    st.divider()
    st.markdown("""<p class="sblbl">State Reference</p>
<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;line-height:2">
<span style="color:#22c55e">✓ ATTENTIVE</span> — looking at screen<br>
<span style="color:#f97316">! DISTRACTED</span> — looking away<br>
<span style="color:#a78bfa">zz SLEEP</span> — eyes closed ≥5s
</div>""",unsafe_allow_html=True)

    st.divider()
    if st.button("Reset Stats",use_container_width=True):
        st.session_state.stats      = dict(total=0,attentive=0,distracted=0,sleep=0)
        st.session_state.history    = deque(maxlen=300)
        st.session_state.last_resp  = None
        _eye_closed_since.clear()
        AttentionVideoProcessor._shared_resp={}
        st.rerun()

# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""<div class="topbar">
  <div>
    <div class="topbar-title">🎓 Student Attention Monitor</div>
    <div class="topbar-sub">GazeNet8 · FP16 TorchScript · MediaPipe Iris · 3-State</div>
  </div>
  <div class="topbar-states">ATTENTIVE &nbsp;·&nbsp; DISTRACTED &nbsp;·&nbsp; SLEEP</div>
</div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab_live,tab_snap,tab_img,tab_vid=st.tabs([
    "📹 Live Camera","📷 Snapshot","🖼️ Image","🎬 Video"
])

# ─────────────────────────────────────────────────────────
#  TAB 1 — LIVE
# ─────────────────────────────────────────────────────────
with tab_live:
    cl,cr=st.columns([3,2],gap="large")
    with cl:
        st.markdown("#### 📹 Live Feed")
        ctx=webrtc_streamer(
            key="att-live",mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers":[
                {"urls":["stun:stun.l.google.com:19302"]},
                {"urls":["stun:stun1.l.google.com:19302"]},
            ]}),
            video_processor_factory=AttentionVideoProcessor,
            media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480}},"audio":False},
            async_processing=False,
        )
        if ctx.video_processor:
            ctx.video_processor.api_url=st.session_state.api_url
            ctx.video_processor.api_interval=api_interval
        if ctx.state.playing:
            st.markdown('<span class="ldot"></span><span style="color:#3a4255;font-family:\'JetBrains Mono\',monospace;font-size:.75rem">LIVE</span>',unsafe_allow_html=True)
            if ctx.video_processor:
                with ctx.video_processor.lock: err=ctx.video_processor.last_error
                if err: st.error(f"API: {err}")
        else:
            st.caption("Click START to begin")

    with cr:
        st.markdown("#### Session Stats")
        # Pull from mailbox → enrich → update (all in main thread)
        with AttentionVideoProcessor._shared_lock:
            shared=dict(AttentionVideoProcessor._shared_resp)
        if shared and "error" not in shared:
            shared=enrich(shared); update_stats(shared)

        resp=st.session_state.last_resp
        render_banner(resp)
        render_cards()

        if resp and "error" not in resp and resp.get("total_faces",0)>0:
            if resp.get("attention_rate",0)*100<alert_threshold:
                st.toast(f"⚠️ Attention below {alert_threshold}%!",icon="⚠️")

        st.divider()
        st.markdown("#### Faces")
        render_faces(resp.get("faces",[]) if resp else [])
        st.divider()
        st.markdown("#### Attention Over Time")
        render_chart("chart_live",alert_threshold)

        col_r,col_a=st.columns(2)
        if col_r.button("Refresh",use_container_width=True): st.rerun()
        auto=col_a.toggle("Auto",value=True)
        if auto and ctx.state.playing:
            time.sleep(2); st.rerun()

# ─────────────────────────────────────────────────────────
#  TAB 2 — SNAPSHOT
# ─────────────────────────────────────────────────────────
with tab_snap:
    cl2,cr2=st.columns([3,2],gap="large")
    with cl2:
        st.markdown("#### 📷 Camera Snapshot")
        cam=st.camera_input("",label_visibility="collapsed")
        ph_b=st.empty(); ph_f=st.empty()
        if cam:
            with st.spinner("Analyzing..."):
                resp=call_api(pil_to_b64(Image.open(cam)))
            if resp and "error" not in resp:
                resp=enrich(resp); update_stats(resp)
                with ph_b: render_banner(resp)
                if resp.get("annotated_frame"):
                    ph_f.image(b64_to_pil(resp["annotated_frame"]),use_container_width=True)
                with st.expander("JSON"): st.json(resp)
            else:
                st.error(f"❌ {(resp or {}).get('error','No response')}")
    with cr2:
        st.markdown("#### Stats"); render_cards()
        st.divider(); render_chart("chart_snap",alert_threshold)
        resp=st.session_state.last_resp
        if resp and resp.get("faces"):
            st.divider(); st.markdown("#### Faces"); render_faces(resp["faces"])

# ─────────────────────────────────────────────────────────
#  TAB 3 — IMAGE
# ─────────────────────────────────────────────────────────
with tab_img:
    cl3,cr3=st.columns([3,2],gap="large")
    with cl3:
        st.markdown("#### 🖼️ Upload Image")
        up=st.file_uploader("Image",type=["jpg","jpeg","png"],label_visibility="collapsed")
        if up:
            fb=np.frombuffer(up.read(),np.uint8); img=cv2.imdecode(fb,cv2.IMREAD_COLOR)
            pc,ac=st.columns(2)
            pc.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),caption="Original",use_container_width=True)
            aph=ac.empty()
            if st.button("Analyze",use_container_width=True):
                with st.spinner("..."):
                    resp=call_api(arr_to_b64(img))
                if resp and "error" not in resp:
                    resp=enrich(resp); update_stats(resp)
                    render_banner(resp)
                    if resp.get("annotated_frame"):
                        aph.image(b64_to_pil(resp["annotated_frame"]),caption="Annotated",use_container_width=True)
                    render_faces(resp.get("faces",[]))
                    with st.expander("JSON"): st.json(resp)
                else: st.error(f"❌ {(resp or {}).get('error','No response')}")
    with cr3:
        st.markdown("#### Stats"); render_cards()
        resp=st.session_state.last_resp
        if resp and resp.get("faces"):
            st.divider(); render_faces(resp["faces"])

# ─────────────────────────────────────────────────────────
#  TAB 4 — VIDEO
# ─────────────────────────────────────────────────────────
with tab_vid:
    cl4,cr4=st.columns([3,2],gap="large")
    with cl4:
        st.markdown("#### 🎬 Upload Video")
        upv=st.file_uploader("Video",type=["mp4","avi","mov"],label_visibility="collapsed")
        fstep=st.slider("Analyze every N frames",1,30,5)
        if upv:
            tmp=f"/tmp/vid_{int(time.time())}.mp4"
            with open(tmp,"wb") as fh: fh.write(upv.read())
            if st.button("Analyze Video",use_container_width=True):
                cap=cv2.VideoCapture(tmp); total_fr=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prog=st.progress(0,text="Starting..."); vf=st.empty(); vb=st.empty(); idx=0
                while True:
                    ret,frame=cap.read()
                    if not ret: break
                    if idx%fstep==0:
                        resp=call_api(arr_to_b64(frame))
                        if resp and "error" not in resp:
                            resp=enrich(resp); update_stats(resp)
                            if resp.get("annotated_frame"):
                                vf.image(b64_to_pil(resp["annotated_frame"]),use_container_width=True)
                            with vb: render_banner(resp)
                    prog.progress(min(idx/max(total_fr,1),1.0),text=f"Frame {idx}/{total_fr}")
                    idx+=1
                cap.release(); st.success(f"Done — {idx} frames")
                hist=list(st.session_state.history)
                if hist:
                    st.download_button("Download CSV",data=pd.DataFrame(hist).to_csv(index=False),
                        file_name=f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",use_container_width=True)
    with cr4:
        st.markdown("#### Stats"); render_cards()
        st.divider(); render_chart("chart_vid",alert_threshold)
