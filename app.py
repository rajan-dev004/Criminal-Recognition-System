# ============================================================
# app.py — MAIN ENTRY POINT (The Streamlit Web App)
# ============================================================
# Cloud-aware surveillance app.
# Automatically detects if running locally on Mac or in Hugging Face:
#   - Local Mac: Uses standard, high-performance OpenCV cv2.VideoCapture(0)
#   - HF Spaces (Cloud): Uses streamlit-webrtc for real-time browser video
# ============================================================

import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import base64

# Import utility modules — each handles one step of the pipeline
from utils.detectors import get_mtcnn, detect_faces      # Step 1: Detect faces
from utils.embedder import get_embedder, compute_embedding  # Step 2: Embed faces
from utils.compare import is_match, cosine_similarity    # Step 3: Compare embeddings
from utils.alerts import play_siren_js                   # Step 4: Play alarm

# Detect environment
IS_CLOUD = "SPACE_ID" in os.environ or "HF_SPACE" in os.environ

# WebRTC imports — only needed on cloud (Hugging Face Spaces)
if IS_CLOUD:
    import threading
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


# ============================================================
# PAGE SETUP
# ============================================================

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    
    [data-testid="stSidebar"] {
        background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7));
    }
    
    /* Make content more readable against the background */
    .main .block-container {
        background: rgba(0, 0, 0, 0.6);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3, p, span, label {
        color: #ffffff !important;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_page_config(page_title="Criminal Recognition System", layout="wide")
set_background('assets/background.png')
st.title("🚨 Criminal Recognition System")


# ============================================================
# SESSION STATE — Streamlit's "Memory" Between Reruns
# ============================================================

if "target_embedding" not in st.session_state:
    st.session_state.target_embedding = None

if "target_file_key" not in st.session_state:
    st.session_state.target_file_key = None

if "embedder" not in st.session_state:
    with st.spinner("Loading FaceNet model..."):
        st.session_state.embedder = get_embedder()

if "mtcnn" not in st.session_state:
    with st.spinner("Loading MTCNN detector..."):
        st.session_state.mtcnn = get_mtcnn()

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0

# Thread-safe flag: set to True when WebRTC processor detects the criminal
if "criminal_detected_flag" not in st.session_state:
    st.session_state.criminal_detected_flag = False


# ============================================================
# SIDEBAR — Admin / Guard Control Panel
# ============================================================

with st.sidebar:
    st.header("🛡️ Admin / Guard Panel")

    threshold = st.slider("Match threshold (cosine distance)", 0.1, 1.0, 0.6, 0.01)
    st.caption("Lower = stricter match required to trigger alarm.")


# ============================================================
# WebRTC Video Processor — processes each frame in real-time
# ============================================================

class CriminalDetectionProcessor(VideoProcessorBase):
    """
    Runs on every video frame streamed via WebRTC.
    Detects faces, computes embeddings, compares against the target,
    and draws bounding boxes + labels directly on the video feed.
    """

    def __init__(self):
        self.target_embedding = None
        self.threshold = 0.6
        self.mtcnn = None
        self.embedder = None
        self._lock = threading.Lock()
        self.criminal_detected = False  # flag per-frame result
        self.session_id = None
        self.last_rerun_time = 0

    def _trigger_rerun(self):
        if self.session_id:
            try:
                from streamlit.runtime import get_instance
                runtime = get_instance()
                if runtime:
                    session_info = runtime._session_info_by_id.get(self.session_id)
                    if session_info and session_info.session:
                        session_info.session.request_rerun()
            except Exception:
                pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with self._lock:
            target_emb = self.target_embedding
            thresh = self.threshold
            mtcnn = self.mtcnn
            embedder = self.embedder

        if target_emb is None or mtcnn is None or embedder is None:
            # Models not ready or no target — pass through raw frame
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Convert BGR → RGB → PIL for MTCNN
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Detect faces in this frame
        try:
            faces = detect_faces(mtcnn, pil)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        detected_criminal = False

        for face_img, (x1, y1, x2, y2) in faces:
            try:
                emb = compute_embedding(embedder, face_img)
                sim = cosine_similarity(emb, target_emb)
                is_detected = sim < thresh
            except Exception:
                is_detected = False

            color = (0, 0, 255) if is_detected else (0, 255, 0)
            label = "CRIMINAL DETECTED" if is_detected else "PERSON"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img, label,
                (x1, max(y1 - 15, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            if is_detected:
                detected_criminal = True

        self.criminal_detected = detected_criminal

        if detected_criminal:
            current_time = time.time()
            if current_time - self.last_rerun_time > 2.0:
                self.last_rerun_time = current_time
                self._trigger_rerun()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# MAIN AREA — Two Tabs: Surveillance + Debug
# ============================================================

mode = st.tabs(["📹 Surveillance", "🔍 Debug / Info"])


# ============================================================
# TAB 1: REAL-TIME SURVEILLANCE
# ============================================================

with mode[0]:
    st.subheader("Real-Time Surveillance")

    # 1. UPLOAD OPTION
    st.write("### 1. Upload Target Image")
    uploaded_target_file = st.file_uploader(
        "Upload an image containing the target face to search for", 
        type=["jpg", "jpeg", "png"], 
        key="target_uploader"
    )

    if uploaded_target_file is not None:
        file_key = f"{uploaded_target_file.name}_{uploaded_target_file.size}"
        if st.session_state.target_file_key != file_key:
            # Process and embed the target face
            with st.spinner("Analyzing target image..."):
                image = Image.open(uploaded_target_file).convert("RGB")
                faces = detect_faces(st.session_state.mtcnn, image)
                
                if len(faces) == 0:
                    st.session_state.target_embedding = None
                    st.session_state.target_file_key = file_key
                    st.warning("⚠️ No face detected in the uploaded target image. Please upload a clear photo of a face.")
                else:
                    face_img, _ = faces[0]
                    st.session_state.target_embedding = compute_embedding(st.session_state.embedder, face_img)
                    st.session_state.target_file_key = file_key
                    st.success("🎯 Target facial data successfully loaded and enrolled.")
    else:
        st.session_state.target_embedding = None
        st.session_state.target_file_key = None

    # 2. SURVEILLANCE RUN CONTROL
    st.write("### 2. Start Surveillance")

    if st.session_state.target_embedding is None:
        st.info("👆 Upload a target image first, then start surveillance below.")
    else:
        # Toggle to start / stop surveillance
        run = st.toggle("Start Surveillance", value=False, key="surveillance_run_toggle")

        if run:
            # ── Cloud path: Real-time via WebRTC ──
            if IS_CLOUD:
                st.info("☁️ Running on Hugging Face Spaces — real-time detection via WebRTC.")

                # RTC Configuration — STUN and TURN servers for NAT traversal on Hugging Face Spaces
                RTC_CONFIGURATION = {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {
                            "urls": [
                                "turn:openrelay.metered.ca:80",
                                "turn:openrelay.metered.ca:443",
                                "turn:openrelay.metered.ca:443?transport=tcp"
                            ],
                            "username": "openrelayproject",
                            "credential": "openrelayproject"
                        }
                    ]
                }

                ctx = webrtc_streamer(
                    key="criminal-surveillance",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=CriminalDetectionProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

                # After the streamer is created, inject the target embedding + models
                # into the processor so it can run detection on each frame
                if ctx.video_processor:
                    ctx.video_processor.target_embedding = st.session_state.target_embedding
                    ctx.video_processor.threshold = threshold
                    ctx.video_processor.mtcnn = st.session_state.mtcnn
                    ctx.video_processor.embedder = st.session_state.embedder

                    # Pass the streamlit script context to the processor to trigger reruns
                    from streamlit.runtime.scriptrunner import get_script_run_ctx
                    ctx_script = get_script_run_ctx()
                    if ctx_script:
                        ctx.video_processor.session_id = ctx_script.session_id

                    # Check if the processor flagged a criminal detection
                    if ctx.state.playing and ctx.video_processor.criminal_detected:
                        st.error("🚨 **CRIMINAL DETECTED! Take Action Immediately!**")
                        current_time = time.time()
                        if current_time - st.session_state.last_alarm_time > 2.0:
                            audio_html = play_siren_js()
                            if audio_html:
                                st.components.v1.html(audio_html, height=0)
                            st.session_state.last_alarm_time = current_time

            # ── Local path: Real-time via OpenCV VideoCapture loop ──
            else:
                st.success("💻 Running locally. Live Webcam loop enabled.")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not access local webcam.")
                else:
                    # Placeholder containers — updated in-place on every frame
                    frame_window = st.empty()   # Displays the live video frame
                    info_box = st.empty()       # Shows status messages
                    alarm_box = st.empty()      # Hidden container for alarm HTML

                    while st.session_state.surveillance_run_toggle:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to read frame.")
                            break

                        # Convert frame from BGR → RGB → PIL Image for MTCNN
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)

                        # Detect all faces in this frame
                        faces = detect_faces(st.session_state.mtcnn, pil)

                        alerts = []
                        boxes = []

                        # For each detected face, embed + compare to target embedding
                        for face_img, box in faces:
                            emb = compute_embedding(st.session_state.embedder, face_img)
                            sim = cosine_similarity(emb, st.session_state.target_embedding)
                            is_detected = sim < threshold
                            boxes.append((box, is_detected))

                            if is_detected:
                                alerts.append("CRIMINAL DETECTED ⚠️")

                        # Draw bounding boxes
                        for (x1, y1, x2, y2), detected in boxes:
                            color = (0, 0, 255) if detected else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = "CRIMINAL DETECTED" if detected else "PERSON"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Display the annotated frame in the Streamlit UI
                        frame_window.image(frame, channels="BGR")

                        # Alert if matched
                        if alerts:
                            info_box.error("🚨 Criminal Detected! Take Action!")
                            
                            current_time = time.time()
                            if current_time - st.session_state.last_alarm_time > 2.0:
                                audio_html = play_siren_js()
                                with alarm_box:
                                    st.components.v1.html(audio_html, height=0)
                                st.session_state.last_alarm_time = current_time
                        else:
                            info_box.info("Monitoring...")
                            alarm_box.empty()

                        # Yield CPU
                        time.sleep(0.01)

                    cap.release()


# ============================================================
# TAB 2: DEBUG / INFO
# ============================================================

with mode[1]:
    st.subheader("Debug / Info")
    st.write("Target enrolled:", "Yes" if st.session_state.target_embedding is not None else "No")
    if st.session_state.target_embedding is not None:
        st.write("Target embedding shape:", st.session_state.target_embedding.shape)
    st.write("Environment:", "☁️ Cloud (Hugging Face)" if IS_CLOUD else "💻 Local Mac")
    st.write("Detection mode:", "WebRTC (real-time)" if IS_CLOUD else "OpenCV VideoCapture (real-time)")