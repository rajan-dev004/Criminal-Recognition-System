# ============================================================
# app.py — MAIN ENTRY POINT (The Streamlit Web App)
# ============================================================
# Cloud-compatible surveillance app with two modes:
#   1. Live Video (WebRTC) - Real-time processing via browser stream
#   2. Snapshot (Standard Camera Input) - 100% reliable fallback working on all networks/devices
# ============================================================

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import base64
import queue
import threading

# streamlit-webrtc — cloud-safe real-time video streaming
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av  # PyAV — used by streamlit-webrtc to handle video frames

# Import utility modules
from utils.detectors import get_mtcnn, detect_faces
from utils.embedder import get_embedder, compute_embedding
from utils.compare import is_match, cosine_similarity
from utils.alerts import play_siren_js


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
# LOAD MODELS VIA CACHE (Thread-safe, avoids st.session_state)
# ============================================================

@st.cache_resource
def get_cached_models():
    """Load models once globally and cache them, making them accessible to any thread."""
    return get_mtcnn(), get_embedder()


# ============================================================
# SESSION STATE — Streamlit's "Memory" Between Reruns
# ============================================================

if "thief_embeddings" not in st.session_state:
    st.session_state.thief_embeddings = []

# Keep track of unique filenames we've already processed to prevent duplicate embeddings on rerun
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0

# Shared queue: VideoProcessor → Streamlit UI (for thief-detected events)
if "alert_queue" not in st.session_state:
    st.session_state.alert_queue = queue.Queue(maxsize=1)

# Shared threshold
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.6


# Load the models on the main Streamlit thread first
with st.spinner("Initializing AI Models..."):
    get_cached_models()


# ============================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================

class FaceRecognitionProcessor(VideoProcessorBase):
    """
    Runs in a background thread for every incoming WebRTC video frame.
    To avoid thread context errors, does NOT access st.session_state.
    Instead, attributes are set from the main thread.
    """

    def __init__(self):
        # Load heavy models once globally (cached)
        self.mtcnn, self.embedder = get_cached_models()
        
        # Placeholders to be updated by the main thread
        self.threshold = 0.6
        self.thief_embeddings = []
        self.alert_queue = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called by streamlit-webrtc for every incoming camera frame.
        """
        # Convert PyAV frame → numpy (BGR) → PIL (RGB) for MTCNN
        img_bgr = frame.to_ndarray(format="bgr24")
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        threshold = self.threshold
        thief_embeddings = self.thief_embeddings

        # STEP 1: Detect all faces in this frame
        faces = detect_faces(self.mtcnn, pil)

        thief_found = False

        # STEP 2 + 3: Embed each face and compare to stored thief embeddings
        for face_img, (x1, y1, x2, y2) in faces:
            emb = compute_embedding(self.embedder, face_img)

            # Compare against every stored thief embedding
            is_detected = False
            if thief_embeddings:
                distances = [cosine_similarity(emb, t) for t in thief_embeddings]
                is_detected = any(d < threshold for d in distances)

            # STEP 4: Draw colored bounding box on the original BGR frame
            color = (0, 0, 255) if is_detected else (0, 255, 0)  # Red = thief, Green = unknown
            label = "THIEF" if is_detected else "PERSON"

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_bgr, label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            if is_detected:
                thief_found = True

        # Push detection result to the shared queue if it exists
        if self.alert_queue is not None:
            try:
                self.alert_queue.put_nowait(thief_found)
            except queue.Full:
                pass

        # Convert annotated numpy array back to av.VideoFrame and return
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# ============================================================
# RTC CONFIGURATION — STUN/TURN servers for NAT traversal
# (Using public Google STUN servers)
# ============================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)


# ============================================================
# SIDEBAR — Admin / Guard Control Panel
# ============================================================

with st.sidebar:
    st.header("🛡️ Admin / Guard Panel")

    threshold = st.slider(
        "Match threshold (cosine distance)", 0.1, 1.0,
        st.session_state.threshold, 0.01
    )
    st.session_state.threshold = threshold
    st.caption("Lower = stricter match required to trigger alarm.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload thief images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_count = 0
        mtcnn, embedder = get_cached_models()
        for uf in uploaded_files:
            # Generate a unique key for the file to prevent duplicate processing on page reruns
            file_key = f"{uf.name}_{uf.size}"
            if file_key in st.session_state.processed_files:
                continue

            image = Image.open(uf).convert("RGB")
            faces = detect_faces(mtcnn, image)

            if len(faces) == 0:
                st.warning(f"No face detected in **{uf.name}**.")
                st.session_state.processed_files.add(file_key)
                continue

            face_img, _ = faces[0]
            emb = compute_embedding(embedder, face_img)
            st.session_state.thief_embeddings.append(emb)
            st.session_state.processed_files.add(file_key)
            new_count += 1

        if new_count > 0:
            st.success(f"✅ {new_count} new face(s) enrolled. Total stored: {len(st.session_state.thief_embeddings)}")

    if st.session_state.thief_embeddings:
        st.info(f"👤 {len(st.session_state.thief_embeddings)} thief face(s) in memory.")
        if st.button("🗑️ Clear all thief embeddings"):
            st.session_state.thief_embeddings = []
            st.session_state.processed_files = set()  # Reset processed files tracker
            st.rerun()
    else:
        st.warning("No thief photos uploaded yet.")


# ============================================================
# MAIN AREA — Two Tabs: Surveillance + Debug
# ============================================================

mode = st.tabs(["📹 Surveillance", "🔍 Debug / Info"])


# ============================================================
# TAB 1: REAL-TIME SURVEILLANCE
# ============================================================

with mode[0]:
    st.subheader("Real-Time Surveillance")

    if not st.session_state.thief_embeddings:
        st.info("ℹ️ Upload at least one thief photo in the sidebar to enable detection.")

    # Choose between Live WebRTC Stream and Native Snapshot Camera Input
    surveillance_type = st.radio(
        "Select Camera Mode:",
        ["Snapshot Mode (100% Reliable Fallback)", "Live Video Stream (WebRTC)"],
        help="Use Snapshot Mode if WebRTC fails to connect due to network/firewall restrictions."
    )

    # --------------------------------------------------------
    # MODE A: SNAPSHOT CAMERA INPUT (100% reliable, works everywhere)
    # --------------------------------------------------------
    if surveillance_type == "Snapshot Mode (100% Reliable Fallback)":
        st.write("Take a photo using your camera to scan for registered thieves.")
        
        # Native Streamlit camera input widget (does not use WebRTC)
        camera_img = st.camera_input("Surveillance Camera")

        if camera_img is not None:
            # Load the captured image
            pil_img = Image.open(camera_img).convert("RGB")
            
            # Load models
            mtcnn, embedder = get_cached_models()
            
            # Detect faces
            faces = detect_faces(mtcnn, pil_img)
            
            if len(faces) == 0:
                st.info("Monitoring… No faces detected in the snapshot.")
            else:
                # Convert PIL to BGR numpy array for drawing
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                thief_detected = False
                
                # Check each detected face
                for face_img, (x1, y1, x2, y2) in faces:
                    emb = compute_embedding(embedder, face_img)
                    
                    is_detected = False
                    if st.session_state.thief_embeddings:
                        distances = [cosine_similarity(emb, t) for t in st.session_state.thief_embeddings]
                        is_detected = any(d < st.session_state.threshold for d in distances)
                        
                    color = (0, 0, 255) if is_detected else (0, 255, 0)
                    label = "THIEF" if is_detected else "PERSON"
                    
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        img_bgr, label,
                        (x1, max(y1 - 15, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                    )
                    
                    if is_detected:
                        thief_detected = True
                
                # Display result
                st.image(img_bgr, channels="BGR", caption="Processed Snapshot Result")
                
                if thief_detected:
                    st.error("🚨 **THIEF DETECTED! Take Action Immediately!**")
                    
                    # Trigger the siren alert
                    audio_html = play_siren_js()
                    if audio_html:
                        st.components.v1.html(audio_html, height=0)
                else:
                    st.success("🟢 Scan Complete. No threats detected.")

    # --------------------------------------------------------
    # MODE B: LIVE VIDEO STREAM (WebRTC)
    # --------------------------------------------------------
    else:
        st.warning(
            "💡 **WebRTC Connection Tip:** If you see the error *'Connection taking longer than expected'*, "
            "it is due to a firewall or router policy blocking peer-to-peer UDP traffic on your network.\n\n"
            "👉 Switch to **Snapshot Mode** above to run the system with 100% reliability."
        )

        st.write("Click **START** below to activate your live video feed.")

        # Launch WebRTC streamer
        ctx = webrtc_streamer(
            key="criminal-recognition",
            video_processor_factory=FaceRecognitionProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Pass parameters to the background processor thread
        if ctx.video_processor:
            ctx.video_processor.threshold = st.session_state.threshold
            ctx.video_processor.thief_embeddings = st.session_state.thief_embeddings
            ctx.video_processor.alert_queue = st.session_state.alert_queue

        status_placeholder = st.empty()
        alarm_placeholder = st.empty()

        # Poll the alert queue and update UI
        if ctx.state.playing:
            alert_queue = st.session_state.alert_queue
            thief_found = False
            try:
                while True:
                    thief_found = alert_queue.get_nowait()
            except queue.Empty:
                pass

            if thief_found:
                status_placeholder.error("🚨 **THIEF DETECTED! Take Action Immediately!**")

                # Play alarm with cooldown
                current_time = time.time()
                if current_time - st.session_state.last_alarm_time > 2.0:
                    audio_html = play_siren_js()
                    if audio_html:
                        with alarm_placeholder:
                            st.components.v1.html(audio_html, height=0)
                    st.session_state.last_alarm_time = current_time
            else:
                status_placeholder.info("🟢 Monitoring… No threats detected.")
                alarm_placeholder.empty()
        else:
            status_placeholder.info("📷 Camera inactive. Click START to begin surveillance.")


# ============================================================
# TAB 2: DEBUG / INFO
# ============================================================

with mode[1]:
    st.subheader("Debug / Info")
    st.write("**Thief embeddings stored:**", len(st.session_state.thief_embeddings))
    st.write("**Current threshold:**", st.session_state.threshold)

    if st.session_state.thief_embeddings:
        st.write("**Embedding shapes:**",
                 [e.shape for e in st.session_state.thief_embeddings])