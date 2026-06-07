# ============================================================
# app.py — MAIN ENTRY POINT (The Streamlit Web App)
# ============================================================
# Uses streamlit-webrtc for cloud-compatible live camera access
# (works on Hugging Face Spaces, unlike cv2.VideoCapture(0)).
#
# HOW TO RUN:
#   streamlit run app.py --server.port=7860 --server.address=0.0.0.0
#
# HIGH-LEVEL FLOW:
#   1. App starts → load AI models once (MTCNN + FaceNet)
#   2. Admin uploads thief photos → detect + embed faces → store embeddings
#   3. WebRTC stream → for each frame:
#        a. Detect faces using MTCNN          (detectors.py)
#        b. Embed each face                   (embedder.py)
#        c. Compare with stored thief embeds  (compare.py)
#        d. If match → draw red box + alarm   (alerts.py)
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
# (Using public STUN servers and Open Relay TURN servers for robust connection)
# ============================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            # STUN Servers
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]},
            
            # Open Relay TURN Servers (Relays WebRTC traffic through ports 80/443 to bypass strict firewalls/NATs)
            {
                "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turns:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            }
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
# TAB 1: REAL-TIME SURVEILLANCE (WebRTC)
# ============================================================

with mode[0]:
    st.subheader("Real-Time Surveillance")

    # Inform the user about iframe security restrictions on Hugging Face
    st.warning(
        "💡 **Hugging Face Iframe Security Tip:** If you see the error *'Connection taking longer than expected'*, "
        "it is because the browser blocks camera access inside Hugging Face's iframe wrapper.\n\n"
        "👉 Please open the app directly using the raw Hugging Face Space URL: "
        "**[https://rajan-2004-c-r-s.hf.space](https://rajan-2004-c-r-s.hf.space)**"
    )

    if not st.session_state.thief_embeddings:
        st.info("ℹ️ Upload at least one thief photo in the sidebar to enable detection.")

    st.write("Click **START** below to activate your camera. Detection runs automatically.")

    # Launch WebRTC streamer — this starts the camera in the browser
    # and calls FaceRecognitionProcessor.recv() for every frame
    ctx = webrtc_streamer(
        key="criminal-recognition",
        video_processor_factory=FaceRecognitionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Pass the threshold, embeddings, and queue from Streamlit thread to the background processor thread
    if ctx.video_processor:
        ctx.video_processor.threshold = st.session_state.threshold
        ctx.video_processor.thief_embeddings = st.session_state.thief_embeddings
        ctx.video_processor.alert_queue = st.session_state.alert_queue

    # Status / alarm area (updated based on messages from the processor)
    status_placeholder = st.empty()
    alarm_placeholder = st.empty()

    # Poll the alert queue and update UI when the stream is active
    if ctx.state.playing:
        alert_queue = st.session_state.alert_queue

        # Drain the queue and show latest result
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