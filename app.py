# ============================================================
# app.py — MAIN ENTRY POINT (The Streamlit Web App)
# ============================================================
# This is the heart of the entire application.
# It wires all the utility modules together and provides
# the web-based user interface via Streamlit.
#
# HOW TO RUN:
#   source .venv/bin/activate && streamlit run app.py --server.port 8501
#
# HIGH-LEVEL FLOW:
#   1. App starts → load AI models once (MTCNN + FaceNet)
#   2. Admin uploads thief photos → detect + embed faces → store embeddings
#   3. Webcam starts → for each frame:
#        a. Detect faces using MTCNN          (detectors.py)
#        b. Embed each face                   (embedder.py)
#        c. Compare with stored thief embeds  (compare.py)
#        d. If match → draw red box + alarm   (alerts.py)
# ============================================================

import streamlit as st          # Streamlit — builds the web UI with Python
import numpy as np              # NumPy — numerical operations (used under the hood)
import cv2                      # OpenCV — capture webcam frames and draw boxes/text
from PIL import Image           # PIL — convert OpenCV frames to PIL for MTCNN
import time                     # time.sleep() to yield control between frames

# Import utility modules — each handles one step of the pipeline
from utils.detectors import get_mtcnn, detect_faces      # Step 1: Detect faces
from utils.embedder import get_embedder, compute_embedding  # Step 2: Embed faces
from utils.compare import is_match, cosine_similarity    # Step 3: Compare embeddings
from utils.alerts import play_siren_js                   # Step 4: Play alarm


# ============================================================
# PAGE SETUP
# ============================================================
import base64

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

st.set_page_config(page_title="AI Security System", layout="wide")
set_background('assets/background.png')
st.title("🔥 AI Security System: Thief Detection & Alarm Alert")


# ============================================================
# SESSION STATE — Streamlit's "Memory" Between Frames
# ============================================================
# Streamlit reruns the entire script on every interaction.
# st.session_state persists data across reruns so we don't
# reload heavy AI models on every frame (would be very slow).

if "thief_embeddings" not in st.session_state:
    # List to store 512-d face vectors for each uploaded thief photo
    st.session_state.thief_embeddings = []

if "embedder" not in st.session_state:
    # Load FaceNet model once at startup — converts faces to embeddings
    st.session_state.embedder = get_embedder()

if "mtcnn" not in st.session_state:
    # Load MTCNN face detector once at startup — finds faces in images
    st.session_state.mtcnn = get_mtcnn()

if "last_alarm_time" not in st.session_state:
    # Track the last time the alarm was played to avoid "stuttering"
    st.session_state.last_alarm_time = 0


# ============================================================
# SIDEBAR — Admin / Guard Control Panel
# ============================================================
with st.sidebar:
    st.header("Admin / Guard Panel")

    # Slider to control how strict face matching is
    # Lower threshold   = stricter (faces must be very similar to match)
    # Higher threshold  = looser (more faces will trigger the alarm)
    threshold = st.slider("Match threshold (cosine distance)", 0.1, 1.0, 0.6, 0.01)
    st.write("Lower is stricter (faces must be very similar).")

    # File uploader — admin uploads one or more photos of known thieves
    uploaded_files = st.file_uploader(
        "Upload thief images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        # Process each uploaded thief image
        for uf in uploaded_files:
            # Open the uploaded file as a PIL Image in RGB format
            image = Image.open(uf).convert("RGB")

            # STEP 1: Detect faces in the uploaded photo
            faces = detect_faces(st.session_state.mtcnn, image)

            if len(faces) == 0:
                # No face found in this image — warn the admin
                st.warning(f"No face detected in {uf.name}.")
                continue

            # Use only the first detected face (in case there are multiple)
            face_img, _ = faces[0]

            # STEP 2: Compute the 512-d embedding for this thief's face
            emb = compute_embedding(st.session_state.embedder, face_img)

            # Store the embedding — this is the "memory" of what the thief looks like
            st.session_state.thief_embeddings.append(emb)

        st.success(f"Stored {len(st.session_state.thief_embeddings)} thief embeddings.")


# ============================================================
# MAIN AREA — Two Tabs: Surveillance + Debug
# ============================================================
# st.tabs() returns a list of tab context managers
mode = st.tabs(["Surveillance", "Debug/Info"])


# ============================================================
# TAB 1: REAL-TIME SURVEILLANCE
# ============================================================
with mode[0]:
    st.subheader("Real-Time Surveillance")
    st.write("Open your webcam, detect faces, and compare with thief embeddings.")

    # Checkbox to start/stop the webcam
    run = st.checkbox("Start webcam")
    source = st.selectbox("Video source", ["Webcam"], index=0)

    # Placeholder containers — updated in-place on every frame
    frame_window = st.empty()   # Displays the live video frame
    info_box = st.empty()       # Shows status messages (monitoring / thief detected)
    alarm_box = st.empty()      # Hidden container for injecting alarm HTML

    cap = None  # OpenCV video capture object

    if run:
        # Open the default webcam (device index 0)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam.")
            run = False  # Stop the loop if camera fails

    # --------------------------------------------------------
    # MAIN WEBCAM LOOP — runs until "Start webcam" is unchecked
    # --------------------------------------------------------
    while run:
        # Capture one frame from the webcam
        ret, frame = cap.read()
        if not ret:
            # Frame read failed (camera disconnected, etc.)
            st.warning("Failed to read frame.")
            break

        # Convert frame from BGR (OpenCV default) → RGB → PIL Image
        # MTCNN and FaceNet expect RGB PIL Images, not OpenCV BGR arrays
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # STEP 1: Detect all faces in this frame
        faces = detect_faces(st.session_state.mtcnn, pil)

        alerts = []   # Collect alert messages for detected thieves
        boxes = []    # Collect bounding boxes + whether each face is a thief

        # STEP 2 + 3: For each detected face, embed + compare
        for face_img, box in faces:
            # Convert this face to a 512-d embedding vector
            emb = compute_embedding(st.session_state.embedder, face_img)

            # Compare this face against every stored thief embedding
            match_scores = []
            for thief_emb in st.session_state.thief_embeddings:
                # cosine_similarity returns a DISTANCE (lower = more similar)
                sim = cosine_similarity(emb, thief_emb)
                match_scores.append(sim)

            # A face is a THIEF if ANY stored embedding is below the threshold distance
            is_detected = any(s < threshold for s in match_scores) if match_scores else False

            # Record this face's bounding box and whether it's a thief
            boxes.append((box, is_detected))

            if is_detected:
                alerts.append("THIEF DETECTED ⚠️")

        # --------------------------------------------------------
        # STEP 4: Draw bounding boxes on the original frame
        # --------------------------------------------------------
        for (x1, y1, x2, y2), detected in boxes:
            # Red box for thieves, green box for unknown persons
            color = (255, 0, 0) if detected else (0, 255, 0)

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label the face as "THIEF" or "PERSON"
            label = "THIEF" if detected else "PERSON"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the annotated frame in the Streamlit UI
        frame_window.image(frame, channels="BGR")

        # STEP 4 (continued): Alert if any thief was detected
        if alerts:
            info_box.error("🚨 Thief Detected! Take Action!")
            
            # Cooldown logic: only play sound if 2 seconds have passed since last play
            current_time = time.time()
            if current_time - st.session_state.last_alarm_time > 2.0:
                audio_html = play_siren_js()
                with alarm_box:
                    st.components.v1.html(audio_html, height=0)
                st.session_state.last_alarm_time = current_time
        else:
            info_box.info("Monitoring...")
            alarm_box.empty()

        # Small sleep to yield CPU — prevents the loop from consuming 100% CPU
        # and allows Streamlit to process any UI interactions (e.g., unchecking the box)
        time.sleep(0.01)

    # Release the webcam when the loop ends (user unchecked "Start webcam")
    if cap is not None:
        cap.release()


# ============================================================
# TAB 2: DEBUG / INFO
# ============================================================
with mode[1]:
    st.subheader("Debug / Info")
    # Show how many thief face embeddings are currently stored in memory
    st.write("Embeddings stored:", len(st.session_state.thief_embeddings))