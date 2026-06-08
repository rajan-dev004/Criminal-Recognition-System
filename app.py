# ============================================================
# app.py — MAIN ENTRY POINT (The Streamlit Web App)
# ============================================================
# Cloud-aware surveillance app.
# Automatically detects if running locally on Mac or in Hugging Face:
#   - Local Mac: Uses standard, high-performance OpenCV cv2.VideoCapture(0)
#   - HF Spaces (Cloud): Uses native st.camera_input (100% reliable fallback)
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

if "thief_embeddings" not in st.session_state:
    st.session_state.thief_embeddings = []

# Keep track of unique filenames we've already processed to prevent duplicate embeddings on rerun
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "embedder" not in st.session_state:
    with st.spinner("Loading FaceNet model..."):
        st.session_state.embedder = get_embedder()

if "mtcnn" not in st.session_state:
    with st.spinner("Loading MTCNN detector..."):
        st.session_state.mtcnn = get_mtcnn()

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0


# ============================================================
# SIDEBAR — Admin / Guard Control Panel
# ============================================================

with st.sidebar:
    st.header("🛡️ Admin / Guard Panel")

    threshold = st.slider("Match threshold (cosine distance)", 0.1, 1.0, 0.6, 0.01)
    st.caption("Lower = stricter match required to trigger alarm.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload thief images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        new_count = 0
        for uf in uploaded_files:
            # Generate a unique key for the file to prevent duplicate processing on page reruns
            file_key = f"{uf.name}_{uf.size}"
            if file_key in st.session_state.processed_files:
                continue

            # Open the uploaded file as a PIL Image in RGB format
            image = Image.open(uf).convert("RGB")

            # STEP 1: Detect faces in the uploaded photo
            faces = detect_faces(st.session_state.mtcnn, image)

            if len(faces) == 0:
                st.warning(f"No face detected in **{uf.name}**.")
                st.session_state.processed_files.add(file_key)
                continue

            # Use only the first detected face (in case there are multiple)
            face_img, _ = faces[0]

            # STEP 2: Compute the 512-d embedding for this thief's face
            emb = compute_embedding(st.session_state.embedder, face_img)

            # Store the embedding
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

    # Choose between Upload Photo and Live Webcam
    surveillance_source = st.radio(
        "Select Surveillance Source:",
        ["📁 Upload Photo", "📹 Live Webcam"],
        help="Upload an image to scan or start the live camera feed."
    )

    if surveillance_source == "📁 Upload Photo":
        # Upload photo of the scene/person to scan
        uploaded_surveillance_file = st.file_uploader(
            "Upload surveillance scene/person photo", type=["jpg", "jpeg", "png"], key="scene_uploader"
        )
        
        # Start Surveillance button
        if st.button("Start Surveillance", type="primary"):
            if uploaded_surveillance_file is not None:
                # Load the uploaded image
                pil_img = Image.open(uploaded_surveillance_file).convert("RGB")
                
                with st.spinner("Processing surveillance image..."):
                    # Detect faces
                    faces = detect_faces(st.session_state.mtcnn, pil_img)
                    
                    if len(faces) == 0:
                        st.warning("No faces detected in the uploaded photo.")
                    else:
                        # Convert PIL to BGR numpy array for drawing
                        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        thief_detected = False
                        
                        # Check each detected face
                        for face_img, (x1, y1, x2, y2) in faces:
                            emb = compute_embedding(st.session_state.embedder, face_img)
                            
                            is_detected = False
                            if st.session_state.thief_embeddings:
                                distances = [cosine_similarity(emb, t) for t in st.session_state.thief_embeddings]
                                is_detected = any(d < threshold for d in distances)
                                
                            color = (0, 0, 255) if is_detected else (0, 255, 0) # Red for thief, Green for person
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
            else:
                st.warning("Please upload a photo first.")

    else:
        # --------------------------------------------------------
        # LIVE WEBCAM MODE
        # --------------------------------------------------------
        if IS_CLOUD:
            st.info("☁️ Running on Hugging Face Spaces. Using native browser snapshot mode.")
            
            # Native Streamlit camera input widget (does not require STUN/TURN, works everywhere)
            camera_img = st.camera_input("Scan for Registered Criminals")

            if camera_img is not None:
                pil_img = Image.open(camera_img).convert("RGB")
                faces = detect_faces(st.session_state.mtcnn, pil_img)
                
                if len(faces) == 0:
                    st.warning("No faces detected in the snapshot.")
                else:
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    thief_detected = False
                    
                    for face_img, (x1, y1, x2, y2) in faces:
                        emb = compute_embedding(st.session_state.embedder, face_img)
                        
                        is_detected = False
                        if st.session_state.thief_embeddings:
                            distances = [cosine_similarity(emb, t) for t in st.session_state.thief_embeddings]
                            is_detected = any(d < threshold for d in distances)
                            
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
                    
                    st.image(img_bgr, channels="BGR", caption="Processed Frame")
                    
                    if thief_detected:
                        st.error("🚨 **THIEF DETECTED! Take Action Immediately!**")
                        audio_html = play_siren_js()
                        if audio_html:
                            st.components.v1.html(audio_html, height=0)
                    else:
                        st.success("🟢 Scan Complete. No threats detected.")
        else:
            st.success("💻 Running locally. Live Webcam loop enabled.")
            
            # Checkbox to start/stop the webcam
            run = st.checkbox("Start webcam")

            # Placeholder containers — updated in-place on every frame
            frame_window = st.empty()   # Displays the live video frame
            info_box = st.empty()       # Shows status messages (monitoring / thief detected)
            alarm_box = st.empty()      # Hidden container for injecting alarm HTML

            cap = None

            if run:
                # Open the default webcam (device index 0)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not access local webcam.")
                    run = False

            # MAIN WEBCAM LOOP — runs until "Start webcam" is unchecked
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame.")
                    break

                # Convert frame from BGR → RGB → PIL Image for MTCNN
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                # STEP 1: Detect all faces in this frame
                faces = detect_faces(st.session_state.mtcnn, pil)

                alerts = []   # Collect alert messages for detected thieves
                boxes = []    # Collect bounding boxes + whether each face is a thief

                # STEP 2 + 3: For each detected face, embed + compare
                for face_img, box in faces:
                    emb = compute_embedding(st.session_state.embedder, face_img)

                    # Compare this face against every stored thief embedding
                    match_scores = []
                    for thief_emb in st.session_state.thief_embeddings:
                        sim = cosine_similarity(emb, thief_emb)
                        match_scores.append(sim)

                    # A face is a THIEF if ANY stored embedding is below the threshold distance
                    is_detected = any(s < threshold for s in match_scores) if match_scores else False
                    boxes.append((box, is_detected))

                    if is_detected:
                        alerts.append("THIEF DETECTED ⚠️")

                # --------------------------------------------------------
                # STEP 4: Draw bounding boxes on the original frame
                # --------------------------------------------------------
                for (x1, y1, x2, y2), detected in boxes:
                    color = (0, 0, 255) if detected else (0, 255, 0) # Red for thief, Green for person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = "THIEF" if detected else "PERSON"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Display the annotated frame in the Streamlit UI
                frame_window.image(frame, channels="BGR")

                # Alert if any thief was detected
                if alerts:
                    info_box.error("🚨 Thief Detected! Take Action!")
                    
                    current_time = time.time()
                    if current_time - st.session_state.last_alarm_time > 2.0:
                        audio_html = play_siren_js()
                        with alarm_box:
                            st.components.v1.html(audio_html, height=0)
                        st.session_state.last_alarm_time = current_time
                else:
                    info_box.info("Monitoring...")
                    alarm_box.empty()

                # Small sleep to yield CPU
                time.sleep(0.01)

            if cap is not None:
                cap.release()


# ============================================================
# TAB 2: DEBUG / INFO
# ============================================================

with mode[1]:
    st.subheader("Debug / Info")
    st.write("Embeddings stored:", len(st.session_state.thief_embeddings))