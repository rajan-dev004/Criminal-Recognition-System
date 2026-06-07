# ============================================================
# detectors.py — STEP 1: Face Detection
# ============================================================
# This file is responsible for FINDING faces inside an image.
# It uses MTCNN (Multi-task Cascaded Convolutional Networks),
# a deep learning model specifically designed to detect human
# faces — even at different sizes, angles, and lighting.
#
# FLOW:
#   app.py calls get_mtcnn() once at startup → stores model
#   app.py calls detect_faces() for every image / webcam frame
#   Returns: list of (cropped_face_image, bounding_box_coords)
# ============================================================

from facenet_pytorch import MTCNN  # Pre-built face detection neural network
from PIL import Image              # Python Imaging Library — handles image data
import numpy as np                 # Numerical operations on arrays


def get_mtcnn():
    """
    Initialize and return the MTCNN face detector.

    - keep_all=True  → detect ALL faces in an image, not just the largest one
    - device='cpu'   → run on CPU (change to 'cuda' if you have a GPU)

    This is called ONCE at app startup and stored in session_state
    so we don't reload the model every frame (that would be very slow).
    """
    return MTCNN(keep_all=True, device='cpu')


def detect_faces(mtcnn: MTCNN, image: Image.Image):
    """
    Detect all faces in a given PIL Image and return their crops + positions.

    Arguments:
        mtcnn  — the MTCNN model loaded by get_mtcnn()
        image  — a PIL Image (RGB format)

    Returns:
        A list of tuples: [(face_crop, (x1, y1, x2, y2)), ...]
        - face_crop     → cropped PIL Image of just the face region
        - (x1,y1,x2,y2) → bounding box coordinates of the face in the original image

    How it works:
        1. mtcnn.detect() runs the neural network on the image.
        2. It returns bounding boxes (rectangles around faces) and
           confidence probabilities for each detected face.
        3. We crop each detected face from the original image.
        4. That crop is what gets sent to the embedder next.
    """
    # Run MTCNN face detection — returns box coordinates and confidence scores
    boxes, probs = mtcnn.detect(image)

    faces = []  # Will hold (face_image_crop, bounding_box) pairs

    if boxes is None:
        # No faces found in this image — return empty list
        return faces

    for box in boxes:
        # Convert box coordinates to integers and clamp to non-negative values
        # (sometimes the detector returns slightly negative coords near image edges)
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]

        # Ensure the box has at least 1 pixel width/height (avoid zero-size crops)
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)

        # Crop the face region from the full image
        cropped = image.crop((x1, y1, x2, y2))

        # Store the face crop along with its position in the original image
        faces.append((cropped, (x1, y1, x2, y2)))

    return faces