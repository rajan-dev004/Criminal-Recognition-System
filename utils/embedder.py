# ============================================================
# embedder.py — STEP 2: Face → Embedding (Numbers)
# ============================================================
# This file converts a face image into a "face embedding" —
# a list of 512 decimal numbers that uniquely represents a face.
#
# Think of it like a face's "fingerprint" in number form.
# Two photos of the SAME person will produce SIMILAR numbers.
# Two photos of DIFFERENT people will produce VERY DIFFERENT numbers.
#
# Model Used: InceptionResnetV1 (FaceNet architecture)
#   - Pre-trained on VGGFace2 dataset (millions of face photos)
#   - Outputs a 512-dimensional vector per face
#
# FLOW:
#   app.py calls get_embedder() once at startup → stores model
#   app.py calls compute_embedding() for each detected face
#   Returns: numpy array of shape (512,) — the face fingerprint
# ============================================================

import torch                              # PyTorch — deep learning framework
from facenet_pytorch import InceptionResnetV1  # FaceNet model for face embeddings
from PIL import Image                     # PIL — image handling
import numpy as np                        # Numerical array operations


def get_embedder():
    """
    Load and return the pre-trained FaceNet (InceptionResnetV1) model.

    - pretrained='vggface2' → uses weights trained on the VGGFace2 dataset
    - .eval()               → sets the model to evaluation/inference mode
                              (turns off dropout and batch norm training behavior)

    Called ONCE at startup and stored in session_state to avoid
    reloading the model on every webcam frame.
    """
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model


def preprocess(image: Image.Image):
    """
    Prepare a face image for input into the FaceNet model.

    The model expects:
        - Size: 160×160 pixels
        - Format: float32 tensor, shape (1, 3, 160, 160)  [Batch, Channel, H, W]
        - Values: approximately in range [-1, 1]

    Steps:
        1. Resize image to 160×160 (FaceNet's required input size)
        2. Convert pixel values from [0, 255] → [0.0, 1.0]  (normalize)
        3. Transpose from HWC (Height, Width, Channel) → CHW (PyTorch format)
        4. Convert to a PyTorch tensor

    Arguments:
        image — PIL Image (the cropped face from detect_faces())

    Returns:
        A PyTorch tensor of shape (3, 160, 160)
    """
    # Resize face image to the model's expected input size
    img = image.resize((160, 160))

    # Convert to float32 and normalize pixel values to [0.0, 1.0]
    arr = np.float32(img) / 255.0

    # Transpose from (Height, Width, Channels) → (Channels, Height, Width)
    # PyTorch models expect the channel dimension first (CHW not HWC)
    arr = arr.transpose(2, 0, 1)

    # Convert numpy array to a PyTorch tensor
    tensor = torch.from_numpy(arr)
    return tensor


def compute_embedding(model: InceptionResnetV1, face_img: Image.Image):
    """
    Convert a face image into a 512-dimensional embedding vector.

    Arguments:
        model    — the FaceNet model from get_embedder()
        face_img — a PIL Image of a cropped face (from detect_faces())

    Returns:
        A numpy array of shape (512,) — the face's unique numerical fingerprint.
        This is used by compare.py to check if two faces belong to the same person.

    Steps:
        1. Preprocess the face image into a tensor
        2. Normalize values from [0, 1] → [-1, 1]  (FaceNet's expected range)
        3. Add a batch dimension: (3, 160, 160) → (1, 3, 160, 160)
        4. Pass through the model with no_grad (no gradient tracking since we only infer)
        5. Squeeze batch dim back: (1, 512) → (512,) and convert to numpy
    """
    # Step 1: Preprocess (resize, normalize to [0,1], convert to tensor)
    x = preprocess(face_img)

    # Step 2: Re-normalize from [0.0, 1.0] → [-1.0, 1.0]
    # Formula: (x - 0.5) / 0.5  →  0 becomes -1, 1 stays 1, 0.5 becomes 0
    x = (x - 0.5) / 0.5

    # Step 3: Add batch dimension: shape (3, 160, 160) → (1, 3, 160, 160)
    # Neural networks always expect a batch of images, even if it's just 1
    x = x.unsqueeze(0)

    # Step 4: Run the model — no_grad means don't compute gradients (saves memory)
    with torch.no_grad():
        emb = model(x)  # Output shape: (1, 512)

    # Step 5: Remove batch dimension and convert to numpy array → shape (512,)
    return emb.squeeze(0).cpu().numpy()