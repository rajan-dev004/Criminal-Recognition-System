# ============================================================
# compare.py — STEP 3: Are Two Faces the Same Person?
# ============================================================
# This file compares two face embeddings (512-number vectors)
# to determine if they belong to the same person.
#
# It uses COSINE SIMILARITY — a mathematical technique that
# measures the "angle" between two vectors in high-dimensional space.
#
# Key idea:
#   - Same person   → embeddings point in similar directions → small distance
#   - Different person → embeddings point in different directions → large distance
#
# FLOW:
#   app.py calls cosine_similarity(live_face_emb, stored_thief_emb)
#   If distance < threshold (e.g. 0.6) → MATCH → thief detected!
# ============================================================

import numpy as np              # Numerical operations
from numpy.linalg import norm   # norm() computes the "length" / magnitude of a vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the COSINE DISTANCE between two face embedding vectors.

    Despite the name "cosine_similarity", this function actually returns
    a DISTANCE (dissimilarity) value, not a similarity score:
        - 0.0  → identical faces (same direction)
        - ~0.6 → different people (threshold boundary)
        - 2.0  → maximally opposite

    Formula:
        cosine_distance = 1 - (A · B) / (||A|| * ||B||)

        where:
            A · B     = dot product (sum of element-wise multiplications)
            ||A||     = magnitude/length of vector A
            ||B||     = magnitude/length of vector B
            1e-8      = tiny epsilon to prevent division-by-zero

    Arguments:
        a — embedding vector of face 1 (numpy array, shape 512)
        b — embedding vector of face 2 (numpy array, shape 512)

    Returns:
        float — cosine distance (lower = more similar)
    """
    # Flatten in case the vectors have extra dimensions (e.g., shape (512,1) → (512,))
    a = a.flatten()
    b = b.flatten()

    # Compute denominator: product of magnitudes + tiny epsilon to avoid division-by-zero
    denom = (norm(a) * norm(b)) + 1e-8

    # Return cosine DISTANCE: 1 minus cosine similarity
    # np.dot(a, b) = sum of element-wise products (how aligned the vectors are)
    return 1.0 - (np.dot(a, b) / denom)


def is_match(a: np.ndarray, b_list: list[np.ndarray], threshold: float) -> bool:
    """
    Check if embedding `a` matches ANY embedding in the list `b_list`.

    This is a helper function (currently not used directly by app.py,
    which does the comparison inline — but it's here for reusability).

    Arguments:
        a         — live face embedding (from current webcam frame)
        b_list    — list of stored thief embeddings (uploaded by admin)
        threshold — cosine distance cutoff; lower = stricter matching
                    e.g., 0.6 means "accept if distance < 0.6"

    Returns:
        True  — if this face matches at least one thief embedding
        False — if no match found, or if b_list is empty
    """
    if not b_list:
        # No thief photos uploaded yet — can't match against nothing
        return False

    for b in b_list:
        # Check each stored thief embedding
        if cosine_similarity(a, b) < threshold:
            # Distance is below threshold → faces are similar → same person!
            return True

    # No thief matched this face
    return False