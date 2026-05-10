# ============================================================
# alerts.py — STEP 4: Play Alarm When Thief Is Detected
# ============================================================
# This file handles the AUDIO ALERT system.
# When a thief is detected in the webcam feed, app.py calls
# play_siren_js() to immediately play the alarm sound.
#
# Challenge: Streamlit runs in a browser, not a desktop app.
# We can't just call Python's audio libraries directly.
# Solution: We inject an HTML <audio autoplay> tag into the
# Streamlit page using st.components.v1.html(), which makes
# the browser itself play the sound.
#
# FLOW:
#   app.py calls play_siren_js() when a thief match is found
#   → alarm.mp3 is read from disk
#   → encoded to base64 (text format safe for embedding in HTML)
#   → injected as a hidden HTML audio element that autoplays
# ============================================================

import streamlit as st   # Streamlit web framework — lets us inject HTML
import os                # OS path utilities — to find the alarm.mp3 file
import base64            # Encode binary audio data as ASCII text for embedding in HTML


def play_siren_js():
    """
    Play the alarm.mp3 sound in the user's browser when a thief is detected.
    This function returns the HTML string for the audio element, 
    allowing the caller to decide where and how to render it.
    """
    file_path = os.path.join(os.path.dirname(__file__), "alarm.mp3")

    try:
        with open(file_path, "rb") as f:
            data = f.read()

        b64 = base64.b64encode(data).decode("ascii")

        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
        """
        return audio_html

    except Exception as e:
        st.warning(f"Could not play alarm sound: {e}")
        return ""