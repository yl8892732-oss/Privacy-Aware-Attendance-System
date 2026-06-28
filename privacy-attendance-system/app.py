import streamlit as st
import cv2
import numpy as np
import threading
import time
from db_manager import (
    init_db,
    register_user,
    recognize_user,
    log_attendance
)
from feature_extraction import (
    get_face_embedding
)
from blink_liveness import (
    verify_liveness as ear_verify,
    get_face_bbox
)
from fas import (
    verify_liveness as fas_verify
)
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

# WebRTC configuration
rtc_config = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)
VIDEO_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 1280, "min": 640},
        "height": {"ideal": 720, "min": 480},
        "frameRate": {"ideal": 30, "min": 15}
    },
    "audio": False
}

# ====================== Dual Liveness Detection ======================
def verify_dual_liveness(frame):
    ear_ok = False
    fas_ok = False
    try:
        ear_ok = ear_verify(frame)
    except Exception:
        pass
    try:
        face_bbox = get_face_bbox(frame)
        fas_ok = fas_verify(frame, face_bbox=face_bbox)
    except Exception:
        pass
    return ear_ok, fas_ok

# ====================== Camera Processing Class ======================
class CameraProcessor(VideoProcessorBase):
    def __init__(self):
        self.ear_pass = False
        self.fas_pass = False
        self.latest_frame = None
        self._lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()

    def recv(self, frame):
        self.frame_count += 1

        elapsed = time.time() - self.start_time

        if elapsed >= 5:
            fps = self.frame_count / elapsed

            print(f"[Performance] FPS = {fps:.2f}")

            self.frame_count = 0
            self.start_time = time.time()
        try:
            img = frame.to_ndarray(format="bgr24")
            with self._lock:
                self.latest_frame = img.copy()

            # Run liveness checks on every real video frame (not once per
            # Streamlit script rerun). Blink detection and the dynamic-texture
            # buffer in fas.py both need a continuous frame stream to work.
            ear_ok, fas_ok = verify_dual_liveness(img)
            if ear_ok:
                self.ear_pass = True
            if fas_ok:
                self.fas_pass = True
        except Exception:
            pass

        return frame

    def get_latest_frame(self):
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

# ====================== Basic Page Configuration ======================
st.set_page_config(
    page_title="Privacy-Aware Attendance System",
    layout="wide"
)
st.title("🔐 Privacy-Aware Attendance System")
st.caption("Face Recognition Attendance System")

# Initialize the database
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")

# Side navigation
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Register User", "Attendance"]
)

# State storage
if "last_menu" not in st.session_state:
    st.session_state["last_menu"] = ""
if "ear_pass" not in st.session_state:
    st.session_state["ear_pass"] = False
if "fas_pass" not in st.session_state:
    st.session_state["fas_pass"] = False

# Reset status when switching pages
if st.session_state["last_menu"] != menu:
    st.session_state["ear_pass"] = False
    st.session_state["fas_pass"] = False
    st.session_state["last_menu"] = menu

# ====================== Home page ======================
if menu == "Home":
    st.header("System Overview")
    st.markdown(
        """
### Workflow
1. Video Stream Capture
2. EAR Blink Detection
3. Texture FAS Detection
4. Face Feature Extraction
5. SQLite Storage
6. Attendance Logging
### Security Layers
- EAR Blink Liveness
- Texture FAS
- Dynamic Texture Analysis
- Face Recognition
- Attendance Logging
        """
    )
    st.success("Only embeddings are stored. No face images are saved.")

# ====================== Registration page ======================
elif menu == "Register User":
    st.header("Register New User")
    student_id = st.text_input("Student ID")

    # Remove video_html_width Incompatible parameters
    ctx = webrtc_streamer(
        key="register",
        video_processor_factory=CameraProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints=VIDEO_CONSTRAINTS
    )

    # Pull liveness state from the live CameraProcessor instance, which is
    # updated continuously per video frame in recv(). Sticky-OR into
    # session_state so progress isn't lost on the next Streamlit rerun.
    if ctx.video_processor:
        if ctx.video_processor.ear_pass:
            st.session_state["ear_pass"] = True
        if ctx.video_processor.fas_pass:
            st.session_state["fas_pass"] = True

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state["ear_pass"]:
            st.success("EAR Blink: PASS")
        else:
            st.warning("EAR Blink: Waiting")
    with col2:
        if st.session_state["fas_pass"]:
            st.success("Texture FAS: PASS")
        else:
            st.warning("Texture FAS: Waiting")

    if not (st.session_state["ear_pass"] and st.session_state["fas_pass"]):
        st.info("Look at the camera and blink naturally. This page refreshes automatically as checks complete.")

    if st.button("Register"):
        if not student_id.strip():
            st.error("Please enter Student ID.")
            st.stop()
        if not (st.session_state["ear_pass"] and st.session_state["fas_pass"]):
            st.error("Dual liveness verification not complete yet. Please blink and wait for the texture check.")
            st.stop()

        # latest_frame lives on the CameraProcessor instance (recv() runs in
        # a separate WebRTC worker; a bare module-level global would not be
        # visible here, so we read it through the processor object that
        # streamlit-webrtc already exposes via ctx.video_processor).
        frame = None
        for _ in range(20):  # up to ~1s total, in case of a brief startup gap
            if ctx.video_processor:
                frame = ctx.video_processor.get_latest_frame()
                if frame is not None:
                    break
            time.sleep(0.05)

        if frame is None:
            st.error("No frame received. Make sure the camera is running and try again.")
            st.stop()

        embedding = get_face_embedding(frame)
        if embedding is None or np.linalg.norm(embedding) == 0:
            st.error("No face detected in frame! Adjust your position.")
            st.stop()

        success = register_user(student_id, embedding)
        if success:
            st.success(f"{student_id} registered successfully.")
        else:
            st.error("Registration failed. Student ID may already exist.")

# ====================== Check-in page ======================
elif menu == "Attendance":
    st.header("Attendance Check-In")
    threshold = st.slider(
        "Recognition Threshold",
        min_value=0.40,
        max_value=1.00,
        value=0.60,
        step=0.05
    )

    # remove video_html_width Incompatible parameters
    ctx = webrtc_streamer(
        key="attendance",
        video_processor_factory=CameraProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints=VIDEO_CONSTRAINTS
    )

    # Display-only indicators, fed continuously from the live processor.
    if ctx.video_processor:
        if ctx.video_processor.ear_pass:
            st.session_state["ear_pass"] = True
        if ctx.video_processor.fas_pass:
            st.session_state["fas_pass"] = True

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state["ear_pass"]:
            st.success("EAR Blink: PASS")
        else:
            st.warning("Blink Please")
    with col2:
        if st.session_state["fas_pass"]:
            st.success("Texture FAS: PASS")
        else:
            st.warning("Texture FAS Checking")

    if st.button("Mark Attendance"):
        frame = None
        for _ in range(20):  # up to ~1s total
            if ctx.video_processor:
                frame = ctx.video_processor.get_latest_frame()
                if frame is not None:
                    break
            time.sleep(0.05)

        if frame is None:
            st.error("No frame received. Make sure the camera is running and try again.")
            st.stop()

        # Read liveness state from the live CameraProcessor at the moment
        # of check-in. This is updated continuously per video frame, so it
        # reflects the person currently in front of the camera right now.
        ear_ok = bool(ctx.video_processor.ear_pass) if ctx.video_processor else False
        fas_ok = bool(ctx.video_processor.fas_pass) if ctx.video_processor else False
        # Require a fresh blink + texture pass for every check-in attempt,
        # so a photo/static face can't ride on a liveness pass achieved
        # earlier in the session by someone else.
        if ctx.video_processor:
            ctx.video_processor.ear_pass = False
            ctx.video_processor.fas_pass = False
        st.session_state["ear_pass"] = False
        st.session_state["fas_pass"] = False

        if not (ear_ok and fas_ok):
            try:
                log_attendance("Unknown", "Spoofing_Detected")
            except Exception as log_err:
                st.warning(f"Log spoof record failed: {log_err}")
            st.error("Spoofing detected. Attendance rejected.")
            st.stop()

        embedding = get_face_embedding(frame)
        if embedding is None or np.linalg.norm(embedding) == 0:
            st.error("No face detected! Adjust your face to camera.")
            try:
                log_attendance("Unknown", "No_Face")
            except:
                pass
            st.stop()

        try:
            student_id = recognize_user(embedding, threshold=threshold)
        except TypeError:
            student_id = recognize_user(embedding)

        if student_id and student_id != "Unknown":
            try:
                log_attendance(student_id, "Success")
                st.success(f"Attendance marked for: {student_id}")
            except Exception as log_err:
                st.success(f"Recognized: {student_id}, attendance log failed: {log_err}")
        else:
            try:
                log_attendance("Unknown", "Recognition_Failed")
            except:
                pass
            st.warning("User not recognized. Complete registration first.")