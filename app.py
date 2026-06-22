import streamlit as st
import cv2
import numpy as np
import threading

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
    verify_liveness as ear_verify
)

from fas import (
    verify_liveness as fas_verify
)

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

# ==================================================
# Global State
# ==================================================
latest_frame = None
frame_lock = threading.Lock()
liveness_lock = threading.Lock()

# Permanent flag: remains True once passed detection
ear_success_flag = False
fas_success_flag = False

rtc_config = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

# ==================================================
# Dual Liveness Optimized Logic
# ==================================================
def verify_dual_liveness(frame):
    global ear_success_flag, fas_success_flag
    with liveness_lock:
        # EAR blink detection, exceptions will not overwrite existing success flag
        try:
            if ear_verify(frame):
                ear_success_flag = True
        except Exception:
            pass

        # FAS anti-spoof detection, optimized exception handling to avoid permanent lock of state due to single frame error
        try:
            fas_res = fas_verify(frame)
            if fas_res:
                fas_success_flag = True
        except Exception:
            pass

        return ear_success_flag and fas_success_flag

# ==================================================
# Video Processor Class
# ==================================================
class CameraProcessor(VideoProcessorBase):
    def recv(self, frame):
        global latest_frame
        img = frame.to_ndarray(format="bgr24")
        with frame_lock:
            latest_frame = img.copy()
        # Execute liveness detection every frame and update permanent flags
        verify_dual_liveness(img)
        return frame

# ==================================================
# Streamlit Page Config (String quote syntax fixed)
# ==================================================
st.set_page_config(
    page_title="Privacy-Aware Attendance System",
    layout="wide"
)
st.title("🔐 Privacy-Aware Attendance System")
st.caption("Face Recognition Attendance System")

# Initialize database
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")

# Sidebar navigation menu
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Register User", "Attendance"]
)

# Reset cached frame and liveness flags when switching pages (global variables at module top do not require global keyword, fully functional)
if "last_menu" not in st.session_state:
    st.session_state["last_menu"] = ""
if st.session_state["last_menu"] != menu:
    with frame_lock:
        latest_frame = None
    with liveness_lock:
        ear_success_flag = False
        fas_success_flag = False
    st.session_state["last_menu"] = menu

# ==================================================
# Home Page
# ==================================================
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

# ==================================================
# Register User Page
# ==================================================
elif menu == "Register User":
    st.header("Register New User")
    student_id = st.text_input("Student ID")

    webrtc_streamer(
        key="register",
        video_processor_factory=CameraProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False}
    )

    col1, col2 = st.columns(2)
    with col1:
        if ear_success_flag:
            st.success("EAR Blink: PASS")
        else:
            st.warning("EAR Blink: Waiting")
    with col2:
        if fas_success_flag:
            st.success("Texture FAS: PASS")
        else:
            st.warning("Texture FAS: Waiting")

    if st.button("Register"):
        if student_id.strip() == "":
            st.error("Please enter Student ID.")
            st.stop()

        with frame_lock:
            if latest_frame is None:
                st.error("No frame received.")
                st.stop()
            frame = latest_frame.copy()

        try:
            # Dual liveness validation
            if not verify_dual_liveness(frame):
                st.error("Dual liveness verification failed. Please blink and complete texture check.")
                st.stop()

            embedding = get_face_embedding(frame)
            # Fix empty face vector judgment: zero vector indicates no detected face
            if embedding is None or np.linalg.norm(embedding) == 0:
                st.error("No face detected in frame! Adjust your position.")
                st.stop()

            success = register_user(student_id, embedding)
            if success:
                st.success(f"{student_id} registered successfully.")
            else:
                st.error("Registration failed. Student ID may already exist.")
        except Exception as e:
            st.exception(e)

# ==================================================
# Attendance Page (Syntax error fixed by adding 'as' keyword)
# ==================================================
elif menu == "Attendance":
    st.header("Attendance Check-In")
    threshold = st.slider(
        "Recognition Threshold",
        min_value=0.40,
        max_value=1.00,
        value=0.60,
        step=0.05
    )

    webrtc_streamer(
        key="attendance",
        video_processor_factory=CameraProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False}
    )

    col1, col2 = st.columns(2)
    with col1:
        if ear_success_flag:
            st.success("EAR Blink: PASS")
        else:
            st.warning("Blink Please")
    with col2:
        if fas_success_flag:
            st.success("Texture FAS: PASS")
        else:
            st.warning("Texture FAS Checking")

    if st.button("Mark Attendance"):
        with frame_lock:
            if latest_frame is None:
                st.error("No frame received.")
                st.stop()
            frame = latest_frame.copy()

        try:
            # Reject attendance if liveness check fails and log spoof attempt
            if not verify_dual_liveness(frame):
                try:
                    log_attendance("Unknown", "Spoofing_Detected")
                except Exception as log_err:
                    st.warning(f"Log spoof record failed: {log_err}")
                st.error("Spoofing detected. Attendance rejected.")
                st.stop()

            embedding = get_face_embedding(frame)
            # Fix empty face vector judgment
            if embedding is None or np.linalg.norm(embedding) == 0:
                st.error("No face detected! Adjust your face to camera.")
                try:
                    log_attendance("Unknown", "No_Face")
                except:
                    pass
                st.stop()

            # Face recognition workflow
            try:
                student_id = recognize_user(embedding, threshold=threshold)
            except TypeError:
                student_id = recognize_user(embedding)

            if student_id and student_id != "Unknown":
                try:
                    log_attendance(student_id, "Success")
                    st.success(f"Attendance marked for: {student_id}")
                # --------------------------
                # Critical syntax fix: add 'as log_err'
                # --------------------------
                except Exception as log_err:
                    st.success(f"Recognized: {student_id}, attendance log failed: {log_err}")
            else:
                try:
                    log_attendance("Unknown", "Recognition_Failed")
                except:
                    pass
                st.warning("User not recognized. Complete registration first.")

        except Exception as e:
            st.exception(e)
