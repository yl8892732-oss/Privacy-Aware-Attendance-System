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

ear_status = False
fas_status = False

rtc_config = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)
# ==================================================
# Dual Liveness
# ==================================================

def verify_dual_liveness(frame):

    global ear_status
    global fas_status

    try:
        ear_status = ear_verify(frame)
    except Exception:
        ear_status = False

    try:
        fas_status = fas_verify(frame)
    except Exception:
        fas_status = False

    return ear_status and fas_status
# ==================================================
# Video Processor
# ==================================================

class CameraProcessor(VideoProcessorBase):

    def recv(self, frame):

        global latest_frame

        img = frame.to_ndarray(
            format="bgr24"
        )

        with frame_lock:
            latest_frame = img.copy()

        return frame
# ==================================================
# Streamlit Config
# ==================================================

st.set_page_config(
    page_title="Privacy-Aware Attendance System",
    layout="wide"
)

st.title(
    "🔐 Privacy-Aware Attendance System"
)

st.caption(
    "Face Recognition Attendance System"
)

try:
    init_db()
except Exception as e:
    st.error(
        f"Database initialization failed: {e}"
    )
# ==================================================
# Sidebar
# ==================================================

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Register User",
        "Attendance"
    ]
)
# ==================================================
# Home
# ==================================================

if menu == "Home":

    st.header(
        "System Overview"
    )

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

    st.success(
        "Only embeddings are stored. No face images are saved."
    )
# ==================================================
# Register User
# ==================================================

elif menu == "Register User":

    st.header(
        "Register New User"
    )

    student_id = st.text_input(
        "Student ID"
    )

    webrtc_streamer(
        key="register",
        video_processor_factory=CameraProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
    col1, col2 = st.columns(2)

    with col1:

        if ear_status:
            st.success(
                "EAR Blink: PASS"
            )
        else:
            st.warning(
                "EAR Blink: Waiting"
            )

    with col2:

        if fas_status:
            st.success(
                "Texture FAS: PASS"
            )
        else:
            st.warning(
                "Texture FAS: Waiting"
            )
            if st.button(
        "Register"
    ):

        if student_id.strip() == "":
            st.error(
                "Please enter Student ID."
            )
            st.stop()

        with frame_lock:

            if latest_frame is None:
                st.error(
                    "No frame received."
                )
                st.stop()

            frame = latest_frame.copy()
                try:

            if not verify_dual_liveness(
                frame
            ):

                st.error(
                    "Dual liveness verification failed."
                )

                st.stop()

            embedding = get_face_embedding(
                frame
            )

            success = register_user(
                student_id,
                embedding
            )

            if success:

                st.success(
                    f"{student_id} registered successfully."
                )

            else:

                st.error(
                    "Registration failed."
                )

        except Exception as e:

            st.exception(e)
# ==================================================
# Attendance
# ==================================================

elif menu == "Attendance":

    st.header(
        "Attendance Check-In"
    )

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
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
    col1, col2 = st.columns(2)

    with col1:

        if ear_status:
            st.success(
                "EAR Blink: PASS"
            )
        else:
            st.warning(
                "Blink Please"
            )

    with col2:

        if fas_status:
            st.success(
                "Texture FAS: PASS"
            )
        else:
            st.warning(
                "Texture FAS Checking"
            )
            if st.button(
        "Mark Attendance"
    ):

        with frame_lock:

            if latest_frame is None:

                st.error(
                    "No frame received."
                )

                st.stop()

            frame = latest_frame.copy()
            try:

            if not verify_dual_liveness(
                frame
            ):

                try:

                    log_attendance(
                        "Unknown",
                        "Spoofing_Detected"
                    )

                except:
                    pass

                st.error(
                    "Spoofing detected."
                )

                st.stop()
            embedding = get_face_embedding(
                frame
            )
            try:

                student_id = recognize_user(
                    embedding,
                    threshold=threshold
                )

            except TypeError:

                student_id = recognize_user(
                    embedding
                )
                if (
                student_id
                and
                student_id != "Unknown"
            ):

                try:

                    log_attendance(
                        student_id,
                        "Success"
                    )

                except:
                    pass

                st.success(
                    f"Attendance marked for: {student_id}"
                )
                else:

                try:

                    log_attendance(
                        "Unknown",
                        "Recognition_Failed"
                    )

                except:
                    pass

                st.warning(
                    "User not recognized."
                )
            except Exception as e:

            st.exception(
                e
            )
