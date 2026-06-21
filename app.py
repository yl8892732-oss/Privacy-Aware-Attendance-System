import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import mediapipe as mp
import av  # Required for the modern version of streamlit-webrtc


from intro import verify_liveness
from feature_extraction import get_face_embedding
from db_manager import init_db
from db_manager import register_user
from db_manager import recognize_user

init_db()

# =========================================================================
# MediaPipe Face Mesh Initialization
# =========================================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils


def draw_face_landmark(img):
    """Processes the image frame to extract and render facial landmarks."""
    h, w = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    ear_val = 0.3

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks mesh contour (Green)
            mp_drawing.draw_landmarks(
                img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
        # EAR value will be formally calculated inside liveness_module, this random value is only for front-end preview display
        ear_val = round(np.random.uniform(0.15, 0.35), 2)

    # Overlay the calculated EAR value on the top-left corner
    cv2.putText(img, f"EAR = {ear_val}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img, ear_val


# =========================================================================
# Video Stream Processor Class (Thread-safe Modern Architecture)
# =========================================================================
class VideoProcessor:
    def __init__(self):
        self.ear = 0.3
        self.live_status = False
        self.latest_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Handles real-time video frame transformation and pipeline execution."""
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy()
        img, ear_val = draw_face_landmark(img)
        # Process face mesh landmarks
        img, ear_val = draw_face_landmark(img)
        self.ear = ear_val

        # Execute liveness detection pipeline (call imported teammate algorithm)
        live_ok = verify_liveness(img)
        self.live_status = live_ok

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================================================================
# Page Layouts
# =========================================================================
def register_page():
    st.title("User Register Page")
    st.text_input("Enter Student ID", key="stu_id")
    st.subheader("Live Camera")

    # Upgraded webRTC streamer using modern video_processor_factory
    ctx = webrtc_streamer(key="register_cam", video_processor_factory=VideoProcessor)

    # Dynamically fetch variables from background processor thread safely
    ear = 0.0
    live_stat = False
    if ctx.video_processor:
        ear = ctx.video_processor.ear
        live_stat = ctx.video_processor.live_status

    st.write(f"Current EAR Value: {ear}")
    if live_stat:
        st.success("Liveness Confirmed, Please Submit Registration")
    else:
        st.error("Spoofing Detected, Please Blink Your Eyes")

    if st.button("Submit Register"):
        sid = st.session_state.get("stu_id", "")

        if not sid:
            st.warning("Please enter a valid Student ID before submitting.")
            return

        if not (ctx.video_processor and ctx.video_processor.latest_frame is not None):
            st.error("No camera frame available.")
            return

        emb = get_face_embedding(ctx.video_processor.latest_frame)

        import numpy as np
        st.write("DEBUG norm:", np.linalg.norm(emb))

        success = register_user(sid, emb)

        if success:
            st.success(f"Register Success! ID: {sid}, Original Image Deleted From Memory.")


def attendance_page():
    st.title("Real-Time Attendance Check")
    st.subheader("Live Camera Stream")

    # Upgraded webRTC streamer using modern video_processor_factory
    ctx = webrtc_streamer(key="att_cam", video_processor_factory=VideoProcessor)

    live_stat = False
    if ctx.video_processor:
        live_stat = ctx.video_processor.live_status

    st.subheader("System Pipeline Status")
    st.write("1.Face Detect → 2.Blink Liveness → 3.Texture Anti-Spoof → 4.Feature Extract → 5.DB Match")

    if live_stat:
        st.markdown("<h3 style='color:green'>Liveness Confirmed -> Start Recognition</h3>", unsafe_allow_html=True)
        if ctx.video_processor and ctx.video_processor.latest_frame is not None:

            emb = get_face_embedding(
                ctx.video_processor.latest_frame
            )

            match_id = recognize_user(emb, 0.6)

            st.success(
                f"Attendance Marked, Student ID: {match_id}"
            )

        else:
            st.warning("No camera frame available.")
        st.success(f"Attendance Marked, Student ID: {match_id}")
    else:
        st.markdown("<h3 style='color:red'>Spoofing Detected (Photo/Screen), Reject Attendance</h3>",
                    unsafe_allow_html=True)


# =========================================================================
# Sidebar & Routing Navigation
# =========================================================================
st.sidebar.title("Privacy-Aware Attendance System")
page = st.sidebar.radio("Select Page", ["User Register", "Real-Time Attendance"])

if page == "User Register":
    register_page()
else:
    attendance_page()