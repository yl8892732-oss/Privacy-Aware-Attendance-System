import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import mediapipe as mp
import av

# Import backend modules developed by team members
from intro import verify_liveness
from feature_extraction import get_face_embedding
from db_manager import init_db, register_user, recognize_user, log_attendance

# Automatically initialize the local SQLite database if it does not exist
init_db()

# Initialize MediaPipe components for background rendering and fallback processing
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# =========================================================================
# Streamlit Session State Initialization (System State Locks)
# =========================================================================
# Because Streamlit reruns the entire script upon any user interaction,
# a global state tracker is required to maintain the "Liveness Passed" condition.
if "liveness_unlocked" not in st.session_state:
    st.session_state.liveness_unlocked = False


def draw_face_landmark(img):
    """
    Processes the raw BGR image frame to extract and render facial landmarks mesh.
    This serves as visual feedback for the user in the frontend UI.
    """
    h, w = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Render the facial mesh contour lines in Green color
            mp_drawing.draw_landmarks(
                img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
    return img


# =========================================================================
# Video Stream Processor Class (Thread-safe WebRTC Architecture)
# =========================================================================
class VideoProcessor:
    def __init__(self):
        # Memory slot to store the unaltered frame (free of green facial mesh overlays)
        self.latest_frame = None
        # Background signal placeholder for liveness validation status
        self.live_status = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Handles real-time video frame transformation and pipeline routing."""
        img = frame.to_ndarray(format="bgr24")

        # CRITICAL FIX: Save a clean duplicate of the raw camera frame for feature extraction.
        # This prevents the facial recognition model from analyzing the drawn green mesh lines.
        self.latest_frame = img.copy()

        # Route the raw frame into the team's blink liveness detection module
        is_live = verify_liveness(img)
        if is_live:
            # Once a blink is registered, latch the active status inside the worker thread
            self.live_status = True

            # Apply visual facial contours onto the image for live stream feedback
        img = draw_face_landmark(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================================================================
# Frontend Page Layouts
# =========================================================================

def register_page():
    st.title("User Register Page")
    # Prompt user to input their unique key identifier
    sid = st.text_input("Enter Student ID", key="stu_id")
    st.subheader("Live Camera")

    # Deploy the WebRTC streaming engine tied to our custom thread processor
    ctx = webrtc_streamer(key="register_cam", video_processor_factory=VideoProcessor)

    # State Synchronization: Bridge the background execution thread with the Streamlit engine
    if ctx.video_processor and ctx.video_processor.live_status:
        st.session_state.liveness_unlocked = True

    # Render dynamic status boxes based on the current state engine
    if st.session_state.liveness_unlocked:
        st.success("Liveness Confirmed! Pipeline unlocked. Ready to register.")
    else:
        st.error("Spoofing Detected! Please BLINK your eyes to unlock the system.")

    # Execution trigger button for registration submission
    if st.button("Submit Register"):
        sid_val = st.session_state.get("stu_id", "").strip()

        # Guard Clause 1: Validate text field inputs
        if not sid_val:
            st.warning("Please enter a valid Student ID before submitting.")
            return

        # Guard Clause 2: Prevent processing if liveness assertion fails
        if not st.session_state.liveness_unlocked:
            st.error("Access Denied: Liveness detection not passed yet. Please blink.")
            return

        # Guard Clause 3: Verify video hardware buffer accessibility
        if ctx.video_processor and ctx.video_processor.latest_frame is not None:
            # Step 1: Extract 512-dimensional vector embedding from the clean camera frame
            with st.spinner("Extracting face embedding vectors..."):
                emb = get_face_embedding(ctx.video_processor.latest_frame)

            # Step 2: Print matrix diagnostics to verify data flow integrity
            st.write("DEBUG Feature Norm Vector:", np.linalg.norm(emb))

            # Step 3: Write data package into the SQLite database architecture
            success = register_user(sid_val, emb)
            if success:
                st.success(f"Register Success! ID: {sid_val} is securely stored.")
                # Step 4: Write standard trace telemetry record to log file
                log_attendance(sid_val, status="Registered")

                # Step 5: Securely reset system locks to prevent subsequent request hijacking
                st.session_state.liveness_unlocked = False
                if ctx.video_processor:
                    ctx.video_processor.live_status = False
            else:
                st.error("Database write execution error.")
        else:
            st.error("No valid camera frame grabbed. Please adjust your positioning.")


def attendance_page():
    st.title("Real-Time Attendance Check")
    st.subheader("Live Camera Stream")

    # Deploy WebRTC instance for the dedicated attendance validation stream
    ctx = webrtc_streamer(key="att_cam", video_processor_factory=VideoProcessor)

    # Sync liveness checks from WebRTC background worker thread to global context
    if ctx.video_processor and ctx.video_processor.live_status:
        st.session_state.liveness_unlocked = True

    st.subheader("System Pipeline Status")
    st.write("1. Face Mesh Track -> 2. Blink Liveness -> 3. Feature Extract -> 4. DB Match -> 5. CSV Logging")

    if st.session_state.liveness_unlocked:
        st.markdown("<h3 style='color:green'>Liveness Confirmed -> Pipeline Unlocked</h3>", unsafe_allow_html=True)

        # Discrete execution lock wrapper button to shield SQLite from multi-threading racing conditions
        if st.button("Clock-in Now (Verify Identity)"):
            if ctx.video_processor and ctx.video_processor.latest_frame is not None:
                # Step 1: Process current frame into an absolute inference vector array
                emb = get_face_embedding(ctx.video_processor.latest_frame)

                # Step 2: Compute Cosine Similarity comparisons across the database backend
                match_id = recognize_user(emb, threshold=0.6)

                # Step 3: Handle execution logic branching outcomes
                if match_id != "Unknown":
                    st.success(f"Attendance Marked! Student ID: {match_id}")
                    log_attendance(match_id, status="Success")
                else:
                    st.error("Attendance Failed: User identity unrecognized.")
                    log_attendance("Unknown", status="Failed_Unrecognized")

                # Step 4: Tear down system lock conditions forcing the next individual to blink re-verify
                st.session_state.liveness_unlocked = False
                if ctx.video_processor:
                    ctx.video_processor.live_status = False
            else:
                st.warning("Camera streaming delay, please try clicking again.")
    else:
        st.markdown("<h3 style='color:red'>Spoofing Detected, Please Blink Your Eyes</h3>", unsafe_allow_html=True)


# =========================================================================
# Sidebar & Navigation State Router (Memory Isolation Controls)
# =========================================================================
st.sidebar.title("Privacy-Aware Attendance")

# Setup routing session hooks on initialization to cleanly swap pages
if "current_page" not in st.session_state:
    st.session_state.current_page = "User Register"

selected_page = st.sidebar.radio("Select Page", ["User Register", "Real-Time Attendance"])

# State Router Intervention: If a page flip is captured, flush out any active validation tokens.
# This prevents liveness authorization flags from leaking across isolated sessions.
if selected_page != st.session_state.current_page:
    st.session_state.liveness_unlocked = False
    st.session_state.current_page = selected_page

# View engine routing router execution
if st.session_state.current_page == "User Register":
    register_page()
else:
    attendance_page()