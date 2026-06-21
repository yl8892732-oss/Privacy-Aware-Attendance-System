import streamlit as st
import cv2
import numpy as np

from db_manager import (
    init_db,
    register_user,
    recognize_user,
    log_attendance,
    get_user_count,
)

from feature_extraction import get_face_embedding
from blink_detector import verify_liveness


# ==========================================================
# Initialize
# ==========================================================

st.set_page_config(
    page_title="Privacy-Aware Attendance System",
    layout="wide"
)

init_db()

st.title("🔐 Privacy-Aware Attendance System")
st.markdown("### Local Face Recognition Attendance System")


# ==========================================================
# Sidebar
# ==========================================================

menu = st.sidebar.selectbox(
    "Select Function",
    [
        "Home",
        "Register User",
        "Attendance"
    ]
)


# ==========================================================
# Home
# ==========================================================

if menu == "Home":

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Registered Users",
            get_user_count()
        )

    with col2:
        st.metric(
            "Database",
            "SQLite Local"
        )

    st.success(
        "Privacy-Aware Design: Only embeddings are stored. No face images saved."
    )


# ==========================================================
# Register User
# ==========================================================

elif menu == "Register User":

    st.subheader("User Registration")

    student_id = st.text_input(
        "Student ID",
        placeholder="AIT2509001"
    )

    image_file = st.camera_input(
        "Capture Face"
    )

    if st.button("Register"):

        if image_file is None:
            st.error("Please capture an image.")
            st.stop()

        if student_id.strip() == "":
            st.error("Please enter student ID.")
            st.stop()

        file_bytes = np.asarray(
            bytearray(image_file.getvalue()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        st.info("Performing liveness detection...")

        if not verify_liveness(frame):

            st.error(
                "Liveness Verification Failed. Please blink."
            )

            del frame
            st.stop()

        st.success("Liveness Verified")

        try:

            embedding = get_face_embedding(frame)

            success = register_user(
                student_id,
                embedding
            )

            del frame

            if success:
                st.success(
                    f"{student_id} registered successfully."
                )
            else:
                st.error(
                    "Registration failed."
                )

        except Exception as e:
            st.error(str(e))


# ==========================================================
# Attendance
# ==========================================================

elif menu == "Attendance":

    st.subheader("Attendance Check-In")

    image_file = st.camera_input(
        "Capture Face"
    )

    if st.button("Mark Attendance"):

        if image_file is None:
            st.error(
                "Please capture an image."
            )
            st.stop()

        file_bytes = np.asarray(
            bytearray(image_file.getvalue()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        st.info("Performing liveness detection...")

        if not verify_liveness(frame):

            st.error(
                "Spoofing Detected!"
            )

            log_attendance(
                "Unknown",
                "Spoofing_Detected"
            )

            del frame
            st.stop()

        st.success(
            "Liveness Verified"
        )

        try:

            embedding = get_face_embedding(frame)

            student_id = recognize_user(
                embedding
            )

            del frame

            if student_id != "Unknown":

                log_attendance(
                    student_id,
                    "Success"
                )

                st.success(
                    f"Attendance Marked: {student_id}"
                )

            else:

                log_attendance(
                    "Unknown",
                    "Recognition_Failed"
                )

                st.warning(
                    "User Not Recognized"
                )

        except Exception as e:
            st.error(str(e))
