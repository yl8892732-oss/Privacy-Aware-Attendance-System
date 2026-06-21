import streamlit as st
import cv2
import numpy as np

from db_manager import (
    init_db,
    register_user,
    recognize_user,
    log_attendance
)

from feature_extraction import get_face_embedding


# ==================================================
# Temporary Liveness Check
# ==================================================

def verify_liveness(frame):
    """
    Temporary implementation.
    Replace with EAR blink detection later.
    """
    return True


# ==================================================
# Streamlit Config
# ==================================================

st.set_page_config(
    page_title="Privacy-Aware Attendance System",
    layout="wide"
)

st.title("🔐 Privacy-Aware Attendance System")
st.caption("Face Recognition Attendance System")

# ==================================================
# Database Init
# ==================================================

try:
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")


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

    st.header("System Overview")

    st.markdown("""
    ### Workflow

    1. Capture Face
    2. Liveness Verification
    3. Face Feature Extraction
    4. SQLite Storage
    5. Attendance Logging
    """)

    st.success(
        "Privacy Protection: Only face embeddings are stored. No face images are saved."
    )

# ==================================================
# Register
# ==================================================

elif menu == "Register User":

    st.header("Register New User")

    student_id = st.text_input(
        "Student ID",
        placeholder="AIT2509001"
    )

    image_file = st.camera_input(
        "Capture Face Image"
    )

    if st.button("Register"):

        if image_file is None:
            st.error("Please capture a face image.")
            st.stop()

        if student_id.strip() == "":
            st.error("Please enter a Student ID.")
            st.stop()

        try:

            file_bytes = np.asarray(
                bytearray(image_file.getvalue()),
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if not verify_liveness(frame):

                st.error(
                    "Liveness verification failed."
                )

                del frame
                st.stop()

            embedding = get_face_embedding(frame)

            success = register_user(
                student_id,
                embedding
            )

            del frame

            if success:
                st.success(
                    f"User {student_id} registered successfully."
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

    st.header("Attendance Check-In")

    image_file = st.camera_input(
        "Capture Face Image"
    )

    threshold = st.slider(
        "Recognition Threshold",
        min_value=0.40,
        max_value=1.00,
        value=0.60,
        step=0.05
    )

    if st.button("Mark Attendance"):

        if image_file is None:
            st.error(
                "Please capture a face image."
            )
            st.stop()

        try:

            file_bytes = np.asarray(
                bytearray(image_file.getvalue()),
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if not verify_liveness(frame):

                log_attendance(
                    "Unknown",
                    "Spoofing_Detected"
                )

                st.error(
                    "Spoofing detected."
                )

                del frame
                st.stop()

            embedding = get_face_embedding(frame)

            try:

                student_id = recognize_user(
                    embedding,
                    threshold=threshold
                )

            except TypeError:

                # Compatible with old version
                student_id = recognize_user(
                    embedding
                )

            del frame

            if student_id and student_id != "Unknown":

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
            st.exception(e)
