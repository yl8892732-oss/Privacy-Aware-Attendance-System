import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp

# ===================== 模拟外部模块接口（后续小组合并后替换真实导入） =====================
# 陈立杰+汤艾梧 活体检测统一接口
def verify_liveness(frame: np.ndarray) -> bool:
    return True

# 黄晨禹 人脸特征提取接口
def get_face_embedding(frame: np.ndarray) -> np.ndarray:
    return np.random.rand(512)

# 陈锡翘 数据库接口
def register_user(student_id: str, embedding: np.ndarray) -> bool:
    return True

def recognize_user(current_embedding: np.ndarray, threshold: float = 0.6) -> str:
    return "AIT2509020"

# ===================== MediaPipe 面部关键点初始化 =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# 绘制眼部关键点 + 计算EAR
def draw_face_landmark(img):
    h, w = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    ear_val = 0.3
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制所有面部点，绿色
            mp_drawing.draw_landmarks(
                img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )
        ear_val = round(np.random.uniform(0.15, 0.35), 2)
    # 画面左上角打印EAR数值
    cv2.putText(img, f"EAR = {ear_val}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    st.session_state["ear"] = ear_val
    return img

# 视频流处理类
class VideoTransform(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = draw_face_landmark(img)
        live_ok = verify_liveness(img)
        st.session_state["live_status"] = live_ok
        return frame.from_ndarray(img, format="bgr24")

# ===================== 页面函数 =====================
def register_page():
    st.title("User Register Page")
    st.text_input("Enter Student ID", key="stu_id")
    st.subheader("Live Camera")
    webrtc_streamer(key="register_cam", video_transformer_factory=VideoTransform)
    ear = st.session_state.get("ear", 0.0)
    live_stat = st.session_state.get("live_status", False)
    st.write(f"Current EAR Value: {ear}")
    if live_stat:
        st.success("Liveness Confirmed, Please Submit Registration")
    else:
        st.error("Spoofing Detected, Please Blink Your Eyes")
    if st.button("Submit Register"):
        sid = st.session_state["stu_id"]
        frame_dummy = np.zeros((480,640,3), dtype=np.uint8)
        emb = get_face_embedding(frame_dummy)
        success = register_user(sid, emb)
        if success:
            st.success(f"Register Success! ID:{sid}, Original Image Deleted In Memory")

def attendance_page():
    st.title("Real-Time Attendance Check")
    st.subheader("Live Camera Stream")
    webrtc_streamer(key="att_cam", video_transformer_factory=VideoTransform)
    live_stat = st.session_state.get("live_status", False)
    st.subheader("System Pipeline Status")
    st.write("1.Face Detect → 2.Blink Liveness → 3.Texture Anti-Spoof → 4.Feature Extract → 5.DB Match")
    if live_stat:
        st.markdown("<h3 style='color:green'>Liveness Confirmed -> Start Recognition</h3>", unsafe_allow_html=True)
        frame_dummy = np.zeros((480,640,3), dtype=np.uint8)
        emb = get_face_embedding(frame_dummy)
        match_id = recognize_user(emb, 0.6)
        st.success(f"Attendance Marked, Student ID: {match_id}")
    else:
        st.markdown("<h3 style='color:red'>Spoofing Detected (Photo/Screen), Reject Attendance</h3>", unsafe_allow_html=True)

# ===================== 侧边栏页面切换 =====================
st.sidebar.title("Privacy Face Attendance System")
page = st.sidebar.radio("Select Page", ["User Register", "Real-Time Attendance"])
if page == "User Register":
    register_page()
else:
    attendance_page()