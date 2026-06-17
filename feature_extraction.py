import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import warnings

# ---------------------------------------------------------
# 全局初始化 (在模块加载时只运行一次，避免每次调用重复加载浪费算力)
# ---------------------------------------------------------

# 1. 初始化 MediaPipe 人脸检测
mp_face_detection = mp.solutions.face_detection
# model_selection=0 适合距离摄像头较近的人脸 (2米以内)，非常适合打卡场景
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 2. 初始化 ONNX Runtime 推理会话
# 注意：你需要去 GitHub 下载一个预训练好的 MobileFaceNet 的 ONNX 模型文件
# 并将其重命名为 "mobilefacenet.onnx" 放在同一目录下。
MODEL_PATH = "mobilefacenet.onnx"
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    # 获取模型的输入节点名称
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    warnings.warn(f"无法加载 ONNX 模型，请确保 {MODEL_PATH} 存在且环境正确。报错: {e}")
    ort_session = None


# ---------------------------------------------------------
# 核心接口实现 (供角色 6 调用)
# ---------------------------------------------------------

def get_face_embedding(frame: np.ndarray) -> np.ndarray:
    """
    输入一帧画面（活体检测通过后），使用轻量级模型提取人脸特征。

    参数:
        frame: np.ndarray, cv2 读取的 BGR 格式图像帧
    返回:
        np.ndarray, 展平的一维数组 (通常 MobileFaceNet 是 128 或 512 维)。
        如果未检测到人脸，返回空的 numpy 数组以防程序崩溃。
    """
    if ort_session is None:
        raise RuntimeError("ONNX 模型未加载，无法提取特征。")

    # 1. BGR 转 RGB (MediaPipe 的硬性要求)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. 使用 MediaPipe 进行快速人脸定位
    results = face_detector.process(rgb_frame)

    # 如果没检测到脸（尽管活体模块理论上已经拦截了，但为了鲁棒性还是要做判断）
    # 返回一个 0 维数组或与团队约定好的默认值
    if not results.detections:
        return np.zeros((512,), dtype=np.float32)  # 假设特征维度为512，请根据你下载的具体模型修改

    # 提取第一张人脸的边界框 (打卡场景默认只有一个人)
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape

    # 将相对坐标转换为绝对像素坐标
    x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
    w, h = int(bboxC.width * iw), int(bboxC.height * ih)

    # 防止坐标越界 (Out of bounds protection)
    x, y = max(0, x), max(0, y)
    w, h = min(iw - x, w), min(ih - y, h)

    # 3. 裁剪人脸区域
    face_img = frame[y:y + h, x:x + w]
    if face_img.size == 0:
        return np.zeros((512,), dtype=np.float32)

    # 4. 图像预处理 (对齐 MobileFaceNet 的输入要求)
    # 大多数轻量级人脸识别模型要求输入尺寸为 112x112
    face_img = cv2.resize(face_img, (112, 112))

    # ONNX 模型通常需要 RGB 格式且需要归一化到 [-1, 1]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0

    # 转换维度排列：HWC (112,112,3) -> CHW (3,112,112)
    face_img = np.transpose(face_img, (2, 0, 1))

    # 增加 Batch 维度：CHW -> NCHW (1,3,112,112)
    face_img = np.expand_dims(face_img, axis=0)

    # 5. ONNX 极速推理
    # run() 返回一个列表，[0] 取出输出张量
    embeddings = ort_session.run(None, {input_name: face_img})[0]

    # 6. 展平为一维数组 (必须满足文档要求的 API Contract)
    flattened_embedding = embeddings.flatten()

    return flattened_embedding