import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import warnings

# ---------------------------------------------------------
# Global Initialization (Runs once at module load to avoid redundant overhead)
# ---------------------------------------------------------

# 1. Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
# model_selection=0 is suitable for faces close to the camera (within 2 meters), perfect for clock-in scenarios
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 2. Initialize ONNX Runtime Inference Session
# Note: You need to download a pre-trained MobileFaceNet ONNX model file from GitHub
# and rename it to "mobilefacenet.onnx" in the same directory.
MODEL_PATH = "mobilefacenet.onnx"
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    # Get the input node name of the model
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    warnings.warn(f"Failed to load ONNX model. Please ensure {MODEL_PATH} exists and the environment is correct. Error: {e}")
    ort_session = None


# ---------------------------------------------------------
# Core Interface Implementation (For Role 6 invocation)
# ---------------------------------------------------------

def get_face_embedding(frame: np.ndarray) -> np.ndarray:
    """
    Inputs a frame (after passing liveness detection) and uses a lightweight model to extract face embeddings.

    Parameters:
        frame: np.ndarray, the BGR image frame read by cv2
    Returns:
        np.ndarray, a flattened 1D array (typically 128 or 512 dimensions for MobileFaceNet).
        Returns an empty numpy array if no face is detected to prevent the program from crashing.
    """
    if ort_session is None:
        raise RuntimeError("ONNX model is not loaded. Cannot extract features.")

    # 1. Convert BGR to RGB (Hard requirement for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Perform fast face localization using MediaPipe
    results = face_detector.process(rgb_frame)

    # If no face is detected (although the liveness module theoretically intercepted it,
    # we still check for robustness)
    # Return a 0D array or the default value agreed with the team
    if not results.detections:
        return np.zeros((512,), dtype=np.float32)  # Assuming 512 dimensions, please modify based on your downloaded model

    # Extract the bounding box of the first face (the clock-in scenario defaults to only one person)
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape

    # Convert relative coordinates to absolute pixel coordinates
    x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
    w, h = int(bboxC.width * iw), int(bboxC.height * ih)

    # Out of bounds protection
    x, y = max(0, x), max(0, y)
    w, h = min(iw - x, w), min(ih - y, h)

    # 3. Crop the face region
    face_img = frame[y:y + h, x:x + w]
    if face_img.size == 0:
        return np.zeros((512,), dtype=np.float32)

    # 4. Image preprocessing (Align with MobileFaceNet input requirements)
    # Most lightweight face recognition models require an input size of 112x112
    face_img = cv2.resize(face_img, (112, 112))

    # ONNX models typically require RGB format and normalization to [-1, 1]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0

    # Transpose dimensions: HWC (112,112,3) -> CHW (3,112,112)
    face_img = np.transpose(face_img, (2, 0, 1))

    # Add Batch dimension: CHW -> NCHW (1,3,112,112)
    face_img = np.expand_dims(face_img, axis=0)

    # 5. ONNX ultra-fast inference
    # run() returns a list, [0] fetches the output tensor
    embeddings = ort_session.run(None, {input_name: face_img})[0]

    # 6. Flatten to a 1D array (Must satisfy the API Contract required by documentation)
    flattened_embedding = embeddings.flatten()

    return flattened_embedding