import cv2
import numpy as np
import mediapipe as mp

# initialize human keypoint detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# key points index of the eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye, h, w, landmarks):
    coords = np.array([(landmarks.landmark[i].x*w, landmarks.landmark[i].y*h) for i in eye])
    vertical1 = np.linalg.norm(coords[1] - coords[5])
    vertical2 = np.linalg.norm(coords[2] - coords[4])
    horizontal = np.linalg.norm(coords[0] - coords[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

EAR_THRESHOLD = 0.2
FRAME_COUNT = 0
BLINK_SUCCESS = False


def get_face_bbox(frame, margin=0.25):
    """
    Return a tight (x1, y1, x2, y2) face crop box using MediaPipe FaceMesh,
    expanded by `margin` (fraction of face size) on each side.
    Returns None if no face is detected.

    This lets other modules (e.g. fas.py) crop the actual face region
    instead of blindly cropping the center of the full camera frame.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0]
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin

    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))

    return (x1, y1, x2, y2)


def verify_liveness(frame):
    global FRAME_COUNT, BLINK_SUCCESS
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        FRAME_COUNT = 0
        BLINK_SUCCESS = False
        return False

    landmarks = result.multi_face_landmarks[0]
    left_ear = calculate_ear(LEFT_EYE, h, w, landmarks)
    right_ear = calculate_ear(RIGHT_EYE, h, w, landmarks)
    avg_ear = (left_ear + right_ear) / 2

    # blink detection
    if avg_ear < EAR_THRESHOLD:
        FRAME_COUNT += 1
    else:
        if FRAME_COUNT >= 2:
            BLINK_SUCCESS = True
        FRAME_COUNT = 0

    if BLINK_SUCCESS:
        BLINK_SUCCESS = False
        return True
    return False

# test code
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        res = verify_liveness(frame)
        cv2.putText(frame, "LIVE" if res else "BLINK PLEASE", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255 if res else 0,0), 2)
        cv2.imshow("Blink Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()