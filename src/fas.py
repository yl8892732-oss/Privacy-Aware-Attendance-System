import sys, os
sys.path.append(os.path.dirname(__file__))
import cv2
import numpy as np
from collections import deque


class DynamicTextureDetector:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size

        self.frame_buffer = deque(maxlen=buffer_size)
        self.gray_buffer = deque(maxlen=buffer_size)
        self.feature_history = deque(maxlen=buffer_size)

        self.texture_var_history = deque(maxlen=buffer_size)
        self.edge_density_history = deque(maxlen=buffer_size)
        self.saturation_history = deque(maxlen=buffer_size)

        self.is_initialized = False

        self.smoothness_threshold = 0.15
        self.variance_threshold = 0.20

    def _extract_texture_features(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        texture_variance = np.var(gray)

        saturation_mean = np.mean(hsv[:, :, 1])

        return {
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'saturation_mean': saturation_mean
        }

    def _compute_temporal_consistency(self, history):
        if len(history) < 3:
            return 0.5

        arr = np.array(history)

        diff = np.abs(np.diff(arr))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        std_diff = np.std(diff)

        if mean_diff < 0.01:
            return 0.0

        consistency_score = 1.0 - (std_diff / (mean_diff + 0.001))
        consistency_score = max(0.0, min(1.0, consistency_score))

        return consistency_score

    def _compute_sequence_diversity(self, history):
        if len(history) < 3:
            return 0.5

        arr = np.array(history)

        if np.std(arr) < 0.01:
            return 0.0

        normalized_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 0.001)

        entropy = -np.sum(normalized_arr * np.log(normalized_arr + 0.001))
        max_entropy = np.log(len(normalized_arr))

        if max_entropy > 0:
            diversity_score = entropy / max_entropy
        else:
            diversity_score = 0.0

        return min(1.0, diversity_score)

    def _detect_repetition_pattern(self, history):
        if len(history) < 4:
            return 0.0

        arr = np.array(history)

        max_autocorr = 0.0
        for lag in range(2, min(len(arr) // 2, 10)):
            if len(arr) > lag * 2:
                corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
                if not np.isnan(corr):
                    max_autocorr = max(max_autocorr, abs(corr))

        return max_autocorr

    def add_frame(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        features = self._extract_texture_features(face_img)
        if features is None:
            return None

        self.frame_buffer.append(face_img)
        self.feature_history.append(features)

        self.texture_var_history.append(features['texture_variance'])
        self.edge_density_history.append(features['edge_density'])
        self.saturation_history.append(features['saturation_mean'])

        if len(self.frame_buffer) < self.buffer_size:
            return None

        return self._analyze_dynamic_texture()

    def _analyze_dynamic_texture(self):
        texture_var_arr = np.array(self.texture_var_history)
        edge_arr = np.array(self.edge_density_history)
        sat_arr = np.array(self.saturation_history)

        # 1. Temporal consistency
        texture_consistency = self._compute_temporal_consistency(texture_var_arr)
        edge_consistency = self._compute_temporal_consistency(edge_arr)
        sat_consistency = self._compute_temporal_consistency(sat_arr)
        avg_consistency = np.mean([texture_consistency, edge_consistency, sat_consistency])

        # 2. Sequence diversity
        texture_diversity = self._compute_sequence_diversity(texture_var_arr)
        edge_diversity = self._compute_sequence_diversity(edge_arr)
        sat_diversity = self._compute_sequence_diversity(sat_arr)
        avg_diversity = np.mean([texture_diversity, edge_diversity, sat_diversity])

        # 3. Repetition pattern detection
        texture_repetition = self._detect_repetition_pattern(texture_var_arr)
        edge_repetition = self._detect_repetition_pattern(edge_arr)
        avg_repetition = np.mean([texture_repetition, edge_repetition])

        # 4. Fluctuation range
        texture_range = np.max(texture_var_arr) - np.min(texture_var_arr)
        edge_range = np.max(edge_arr) - np.min(edge_arr)
        sat_range = np.max(sat_arr) - np.min(sat_arr)
        avg_range = np.mean([texture_range, edge_range, sat_range])


        consistency_score = avg_consistency
        diversity_score = avg_diversity

        if avg_repetition > 0.7:
            repetition_penalty = -0.4
        elif avg_repetition > 0.5:
            repetition_penalty = -0.2
        else:
            repetition_penalty = 0.1

        if avg_range < 10:
            range_penalty = -0.3
        elif avg_range > 200:
            range_penalty = 0.2
        else:
            range_penalty = 0.1

        live_score = (
                consistency_score * 0.4 +
                diversity_score * 0.3 +
                range_penalty * 0.15 +
                repetition_penalty * 0.15
        )

        live_score = max(0.0, min(1.0, live_score))

        is_live = live_score > 0.3


        return {
            'is_live': is_live,
            'confidence': live_score,
            'scores': {
                'consistency': consistency_score,
                'diversity': diversity_score,
                'repetition': avg_repetition,
                'range': avg_range
            }
        }

    def reset(self):
        self.frame_buffer.clear()
        self.gray_buffer.clear()
        self.feature_history.clear()
        self.texture_var_history.clear()
        self.edge_density_history.clear()
        self.saturation_history.clear()


class DualLivenessDetector:
    def __init__(self, buffer_size=10):
        from wenli import TextureLivenessDetector

        self.static_detector = TextureLivenessDetector(model_path=None, device='cpu')
        self.dynamic_detector = DynamicTextureDetector(buffer_size=buffer_size)

        self.static_weight = 0.8
        self.dynamic_weight = 0.2

        self.frame_count = 0

        print("[DualLiveness] Dual liveness detector initialized")
        print(f"[DualLiveness] Static texture detection + Dynamic texture detection (buffer: {buffer_size} frames)")

    def detect(self, face_img):
        if face_img is None or face_img.size == 0:
            return {
                'is_live': False,
                'confidence': 0.0,
                'attack_type': 'invalid_input'
            }

        self.frame_count += 1

        static_result = self.static_detector.detect(face_img)
        dynamic_result = self.dynamic_detector.add_frame(face_img)

        if dynamic_result is None:
            return {
                'is_live': static_result['is_live'],
                'confidence': static_result['confidence'],
                'attack_type': static_result['attack_type'],
                'static_result': static_result,
                'dynamic_result': None,
                'status': 'waiting_for_frames'
            }

        static_score = static_result['confidence']

        if static_score >= 0.95:
            is_live = True
            attack_type = 'real'
        elif static_score >= 0.85:
            is_live = False
            attack_type = 'screen_replay'
        else:
            is_live = False
            attack_type = static_result['attack_type']

        return {
            'is_live': is_live,
            'confidence': static_score,
            'attack_type': attack_type,
            'static_result': static_result,
            'dynamic_result': dynamic_result,
            'status': 'complete'
        }

    def reset(self):
        self.dynamic_detector.reset()
        self.frame_count = 0


def main():
    detector = DualLivenessDetector(buffer_size=10)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("Starting detection, press 'q' to exit")
    print("Waiting for buffer frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        face_crop = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]

        result = detector.detect(face_crop)

        if result['status'] == 'complete':
            status = "Live" if result['is_live'] else "Attack"
            print(f"[{result['attack_type']}] Confidence: {result['confidence']:.3f}")

        cv2.imshow('Dual Liveness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

_dual_detector = None


def _get_detector():
    global _dual_detector
    if _dual_detector is None:
        _dual_detector = DualLivenessDetector(buffer_size=10)
    return _dual_detector


def verify_liveness(frame: np.ndarray, face_bbox=None) -> bool:
    if frame is None or frame.size == 0:
        return False

    detector = _get_detector()

    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        face_crop = frame[y1:y2, x1:x2]
        if face_crop is None or face_crop.size == 0:
            face_crop = frame
    else:
        h, w = frame.shape[:2]
        if h > 64 and w > 64:
            face_crop = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
        else:
            face_crop = frame

    result = detector.detect(face_crop)

    if result.get('status') == 'waiting_for_frames':
        return bool(result.get('is_live', False))

    return bool(result.get('is_live', False))
