import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import os


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MiniFASNet(torch.nn.Module):
    def __init__(self, num_classes=2, input_size=112):
        super(MiniFASNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 16, stride=1),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 32, stride=1),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
        )

        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, kernel_size=1, bias=False),
            torch.nn.Sigmoid()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128 * 7 * 7, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        attn_weight = self.attention(x)
        x = x * attn_weight
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TextureLivenessDetector:
    def __init__(self, model_path=None, device='cpu', input_size=112):
        self.device = device
        self.input_size = input_size
        self.model = None

        if model_path is not None and os.path.exists(model_path):
            self.model = self._load_model(model_path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.edge_low = 0.02
        self.edge_high = 0.30
        self.var_min = 300
        self.sat_min = 60
        self.sat_max = 190
        self.reflection_max = 0.45
        self.liveness_threshold = 0.5

    def _load_model(self, model_path):
        try:
            model = MiniFASNet(num_classes=2)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None

    def _preprocess(self, face_img):
        tensor = self.transform(face_img)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _extract_texture_features(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        texture_variance = np.var(gray)

        saturation_mean = np.mean(hsv[:, :, 1])

        _, bright_regions = cv2.threshold(hsv[:, :, 2], 220, 255, cv2.THRESH_BINARY)
        reflection_ratio = np.sum(bright_regions > 0) / (gray.shape[0] * gray.shape[1])

        return {
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'saturation_mean': saturation_mean,
            'reflection_ratio': reflection_ratio
        }

    def _heuristic_detect(self, features):
        score = 0.0

        ed = features['edge_density']
        if 0.005 < ed < 0.25:
            score += 0.30
        elif 0.25 <= ed < 0.40:
            score += 0.15
        elif ed >= 0.40:
            score -= 0.20

        tv = features['texture_variance']
        if tv > 500:
            score += 0.35
        elif tv > 300:
            score += 0.20
        elif tv < 150:
            score -= 0.20

        sm = features['saturation_mean']
        if 15 < sm < 200:
            score += 0.25
        elif sm < 10 or sm > 220:
            score -= 0.20

        rr = features['reflection_ratio']
        if rr < 0.10:
            score += 0.15
        elif rr > 0.50:
            score -= 0.25

        if tv > 3000:
            score += 0.10

        score = max(0.0, min(1.0, score))
        is_live = score > 0.45


        if is_live:
            attack_type = 'real'
        elif rr > 0.50:
            attack_type = 'screen_replay'
        elif tv < 150:
            attack_type = 'print_photo'
        elif ed >= 0.40:
            attack_type = 'screen_replay'
        else:
            attack_type = 'video_replay'

        return is_live, score, attack_type

    def detect(self, face_img):
        if face_img is None or face_img.size == 0:
            return {
                'is_live': False,
                'confidence': 0.0,
                'attack_type': 'invalid_input',
                'model_score': 0.0,
                'heuristic_score': 0.0,
                'features': {}
            }

        features = self._extract_texture_features(face_img)

        heu_is_live, heu_score, heu_type = self._heuristic_detect(features)

        model_is_live = False
        model_score = 0.0

        if self.model is not None:
            try:
                tensor = self._preprocess(face_img)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = F.softmax(outputs, dim=1)
                    model_score = probs[0][0].item()
                    model_is_live = model_score > self.liveness_threshold
            except Exception as e:
                pass

        if self.model is not None:
            if model_is_live and heu_is_live:
                is_live = True
                confidence = (model_score + heu_score) / 2
                attack_type = 'real'
            elif model_is_live and not heu_is_live:
                is_live = False
                confidence = heu_score
                attack_type = heu_type
            else:
                is_live = False
                confidence = min(model_score, heu_score)
                attack_type = heu_type
        else:
            is_live = heu_is_live
            confidence = heu_score
            attack_type = heu_type

        return {
            'is_live': is_live,
            'confidence': max(0.0, min(1.0, confidence)),
            'attack_type': attack_type,
            'model_score': model_score,
            'heuristic_score': heu_score,
            'features': features
        }