# Privacy-Aware Intelligent Attendance System

A lightweight, privacy-preserving face recognition attendance system equipped with dual liveness detection. Designed specifically for local edge deployment on standard consumer PCs and low-power devices such as Raspberry Pi.

---

## 🌟 Key Features

* **Privacy-First Architecture & Local-Only Storage**
Implements an ephemeral, in-memory processing pipeline. Raw facial images reside exclusively in volatile memory (RAM). Once feature vectors are extracted via EdgeFace/MobileFaceNet, the original raw image data is immediately purged using explicit memory dereferencing (`del` / garbage collection). Only encrypted feature embeddings and user IDs are stored in a local SQLite database, ensuring zero cloud dependency and strict physical-level data isolation.
* **Dual-Layer Liveness Detection**
* **Micro-Motion Detection (EAR)**: Leverages MediaPipe Face Mesh for Eye Aspect Ratio (EAR) blink analysis, effectively neutralizing 2D static photograph spoofing.
* **Facial Texture Analysis (Silent Anti-Spoofing)**: Integrates a lightweight silent anti-spoofing network (MiniFASNet) to identify fine-grained texture anomalies and resist high-resolution screen-replay attacks.


* **Lightweight, High-Throughput Funnel Pipeline**
Employs a tiered early-exit execution pipeline (*Face Detection → EAR Blink Verification → FAS Anti-Spoofing → Feature Extraction*). Compute-heavy deep learning models are only triggered after lightweight geometric checks pass, minimizing CPU/RAM consumption and ensuring sub-second inference latency on resource-constrained edge hardware.

---

## 🛠 Tech Stack

* **UI & Video Streaming**: Streamlit, `streamlit-webrtc`
* **Computer Vision**: OpenCV, Google MediaPipe
* **Inference Engine**: ONNX Runtime / PyTorch
* **Storage & Data Processing**: SQLite3, Pandas

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

```

### 2. Set Up Environment & Dependencies

```bash
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

```

### 3. Run the Application

```bash
streamlit run app.py

```

---

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
