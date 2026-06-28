# 隐私导向的智能打卡系统 (Privacy-Aware Intelligent Attendance System)

这是一个主打隐私保护、具备双重活体检测功能的轻量级人脸识别打卡系统。专为普通 PC 和边缘设备（如树莓派）的本地部署而设计。

## 🌟 核心卖点 (Core Features)

* **隐私优先与本地存储 (Privacy-First & Local Storage)**: 采用“阅后即焚”的内存处理架构。人脸图像仅在内存 (RAM) 中流转，通过 EdgeFace/MobileFaceNet 提取特征后，立即使用 Python 的 `del` 命令彻底销毁原始图片。系统仅将特征向量 (Embeddings) 和学号加密存储在本地 SQLite 数据库中，实现物理级别的“零云端上传”。
* **双重活体检测 (Dual Liveness Detection)**:
  * **微表情检测 (Micro-motion EAR)**: 基于 MediaPipe 的眼睛纵横比 (EAR) 检测，有效防御静态照片攻击。
  * **面部纹理分析 (Facial Texture Analysis)**: 接入轻量级静默活体检测 (MiniFASNet)，有效防御高清屏幕重放 (Screen replay) 攻击。
* **极速轻量化架构 (Lightweight & High-Speed)**: 采用串行“漏斗式”流水线（人脸检测 -> EAR 眨眼检测 -> FAS 静默活体 -> 特征提取），极大降低了 CPU/RAM 占用，确保在低算力设备上的低延迟推理。

## 🛠 技术栈 (Tech Stack)

* **前端交互与视频流**: Streamlit, streamlit-webrtc
* **计算机视觉**: OpenCV, MediaPipe
* **深度学习推理**: ONNXRuntime / PyTorch
* **数据库与日志**: SQLite, Pandas

## 👥 团队分工 (Team Roles)

* **盘先琰**: Web 前端与视频流开发
* **黄晨禹**: 轻量级人脸特征提取模块
* **陈立杰**: 微表情活体检测模块 (EAR)
* **汤艾梧**: 动态纹理活体检测模块 (FAS)
* **陈锡翘**: 本地数据库与隐私管理逻辑
* **陈星弛**: 全栈集成与边缘设备性能压测

## 🚀 部署与运行 (Quick Start)

1. 克隆本项目仓库：
```bash
   git clone <your-repository-url>
安装环境依赖：

Bash
   pip install -r requirements.txt
启动 Streamlit 系统：

Bash
   streamlit run app.py
