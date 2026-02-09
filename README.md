
# 🔥 Intelligent Fire Detection System using YOLO11

An advanced computer vision system trained on **42,000+ images** to detect fire in various environments with high precision.

## 📊 Performance Metrics
- **Dataset Size:** 42,000 images (Custom Chunked Dataset)
- **Model:** YOLO11 Nano (Optimized for Edge Devices)
- **mAP@50:** 0.736 (73.6%)
- **Precision:** 0.79
- **Inference Speed:** 1.8ms per image (on Tesla P100)
- **Model Size:** ~5.5 MB (Lightweight & Deployment Ready)

## 🚀 Key Features
- **Multi-Phase Training:** Model was trained in 6 distinct phases to ensure convergence on a massive dataset.
- **Optimized for Real-time:** Extremely low latency, suitable for CCTV and Drone integration.
- **Robustness:** Trained on diverse datasets including forest fires, indoor fires, and industrial settings.

## 🛠️ Tech Stack
- **Framework:** PyTorch, Ultralytics YOLO11
- **Platform:** Kaggle (GPU P100)
- **Language:** Python 3.12

## 📁 Project Structure
- `/notebooks`: Contains the full training execution and logs.
- `/weights`: Pre-trained `best.pt` file.
- `data.yaml`: Configuration for classes (Fire/Smoke).

## 📝 Author
**Ankit Kumar**
https://www.linkedin.com/in/ankit150802/

---
*Note: The dataset used is private. For access or collaboration, please open an issue.*
