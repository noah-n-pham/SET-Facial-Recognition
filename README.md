# Facial Recognition System

Build a complete real-time facial recognition system using pretrained MobileFaceNet from InsightFace.

## 🎯 What You'll Build

- **Face Detection** using YuNet (OpenCV)
- **Face Recognition** using pretrained MobileFaceNet (InsightFace)
- **Real-Time Recognition** via webcam at 30+ FPS
- **Hardware Deployment** to Jetson Nano + Arduino (optional)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Learning!
👉 **Open `LEARNING_GUIDE.md` and follow the step-by-step instructions!**

## 📚 Learning Approach

This project uses **TODO-based learning**:
- Read concept explanations in `LEARNING_GUIDE.md`
- Navigate to specific files as directed
- Implement TODOs with detailed guidance
- Test your implementation at each phase
- Build a fully functional system!

## ⏱️ Time Estimate

- **Phase 1:** Load Pretrained Model (30-45 min)
- **Phase 2:** Build Face Database (1-2 hours)
- **Phase 3:** Real-Time Recognition (2-3 hours)
- **Phase 4:** Hardware Deployment (optional, 2-3 hours)

**Total: 6-8 hours**

## 📊 What You'll Learn

✅ Face detection and recognition concepts  
✅ Embeddings and similarity matching  
✅ Working with pretrained models (industry standard)  
✅ Real-time computer vision  
✅ Hardware deployment  

## 🏗️ Project Structure

```
models/
  └── face_model.py           # Phase 1: Pretrained model wrapper

data/
  └── face_capture.py         # Phase 2A: Collect photos

core/
  ├── generate_embeddings.py  # Phase 2B: Build reference database
  └── face_recognizer.py      # Phase 3: Real-time recognition

utils/
  └── face_detector.py        # Helper: YuNet wrapper

deployment/                   # Phase 4: Optional
  └── jetson_inference.py
```

## 🎓 Educational Goals

This project teaches you:
- Modern face recognition systems
- Industry-standard practices (pretrained models)
- Real-time computer vision
- Edge device deployment
- Python, OpenCV, NumPy, and deep learning concepts

## 💡 Why Pretrained Models?

Instead of training our own model, we use **MobileFaceNet from InsightFace**:
- Trained on millions of faces
- Industry-standard accuracy
- No GPU or training time needed
- This is how production systems work!

## 📝 Requirements

- Python 3.8+
- Webcam (for data collection and real-time recognition)
- Basic Python knowledge

## 🚀 Ready to Start?

👉 **Open `LEARNING_GUIDE.md` now!**

---

**Questions?** All concepts are explained in `LEARNING_GUIDE.md` with detailed reasoning and instructions.
