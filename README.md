# Facial Recognition + Emotion Recognition System

A dual-pipeline computer vision project built by the SASE Engineering Team. This system performs real-time **identity recognition** (who is this?) and **emotion classification** (how do they feel?) in parallel. Designed for deployment on Jetson Nano for autonomous robot interaction.

## Architecture

The system uses a parallel pipeline architecture:

```
Camera → Face Detection → Crop Face → ┬→ Identity Pipeline  → "Ben"
         (YuNet)                       │   (MobileFaceNet)
                                       │
                                       └→ Emotion Pipeline  → "Happy"
                                           (EfficientNet)

Output: "Ben | Happy (92%)"
```

### Semester 1: Identity Recognition
1. **Face Detection** - YuNet localizes faces in video frames
2. **Face Embedding** - MobileFaceNet generates 512D embeddings
3. **Identity Matching** - Cosine similarity against reference database

### Semester 2: Emotion Recognition
1. **Emotion Classification** - EfficientNet classifies 8 emotions
2. **Temporal Smoothing** - Moving average for stable output
3. **Parallel Integration** - Runs alongside identity pipeline

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Download Models
```bash
# Emotion model (Semester 2) - Clone and copy from HSEmotion repository
git clone --depth 1 https://github.com/HSE-asavchenko/face-emotion-recognition.git /tmp/face-emotion-recognition
cp /tmp/face-emotion-recognition/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx assets/enet_b0_8.onnx
```

### Check Progress
```bash
python check_progress.py
```

### Implementation Guides
- **Semester 1:** See `LEARNING_GUIDE.md` for face recognition
- **Semester 2:** See `SEMESTER_2_GUIDE.md` for emotion recognition

## Project Structure

```
Facial-Recognition/
├── models/
│   ├── face_model.py          # Semester 1: Identity embeddings
│   └── emotion_model.py       # Semester 2: Emotion classification
├── utils/
│   ├── face_detector.py       # YuNet face detection
│   └── emotion_smoother.py    # Semester 2: Temporal smoothing
├── core/
│   ├── face_recognizer.py     # Main recognition system (both semesters)
│   └── generate_embeddings.py # Reference database generation
├── data/                      # Photo capture and dataset
├── deployment/                # Jetson Nano deployment scripts
├── configs/                   # System configuration
└── assets/
    ├── face_detection_yunet_2023mar.onnx  # Face detection model
    └── enet_b0_8.onnx                      # Emotion model (download required)
```

## Tech Stack

- Python 3.8+
- InsightFace (buffalo_l model pack) - Identity
- ONNX Runtime - Emotion model inference
- OpenCV - Face detection (YuNet) and image processing
- NumPy - Numerical computing

## Emotion Classes

The system recognizes 8 emotions (AffectNet 8-class):

| Emotion | Description |
|---------|-------------|
| Anger | Furrowed brow, tight lips |
| Contempt | Asymmetric lip raise |
| Disgust | Wrinkled nose |
| Fear | Wide eyes, raised eyebrows |
| Happiness | Smile, raised cheeks |
| Neutral | Relaxed face |
| Sadness | Downturned mouth |
| Surprise | Wide eyes, open mouth |

## Deployment

Designed for deployment on Jetson Nano with >15 FPS real-time performance.

## About

This project is part of the SASE Engineering Team's autonomous robotics initiative. The robot can identify team members AND respond appropriately to their emotional state, enabling natural human-robot interaction.

## References

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [YuNet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet) - Face detection
- [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) - Emotion recognition
