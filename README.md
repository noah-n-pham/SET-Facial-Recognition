# Facial Recognition System

A facial recognition project built by the SASE Engineering Team using pretrained models from InsightFace. This system is designed for real-time operation and will be integrated into our autonomous robot, deployed on Jetson Nano for person identification and interaction.

## Architecture

The system uses a two-stage pipeline:

1. **Face Detection** - YuNet localizes faces in video frames
2. **Face Recognition** - InsightFace recognition model generates 512D embeddings for identity matching against a reference database

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Check Progress
```bash
python check_progress.py
```

### Implementation Guide
See `LEARNING_GUIDE.md` for detailed instructions on each phase.

## Project Structure

```
Facial-Recognition/
├── models/          # Face embedding model wrapper
├── utils/           # Face detection utilities
├── data/            # Photo capture and dataset
├── core/            # Recognition system and database generation
├── deployment/      # Jetson Nano deployment scripts
└── configs/         # System configuration
```

## Tech Stack

- Python 3.8+
- InsightFace (buffalo_l model pack)
- OpenCV (YuNet detection)
- NumPy
- ONNX Runtime

## Deployment

This system is designed for deployment on Jetson Nano.

## About

This project is part of the SASE Engineering Team's autonomous robotics initiative. We're building a face recognition system to enable our robot to identify and interact with team members, combining computer vision with embedded systems deployment.

## References

- [InsightFace](https://github.com/deepinsight/insightface)
- [YuNet Face Detection](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
