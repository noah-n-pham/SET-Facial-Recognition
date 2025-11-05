# Facial Recognition System

Real-time facial recognition using ResNet-18 + ArcFace loss, deployed on Jetson Nano with Arduino hardware control.

## Tech Stack

- **Face Detection**: OpenCV + YuNet
- **Training**: PyTorch + ResNet-18 (torchvision) + ArcFace loss
- **Transfer Learning**: Frozen backbone approach (ResNet-18 frozen, head trainable)
- **Augmentation**: Albumentations + torchvision.transforms
- **Arrays/Math**: NumPy (L2-norm, cosine similarity, embeddings)
- **Deployment**: ONNX + TensorRT (Jetson Nano)
- **Hardware**: PySerial + Arduino

## Transfer Learning Strategy

This project uses **Frozen Backbone + Trainable Head** for efficient training:
- ❄️ **ResNet-18 backbone**: Frozen (~11M params, pretrained features)
- 🔥 **Embedding + ArcFace head**: Trainable (~264K params)

Benefits: 2x faster training, less overfitting, better for small datasets.  
📖 See the Transfer Learning Strategy section below for details.

## Project Structure

```
Facial-Recognition/
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── configs/
│   └── config.yaml            # Training & inference config
│
├── assets/                     # Pre-trained models
│   ├── face_detection_yunet_2023mar.onnx
│   └── opencv_bootcamp_assets_12.zip
│
├── data/
│   └── raw/
│       └── Dataset/           # 9 people × 100 images (900 total)
│           ├── ben/
│           ├── hoek/
│           ├── james/
│           ├── janav/
│           ├── joyce/
│           ├── nate/
│           ├── noah/
│           ├── rishab/
│           └── tyler/
│
└── src/                       # Source code
    ├── data/                  # Data handling
    │   ├── augmentation.py    # Transform definitions (on-the-fly)
    │   └── collection.py      # Face capture from webcam
    │
    ├── inference/             # Real-time recognition
    │   ├── face_detection.py  # YuNet wrapper
    │   └── inference.py       # Recognition + Arduino control
    │
    ├── models/                # Model architectures (to implement)
    ├── training/              # Training pipeline (to implement)
    ├── export/                # ONNX export (to implement)
    └── utils/                 # Utilities (to implement)
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (Python 3.8+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Data (Optional)

```bash
python src/data/collection.py
```

### 3. Train Model (To Implement)

```bash
# Will train ResNet-18 + ArcFace on the 900 face images
python src/training/train.py --config configs/config.yaml
```

### 4. Export to ONNX (To Implement)

```bash
python src/export/export_onnx.py --checkpoint models/best_model.pth
```

### 5. Run Inference

```bash
# Current: Face detection only
python src/inference/inference.py

# Future: With trained model + Arduino
python src/inference/inference.py --model models/face_model.onnx --arduino /dev/ttyUSB0
```

## Implementation Roadmap

### ✅ Ready to Use
- YuNet face detection (working)
- Data collection script (working)
- Transform definitions (on-the-fly augmentation)
- Configuration system (config.yaml with freeze_backbone setting)
- Transfer learning strategy documented in this README

### 📝 To Implement
1. **`src/data/dataset.py`** - PyTorch Dataset class for loading images with augmentation
2. **`src/models/resnet_arcface.py`** - ResNet-18 backbone + ArcFace head
3. **`src/models/losses.py`** - ArcFace loss implementation
4. **`src/training/train.py`** - Training loop with backpropagation
5. **`src/export/export_onnx.py`** - Export trained model to ONNX/TensorRT
6. **`src/inference/inference.py`** - Update for embedding comparison + Arduino control
7. **Arduino sketch** - Serial communication for hardware control

## Configuration

Edit `configs/config.yaml` for:
- Dataset paths and batch size
- Model architecture (ResNet-18, embedding dim)
- ArcFace parameters (margin=0.5, scale=64.0)
- Training hyperparameters (epochs, learning rate, optimizer)
- Inference settings (thresholds)
- Hardware settings (Arduino port, CUDA device)

## Development vs Deployment

- **Development**: Train on laptop/desktop with GPU
- **Deployment**: Inference on Jetson Nano with TensorRT (2-4× speedup)

## Dataset

- **9 team members**: ben, hoek, james, janav, joyce, nate, noah, rishab, tyler
- **100 images per person** = 900 total images
- Located in `data/raw/Dataset/`

## Requirements

- Python 3.8+
- CUDA-capable GPU (for training)
- Jetson Nano (for deployment)
- Arduino (for hardware control)

---

**Current Status**: Clean codebase ready for implementation! 🚀
