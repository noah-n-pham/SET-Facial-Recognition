# Implementation Guide

This document explains the project structure and what needs to be implemented in each file.

---

## 📂 Project Structure

```
Facial-Recognition/
├── .gitignore                          # Git ignore (data, models, cache)
├── README.md                           # Project overview & quick start
├── requirements.txt                    # Python dependencies
├── IMPLEMENTATION_GUIDE.md             # This file
│
├── configs/
│   └── config.yaml                     # Hyperparameters & settings
│
├── assets/
│   ├── face_detection_yunet_2023mar.onnx    # YuNet face detection model
│   └── opencv_bootcamp_assets_12.zip        # Additional assets
│
├── data/
│   └── raw/
│       └── Dataset/                    # 9 people × 100 images = 900 total
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
└── src/
    ├── data/
    │   ├── augmentation.py             # ✅ DONE - Transform definitions
    │   └── collection.py               # ✅ DONE - Data collection script
    │
    ├── inference/
    │   ├── face_detection.py           # ✅ DONE - YuNet face detection
    │   └── inference.py                # ⚠️  UPDATE - Add model inference
    │
    ├── models/                         # 📝 TO IMPLEMENT
    ├── training/                       # 📝 TO IMPLEMENT
    ├── export/                         # 📝 TO IMPLEMENT
    └── utils/                          # 📝 TO IMPLEMENT
```

---

## ✅ Completed Files (No Action Needed)

### `configs/config.yaml`
**Purpose**: Centralized configuration for all hyperparameters  
**Contains**: Dataset paths, batch size, learning rate, ArcFace parameters, device settings  
**Action**: Review and adjust values as needed during training

### `src/data/augmentation.py`
**Purpose**: Define data augmentation transforms  
**Contains**: 
- `get_train_transforms()` - Aggressive augmentations for training
- `get_val_transforms()` - Minimal augmentations for validation
**Action**: None - ready to use

### `src/data/collection.py`
**Purpose**: Collect face images from webcam  
**Contains**: YuNet face detection + image saving logic  
**Action**: None - use to collect more data if needed

### `src/inference/face_detection.py`
**Purpose**: Standalone face detection demo  
**Contains**: YuNet face detection with visualization  
**Action**: None - working demo script

---

## 📝 Files To Implement

### Priority 1: Core Training Pipeline

#### `src/data/dataset.py`
**Purpose**: PyTorch Dataset class for loading images with on-the-fly augmentation  
**What to implement**:
- Custom Dataset class inheriting from `torch.utils.data.Dataset`
- `__init__`: Load image paths and labels from `data/raw/Dataset/`
- `__getitem__`: Load image, apply Albumentations transforms, return (image_tensor, label)
- `__len__`: Return total number of images
- Train/validation split logic (80/20)

**Key requirements**:
- Use `get_train_transforms()` and `get_val_transforms()` from augmentation.py
- Apply different augmentations each epoch (on-the-fly)
- Convert BGR (OpenCV) to RGB if needed
- Return normalized tensors ready for ResNet-18

---

#### `src/models/resnet_arcface.py`
**Purpose**: ResNet-18 backbone + ArcFace classification head  
**What to implement**:
- Load pretrained ResNet-18 from torchvision
- Remove final fully connected layer
- Add embedding layer (512 dimensions recommended)
- Add ArcFace head for 9 classes (team members)
- Forward pass returns both embeddings and logits

**Key requirements**:
- Use `torchvision.models.resnet18(pretrained=True)`
- Embedding layer output should be L2-normalized
- ArcFace head applies angular margin during training
- Model should output raw embeddings during inference

---

#### `src/models/losses.py`
**Purpose**: ArcFace loss implementation  
**What to implement**:
- ArcFace loss function with angular margin
- Takes embeddings, labels, margin (m), and scale (s) as inputs
- Computes cosine similarity between embeddings and class weights
- Applies angular margin to target class
- Returns classification loss

**Key requirements**:
- Margin (m) = 0.5, Scale (s) = 64.0 (from config.yaml)
- Use cosine similarity for angular distance
- Add margin only to the target class
- Compatible with PyTorch autograd for backpropagation

---

#### `src/training/train.py`
**Purpose**: Main training script  
**What to implement**:
- Load config from `configs/config.yaml`
- Create train/val DataLoaders using custom Dataset
- Initialize ResNet-18 + ArcFace model
- Setup optimizer (Adam) and learning rate scheduler
- Training loop:
  - Forward pass through model
  - Compute ArcFace loss
  - Backpropagation
  - Update weights
  - Track metrics (loss, accuracy)
- Validation loop after each epoch
- Save best model checkpoint
- Early stopping if validation loss plateaus

**Key requirements**:
- Use config.yaml for all hyperparameters
- Save checkpoints to `models/checkpoints/`
- Log training progress (loss, accuracy per epoch)
- Save best model based on validation accuracy
- Support resuming training from checkpoint

---

### Priority 2: Model Export

#### `src/export/export_onnx.py`
**Purpose**: Export trained PyTorch model to ONNX format  
**What to implement**:
- Load trained model checkpoint
- Set model to eval mode
- Create dummy input tensor (1, 3, 224, 224)
- Export to ONNX using `torch.onnx.export()`
- Verify ONNX model loads correctly
- Optional: Optimize for TensorRT (Jetson Nano deployment)

**Key requirements**:
- Input: PyTorch .pth checkpoint
- Output: ONNX model in `models/exported/`
- Ensure ONNX opset compatibility
- Test exported model produces same outputs as PyTorch
- Target: Jetson Nano with TensorRT optimization

---

### Priority 3: Inference & Deployment

#### Update `src/inference/inference.py`
**Purpose**: Real-time face recognition with hardware control  
**What to implement**:
- Load ONNX model for inference
- Precompute embeddings for all 9 team members (reference database)
- Real-time workflow:
  1. Detect face with YuNet (already working)
  2. Extract face ROI
  3. Preprocess face (resize, normalize)
  4. Run through ONNX model → get embedding
  5. Compare with reference embeddings (cosine similarity)
  6. If similarity > threshold: identify person
  7. Send coordinates + person ID to Arduino via PySerial
- Display recognized name on video frame

**Key requirements**:
- Load reference embeddings from file (numpy array)
- Use cosine similarity for comparison
- Threshold from config.yaml (default: 0.6)
- PySerial communication to Arduino
- Handle "unknown person" case
- Real-time performance (30+ FPS)

---

### Priority 4: Utilities (Optional)

#### `src/training/utils.py`
**Purpose**: Training helper functions  
**What to implement**:
- `save_checkpoint()` - Save model, optimizer, epoch, metrics
- `load_checkpoint()` - Load checkpoint for resuming training
- `plot_training_curves()` - Visualize loss/accuracy over epochs
- Early stopping class

---

#### `src/utils/metrics.py`
**Purpose**: Evaluation metrics  
**What to implement**:
- Accuracy calculation
- Confusion matrix
- Per-class precision/recall/F1
- ROC curves for face verification

---

#### `src/utils/visualization.py`
**Purpose**: Visualization tools  
**What to implement**:
- Plot augmented samples
- Visualize embeddings (t-SNE/UMAP)
- Display confusion matrix
- Show training curves

---

## 🔧 Hardware: Arduino Code

### Arduino Sketch
**Purpose**: Receive serial commands and control hardware  
**What to implement**:
- Setup serial communication (9600 baud)
- Parse incoming messages (e.g., "X100Y200\n" or "PERSON:ben\n")
- Control servos/motors based on face coordinates
- Control LEDs/displays based on recognized person
- Send acknowledgment back to Jetson Nano

**Key requirements**:
- Match baud rate in PySerial code
- Use newline `\n` as message delimiter
- Handle command parsing robustly
- Non-blocking code for real-time response

---

## 🎯 Implementation Order

### Phase 1: Training (Desktop/Laptop)
1. `src/data/dataset.py` - Load data with augmentation
2. `src/models/resnet_arcface.py` - Model architecture
3. `src/models/losses.py` - ArcFace loss
4. `src/training/train.py` - Training loop
5. Train model on 900 images (~30-50 epochs)

### Phase 2: Export
6. `src/export/export_onnx.py` - Export to ONNX
7. Test ONNX model accuracy matches PyTorch

### Phase 3: Deployment (Jetson Nano)
8. Update `src/inference/inference.py` - Add model inference
9. Create reference embedding database
10. Test real-time recognition on Jetson Nano
11. Optimize with TensorRT if needed

### Phase 4: Hardware Integration
12. Arduino sketch - Serial communication + hardware control
13. Connect Arduino to Jetson Nano via USB
14. Test full pipeline: detection → recognition → hardware response

---

## 📊 Expected Outcomes

### Training Metrics
- **Training accuracy**: >95% (on 720 training images)
- **Validation accuracy**: >90% (on 180 validation images)
- **Training time**: 30-50 epochs (~10-20 minutes on GPU)

### Inference Performance
- **Face detection**: Real-time (30+ FPS)
- **Recognition**: <50ms per face on Jetson Nano with TensorRT
- **Accuracy**: >90% on known team members
- **False positives**: Minimal with proper threshold tuning

---

## 💡 Key Concepts

### On-the-fly Augmentation
- Apply random augmentations during training, not saved to disk
- Each epoch sees different augmented versions of images
- Increases effective dataset size and prevents overfitting

### ArcFace Loss
- Adds angular margin to embeddings for better separation
- Forces model to learn more discriminative features
- Better than softmax for face recognition

### Embedding Space
- 512-dimensional vectors representing each face
- Similar faces have high cosine similarity (close to 1)
- Different faces have low cosine similarity (close to 0)
- Recognition = find closest embedding in reference database

### ONNX + TensorRT
- ONNX: Platform-independent model format
- TensorRT: NVIDIA optimization for Jetson Nano
- 2-4× speedup compared to PyTorch inference

---

## 🔍 Testing Strategy

1. **Overfit test**: Train on 5 images, should reach 100% accuracy
2. **Full training**: Train on full dataset, monitor loss curves
3. **Validation**: Check accuracy on held-out validation set
4. **Export verification**: Ensure ONNX matches PyTorch outputs
5. **Real-world test**: Test on live webcam with various lighting
6. **Unknown person test**: Verify system rejects unfamiliar faces
7. **Hardware test**: Verify Arduino responds correctly to commands

---

## 📝 Configuration Tips

Edit `configs/config.yaml` for:
- **Batch size**: Reduce if GPU memory limited (16 or 32)
- **Learning rate**: Start with 0.001, reduce if loss plateaus
- **Epochs**: 50 is default, adjust based on convergence
- **ArcFace margin**: 0.5 is standard, increase for harder separation
- **Similarity threshold**: 0.6 default, tune to balance accuracy vs false positives

---

**Ready to implement!** Follow the priority order and test each component before moving to the next. 🚀

