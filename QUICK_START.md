# 🚀 Quick Start Guide

## 🎓 Learning Approach: Frozen Backbone + ArcFace Head

This project uses **transfer learning with a frozen backbone**:
- ❄️ ResNet-18 backbone is **FROZEN** (pretrained weights from ImageNet, not trainable)
- 🔥 Embedding layer and ArcFace head are **TRAINABLE** (adapted for face recognition)

**Why?** Training only ~264K parameters instead of 11M:
- ✅ Faster training (5-10 min vs 10-20 min)
- ✅ Less overfitting on small dataset (900 images)
- ✅ Better generalization
- ✅ Less GPU memory needed

📖 See `TRANSFER_LEARNING_GUIDE.md` for detailed explanation.

---

## Step-by-Step Implementation

### ✅ Phase 1: Environment Setup (Complete These First)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test GPU access
python test_gpu.py
# Expected: "✅ GPU is ready for training!" (or warning if CPU-only)

# 4. Verify dataset
python verify_dataset.py
# Expected: ~900 images across 9 people

# 5. Create output directories
mkdir -p models/checkpoints models/exported logs

# 6. Test face detection (existing code)
python src/inference/face_detection.py
# Expected: Window showing face detection (press ESC to exit)
```

### 📚 Phase 2: Dataset Implementation

**File to implement:** `src/data/dataset.py`

**TODOs to complete:**
1. Initialize data structures in `__init__`
2. Build image list and labels from folders
3. Split into train/val sets
4. Implement `__len__` and `__getitem__`
5. Implement `get_class_names`
6. Complete `create_dataloaders` function

**Test your implementation:**
```bash
python test_dataset.py
# Expected output:
# - TRAIN dataset: ~720 images, 9 classes
# - VAL dataset: ~180 images, 9 classes
# - Class names: ['ben', 'hoek', 'james', ...]
# - Image shape: [3, 224, 224]
```

**Visualize augmentations:**
```bash
python visualize_augmentations.py
# Check logs/augmentation_samples.png - faces should look realistic
```

### 🧠 Phase 3: Model & Training

**Files to implement (in order):**

#### 3.1 Model Architecture: `src/models/resnet_arcface.py`
- Load ResNet-18 and remove final layer
- **FREEZE the backbone** (set `requires_grad = False` for backbone params)
- Add embedding layer with BatchNorm (trainable)
- Add ArcFace classification head (trainable)
- Implement forward pass with L2 normalization
- Implement `extract_embedding` for inference
- Implement `count_parameters`

**Test:**
```bash
python test_model.py
# Expected: ~11M total parameters, ~264K trainable
# Embeddings shape: [2, 512], Logits shape: [2, 9]
```

**Verify backbone is frozen:**
```python
# In test_model.py, add:
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / Total: {total:,}")
# Should show ~264K trainable out of ~11M total
```

#### 3.2 Loss Function: `src/models/losses.py`
- Implement ArcFace loss (or start with SimplifiedLoss)
- Compute cosine similarities
- Add angular margin to target class
- Apply scale and compute cross-entropy

**Test:**
```bash
python test_loss.py
# Expected: Loss value around 2-3
```

#### 3.3 Quick Sanity Check: `quick_overfit_test.py`
- Reduce dataset to 5 images
- Train for 50 iterations
- Should reach 100% accuracy

**Test:**
```bash
python quick_overfit_test.py
# Expected: Accuracy reaches 100% in ~20-30 iterations
```

#### 3.4 Full Training: `src/training/train.py`
- Implement `train_one_epoch` function
- Implement `validate` function
- Setup model, optimizer, scheduler in `main`
- Implement training loop with checkpointing

**Train the model:**
```bash
python src/training/train.py
# Expected: 5-10 minutes on GPU (faster due to frozen backbone!)
# Training accuracy should reach >90%
# Validation accuracy should reach 85-92%
# Best model saved to models/checkpoints/best_model.pth
# Note: Frozen backbone trains faster with similar accuracy
```

### 🎥 Phase 4: Local Inference

**Files to implement:**

#### 4.1 Generate References: `src/utils/generate_embeddings.py`
- Load trained model
- Extract embeddings for each person
- Average multiple images per person
- Save to `models/reference_embeddings.npy`
- Save class names to `models/label_names.txt`

**Run:**
```bash
python src/utils/generate_embeddings.py
# Expected: Creates reference_embeddings.npy [9, 512] and label_names.txt
```

#### 4.2 Webcam Recognition: `src/inference/webcam_recognition.py`
- Load model and references
- Initialize YuNet detector
- Implement `recognize_face` with similarity comparison
- Implement `run_webcam` main loop

**Test:**
```bash
python src/inference/webcam_recognition.py
# Expected: Window showing webcam with names above faces (press ESC to exit)
```

### 🤖 Phase 5: Jetson Deployment

**Files to implement:**

#### 5.1 Export to ONNX: `src/export/export_onnx.py`
- Load PyTorch model
- Export using `torch.onnx.export`
- Verify with ONNX Runtime
- Compare PyTorch vs ONNX outputs

**Run:**
```bash
python src/export/export_onnx.py
# Expected: Creates models/exported/face_recognition.onnx
```

#### 5.2 Setup Jetson Nano
```bash
# On Jetson:
sudo apt-get install python3-pip python3-opencv
pip3 install numpy pyyaml albumentations onnxruntime

# On laptop (copy files to Jetson):
scp -r models/ configs/ src/ jetson@<JETSON_IP>:~/Facial-Recognition/
```

#### 5.3 Jetson Inference: `src/inference/jetson_inference.py`
- Load ONNX model with ONNX Runtime
- Implement `preprocess_face` (manual normalization)
- Implement `recognize_face` with ONNX inference
- Implement `run` with FPS tracking

**Test on Jetson:**
```bash
python3 src/inference/jetson_inference.py
# Expected: 10+ FPS with face recognition
```

### 🔌 Phase 6: Arduino Integration

**Files to implement:**

#### 6.1 Arduino Sketch: `arduino/face_recognition_controller/face_recognition_controller.ino`
- Setup serial communication (9600 baud)
- Implement `loop` to read serial data
- Implement `processCommand` to parse messages
- Implement `handlePersonRecognized` for LED control
- Implement `handleCoordinates` for servo control
- Implement `handleUnknown` for warnings

**Upload:**
- Open in Arduino IDE
- Select board and port
- Upload sketch
- Test with Serial Monitor (send "PERSON:ben")

#### 6.2 Find Arduino Port: `tools/find_arduino.py`
- List all serial ports
- Identify Arduino device
- Print port information

**Run on Jetson:**
```bash
python3 tools/find_arduino.py
# Expected: Shows /dev/ttyUSB0 or /dev/ttyACM0
# Update configs/config.yaml with correct port
```

#### 6.3 Update Jetson Inference
- Add serial connection in `__init__`
- Implement `send_to_arduino` method
- Send commands after face recognition

**Test full pipeline:**
```bash
bash test_full_pipeline.sh
# Should check all components and run inference
```

## 🎯 Success Criteria

- ✅ Training completes without errors
- ✅ Validation accuracy >90%
- ✅ Webcam recognizes team members correctly
- ✅ ONNX model produces same results as PyTorch
- ✅ Jetson runs at 10+ FPS
- ✅ Arduino responds to recognition events

## 💡 Troubleshooting

**Dataset issues:**
- Check paths in `config.yaml`
- Verify 9 folders exist in `data/raw/Dataset/`
- Each folder should have ~100 images

**Training issues:**
- Start with SimplifiedLoss if ArcFace is complex
- Reduce batch size if GPU memory errors
- Check model architecture with `test_model.py`

**Inference issues:**
- Verify checkpoint exists: `models/checkpoints/best_model.pth`
- Check reference embeddings exist
- Test camera access: `ls /dev/video*`

**Arduino issues:**
- Find correct port with `tools/find_arduino.py`
- Match baud rate (9600) in both Python and Arduino
- Add user to dialout group: `sudo usermod -a -G dialout $USER`

## 📖 Additional Resources

- PyTorch tutorials: https://pytorch.org/tutorials/
- ArcFace paper: https://arxiv.org/abs/1801.07698
- ONNX documentation: https://onnx.ai/
- Arduino serial: https://www.arduino.cc/reference/en/language/functions/communication/serial/

---

**Ready to start?** Begin with Phase 1 and work through sequentially! 🚀

