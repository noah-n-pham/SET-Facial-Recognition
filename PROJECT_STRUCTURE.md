# Project Structure Guide

This document explains the complete structure of the reorganized facial recognition system.

---

## 📁 Directory Structure

```
Facial-Recognition/
│
├── 📖 README.md                       # Project overview and quick start
├── 📘 LEARNING_GUIDE.md              # Complete step-by-step instruction guide
├── 📋 PROJECT_STRUCTURE.md           # This file
│
├── requirements.txt                   # Python dependencies
├── configs/
│   └── config.yaml                    # System configuration
│
├── models/                            # Phase 1: Load Pretrained Model
│   ├── __init__.py
│   └── face_model.py                  # MobileFaceNet wrapper (3 TODOs)
│
├── data/                              # Phase 2A: Data Collection
│   ├── face_capture.py                # Webcam photo capture (2 TODOs)
│   └── raw/
│       └── Dataset/                   # Collected photos go here
│           ├── person1/
│           ├── person2/
│           └── ...
│
├── utils/                             # Helper Modules
│   ├── __init__.py
│   └── face_detector.py               # YuNet detector wrapper (2 TODOs)
│
├── core/                              # Core Recognition System
│   ├── __init__.py
│   ├── generate_embeddings.py         # Phase 2B: Build reference database (3 TODOs)
│   └── face_recognizer.py             # Phase 3: Real-time recognition (4 TODOs)
│
├── deployment/                        # Phase 4: Hardware Deployment (Optional)
│   ├── __init__.py
│   └── jetson_inference.py            # Jetson Nano deployment
│
├── arduino/                           # Arduino Integration (Optional)
│   └── face_recognition_controller/
│       └── face_recognition_controller.ino
│
├── assets/                            # Pre-trained Models
│   └── face_detection_yunet_2023mar.onnx
│
└── archive/                           # Archived Files (for reference only)
    ├── old_training_approach/         # Old ResNet-18 training files
    └── ...                            # Old planning documents
```

---

## 📚 Key Files Explained

### Documentation Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `README.md` | 5-minute overview | Start here |
| `LEARNING_GUIDE.md` | Complete step-by-step guide | Main learning document |
| `PROJECT_STRUCTURE.md` | This file - structure reference | When navigating codebase |

### Implementation Files (in order of learning)

| Phase | File | TODOs | Purpose |
|-------|------|-------|---------|
| **Phase 1** | `models/face_model.py` | 3 | Load and use pretrained MobileFaceNet |
| **Phase 2A** | `data/face_capture.py` | 2 | Capture team photos with webcam |
| **Phase 2A** | `utils/face_detector.py` | 2 | YuNet face detection wrapper |
| **Phase 2B** | `core/generate_embeddings.py` | 3 | Build reference embedding database |
| **Phase 3** | `core/face_recognizer.py` | 4 | Real-time face recognition system |
| **Phase 4** | `deployment/jetson_inference.py` | - | Deploy to Jetson Nano (optional) |

**Total: 14 TODOs** (down from 153 in old approach!)

---

## 🎯 Learning Path

### Phase 1: Load Pretrained Model (30-45 min)
1. Read LEARNING_GUIDE.md → Phase 1
2. Open `models/face_model.py`
3. Implement TODOs 1-3
4. Test: `python models/face_model.py`

### Phase 2A: Collect Photos (30 min + photo time)
1. Read LEARNING_GUIDE.md → Phase 2A
2. Open `utils/face_detector.py`
3. Implement TODOs 4-5
4. Test: `python utils/face_detector.py`
5. Open `data/face_capture.py`
6. Implement TODOs 6-7
7. Run: `python data/face_capture.py` to collect photos

### Phase 2B: Build Reference Database (30-45 min)
1. Read LEARNING_GUIDE.md → Phase 2B
2. Open `core/generate_embeddings.py`
3. Implement TODOs 8-10
4. Run: `python core/generate_embeddings.py`

### Phase 3: Real-Time Recognition (2-3 hours)
1. Read LEARNING_GUIDE.md → Phase 3
2. Open `core/face_recognizer.py`
3. Implement TODOs 11-13
4. Run: `python core/face_recognizer.py`

### Phase 4: Deploy to Hardware (Optional, 2-3 hours)
1. Read LEARNING_GUIDE.md → Phase 4
2. Deploy to Jetson Nano
3. Integrate with Arduino

---

## 🔧 Configuration

All settings are in `configs/config.yaml`:

```yaml
# Key settings you might want to adjust:

model:
  model_name: "buffalo_l"              # or "buffalo_s" for faster
  device: "cpu"                        # or "cuda" if you have GPU

recognition:
  similarity_threshold: 0.6            # 0.5-0.7 range recommended

camera:
  device_id: 0                         # Change if you have multiple cameras
```

---

## 📊 File Dependencies

```
models/face_model.py (standalone - no dependencies on other project files)
    ↓
data/face_capture.py (uses face_detector.py)
utils/face_detector.py (standalone)
    ↓
core/generate_embeddings.py (uses face_model.py)
    ↓
core/face_recognizer.py (uses face_model.py + face_detector.py)
    ↓
deployment/jetson_inference.py (uses all above)
```

**Key Insight:** Files are designed to be implemented in order, with each building on previous ones.

---

## 🧪 Testing

Each file has a `if __name__ == '__main__':` section for testing:

```bash
# Test each file individually:
python models/face_model.py           # Test model loading
python utils/face_detector.py         # Test face detection
python data/face_capture.py           # Capture photos
python core/generate_embeddings.py    # Generate database
python core/face_recognizer.py        # Run recognition
```

---

## 📦 Generated Files

After completing the phases, these files will be created:

```
data/raw/Dataset/
    person1/
        person1_0.png
        person1_1.png
        ...
    person2/
        ...

models/
    reference_embeddings.npy           # [num_people, 512] array
    label_names.txt                    # List of names
```

These are the **reference database** used for recognition.

---

## 🎓 What Each TODO Teaches

### Phase 1 TODOs (Pretrained Models)
- **TODO 1:** How to use InsightFace library
- **TODO 2:** Importance of color space (BGR vs RGB)
- **TODO 3:** Extracting embeddings from faces

### Phase 2 TODOs (Data Collection & Processing)
- **TODO 4-5:** Real-time face detection with YuNet
- **TODO 6-7:** Building data collection tools
- **TODO 8-10:** Processing images and building databases

### Phase 3 TODOs (Recognition System)
- **TODO 11:** Loading models and databases
- **TODO 12:** Similarity-based recognition
- **TODO 13:** Real-time video processing

---

## 🚀 Quick Start Checklist

- [ ] Read `README.md` (5 min)
- [ ] Read `LEARNING_GUIDE.md` → Phase 1 (15 min)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Implement Phase 1 (30-45 min)
- [ ] Implement Phase 2A (30 min)
- [ ] Collect photos (15 min per person)
- [ ] Implement Phase 2B (30-45 min)
- [ ] Implement Phase 3 (2-3 hours)
- [ ] (Optional) Phase 4: Deploy to Jetson

**Total Time: 6-8 hours**

---

## 💡 Key Differences from Old Approach

### What Changed

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| Model | Train ResNet-18 + ArcFace | Use pretrained MobileFaceNet |
| Training | Required (5-10 min) | Not needed |
| Complexity | 153 TODOs | 14 TODOs (-91%) |
| Time | 12-15 hours | 6-8 hours (-47%) |
| Documentation | 9 separate files | 1 main guide |
| Files | Complex `src/` structure | Flat, numbered structure |

### What Stayed the Same

- ✅ Face detection (YuNet)
- ✅ Embeddings concept
- ✅ Cosine similarity
- ✅ Real-time recognition
- ✅ Hardware deployment
- ✅ Arduino integration

---

## 📝 Notes for Instructors

### Educational Value
- Students learn modern AI development (using pretrained models)
- Focus on practical system building, not training internals
- Industry-relevant skills
- Less overwhelming, higher completion rate

### Customization
- Can adjust similarity threshold in `config.yaml`
- Can swap different InsightFace models
- Can add more features (age detection, emotion, etc.)

### Troubleshooting
- All common issues documented in LEARNING_GUIDE.md
- Each file has extensive comments explaining concepts
- Test sections provide immediate feedback

---

## 🎯 Success Metrics

After completing all phases, students should have:

✅ Working face recognition system  
✅ Understanding of embeddings and similarity  
✅ Experience with pretrained models  
✅ Real-time computer vision skills  
✅ Hardware deployment knowledge  

---

**Ready to start? Open `README.md` then `LEARNING_GUIDE.md`!**

