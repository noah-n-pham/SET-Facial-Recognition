# Emotion Recognition Learning Guide (Semester 2)

**Time to Complete:** 3-4 hours  
**What You'll Build:** Real-time emotion classification running parallel to face recognition  
**Learning Method:** Implement code step-by-step by filling in TODOs

---

## Overview

### What You'll Learn

1. **Emotion Classification** - Classifying faces into 7 emotion categories
2. **ONNX Runtime** - Loading and running pretrained models directly
3. **Softmax Function** - Converting raw scores to probabilities
4. **ImageNet Normalization** - Preprocessing for transfer learning models
5. **Temporal Smoothing** - Reducing noise in real-time predictions
6. **Parallel Pipelines** - Running two models on the same input

### The Big Picture

```
Camera → Face Detection → Crop Face → ┬→ Identity (Semester 1)  → Name
         (YuNet)                       │   (MobileFaceNet)
                                       │
                                       └→ Emotion (Semester 2)  → Emotion
                                           (MobileNet)
                                                   ↓
                                           Display: "Ben | Happy"
```

**Key Insight:** Same face crop, two different analyses running in parallel!

---

## Core Concepts (Read This First!)

### Concept 5: Classification vs Verification

In Semester 1, you built a **verification** system:
- "Is this face the same person as the reference?"
- Open-set: can reject unknown people
- Method: Cosine similarity between embeddings

In Semester 2, you're building a **classification** system:
- "Which of these 7 emotions does this face show?"
- Closed-set: must pick one of the known classes
- Method: Softmax over class logits

```
VERIFICATION (Semester 1):              CLASSIFICATION (Semester 2):
┌─────────────┐                         ┌─────────────┐
│  Face Crop  │                         │  Face Crop  │
└──────┬──────┘                         └──────┬──────┘
       ↓                                       ↓
┌─────────────┐                         ┌─────────────┐
│  Embedding  │ [512 numbers]           │   Logits    │ [7 numbers]
└──────┬──────┘                         └──────┬──────┘
       ↓                                       ↓
┌─────────────┐                         ┌─────────────┐
│   Cosine    │ Compare with refs       │   Softmax   │ Convert to probs
└──────┬──────┘                         └──────┬──────┘
       ↓                                       ↓
   Similarity                              Probabilities
   0.0 to 1.0                              Sum to 1.0
       ↓                                       ↓
 "Ben" or "Unknown"                     "Happiness" (87%)
```

### Concept 6: The Softmax Function

Softmax converts raw model outputs (logits) into probabilities.

**Mathematical Definition:**
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**Example:**
```python
logits = [2.0, 1.0, 0.1]        # Raw scores (can be any value)
# exp:   [7.4, 2.7, 1.1]        # Always positive
# sum:   11.2
# prob:  [0.66, 0.24, 0.10]     # Sum to 1.0

# Interpretation: 66% class 0, 24% class 1, 10% class 2
```

**Why Softmax?**
- Converts arbitrary numbers to probabilities
- Preserves ranking (highest logit → highest probability)
- Probabilities sum to 1.0 (valid probability distribution)
- Amplifies differences (winner takes more)

**Numerical Stability:**
```python
# WRONG - can overflow!
def bad_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))  # exp(1000) = infinity!

# RIGHT - subtract max first
def softmax(x):
    shifted = x - np.max(x)  # Now max value is 0
    exp_x = np.exp(shifted)  # exp(0) = 1, no overflow
    return exp_x / np.sum(exp_x)
```

### Concept 7: Temporal Smoothing

Emotion predictions are noisy frame-to-frame:
```
Frame 1: Happy (78%)
Frame 2: Neutral (52%)   ← Brief glitch
Frame 3: Happy (81%)
Frame 4: Happy (75%)
Frame 5: Surprise (45%)  ← Another glitch
```

Without smoothing, the display flickers annoyingly. With smoothing:
```
Smoothed (5-frame average): Happy (72%)  ← Stable!
```

**Implementation: Circular Buffer**
```
Buffer size = 5
         [Frame1, Frame2, Frame3, Frame4, Frame5]
              ↑
           Position (wraps around)

When buffer is full, new entries overwrite oldest.
Average all entries to get smoothed prediction.
```

### Concept 8: ImageNet Normalization

MobileNet was trained on ImageNet with specific preprocessing:
```python
mean = [0.485, 0.456, 0.406]  # RGB channel means
std  = [0.229, 0.224, 0.225]  # RGB channel stds

normalized = (image / 255.0 - mean) / std
```

**Why normalize?**
- Model expects input in same format as training data
- Without it: garbage in, garbage out
- Like using Celsius data with a Fahrenheit model

**Contrast with Semester 1:**
- InsightFace handled normalization internally
- Now YOU must do it explicitly
- Same concept, explicit implementation

---

## Setup (15 minutes)

### Step 1: Download the Emotion Model

```bash
# Download MobileNet 7-class emotion model
curl -L -o assets/mobilenet_7.onnx \
  "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/mobilenet_7.onnx?raw=true"
```

Or manually download from:
- URL: https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/mobilenet_7.onnx
- Save to: `assets/mobilenet_7.onnx`

### Step 2: Verify Download

```bash
ls -lh assets/mobilenet_7.onnx
# Should show ~13 MB file
```

### Step 3: Verify Dependencies

```bash
python -c "import onnxruntime; print('✅ ONNX Runtime:', onnxruntime.__version__)"
```

ONNX Runtime was already installed in Semester 1 (used by InsightFace).

### Step 4: Understand the File Structure

```
Facial-Recognition/
├── models/
│   ├── face_model.py              # Semester 1: Identity
│   └── emotion_model.py           # Semester 2: Emotion (NEW)
│
├── utils/
│   ├── face_detector.py           # Semester 1: Detection
│   └── emotion_smoother.py        # Semester 2: Smoothing (NEW)
│
├── core/
│   └── face_recognizer.py         # Updated: Now includes emotion
│
└── assets/
    ├── face_detection_yunet_2023mar.onnx   # Semester 1
    └── mobilenet_7.onnx                     # Semester 2 (NEW)
```

---

## Phase 1: Emotion Model (1-2 hours)

### Concept: ONNX Runtime

ONNX (Open Neural Network Exchange) is a universal format for ML models.
- Models trained in PyTorch/TensorFlow can be exported to ONNX
- ONNX Runtime runs these models efficiently on any platform
- No need for PyTorch/TensorFlow at runtime!

**Why ONNX for Jetson Nano?**
- Lightweight runtime (~50MB vs ~2GB for PyTorch)
- Optimized for edge devices
- Same code works on CPU, GPU, or specialized hardware

### Implementation

**File:** `models/emotion_model.py`

#### TODO 14: Load ONNX Model

Navigate to `models/emotion_model.py` and find:

```python
def __init__(self, model_path='assets/mobilenet_7.onnx'):
    # TODO 14: Load ONNX model and validate output shape
```

**What to implement:**
1. Create ONNX Runtime InferenceSession
2. Get input name from model metadata
3. Validate output shape is exactly 7 (fail fast if wrong model)
4. Store session and input name

**Reasoning:**
- InferenceSession is the main ONNX Runtime interface
- Input name is needed to feed data to the model
- Validation prevents silent failures from wrong model files

**Hint:** Check the constants at the top of the file for NUM_CLASSES.

#### TODO 15: Implement Preprocessing

```python
def preprocess(self, face_img):
    # TODO 15: Implement preprocessing with ImageNet normalization
```

**What to implement:**
1. Resize to 224x224 (MobileNet input size)
2. Convert BGR to RGB (OpenCV → model format)
3. Scale to 0-1 range
4. Apply ImageNet normalization (subtract mean, divide by std)
5. Transpose HWC to CHW (height/width/channels → channels/height/width)
6. Add batch dimension

**Reasoning:**
- Model expects specific input format matching training data
- Wrong preprocessing = wrong predictions
- NCHW format is standard for ONNX models

#### TODO 16: Implement Softmax

```python
def softmax(self, logits):
    # TODO 16: Implement Softmax function manually
```

**What to implement:**
1. Subtract max for numerical stability
2. Compute exponentials
3. Normalize by sum

**Reasoning:**
- Converts raw scores to probabilities
- Subtracting max prevents overflow (exp of large numbers)
- This is the core of classification!

#### TODO 17: Run Inference

```python
def predict(self, face_img):
    # TODO 17: Run inference and return prediction
```

**What to implement:**
1. Preprocess image
2. Run ONNX session
3. Apply softmax to get probabilities
4. Find class with highest probability
5. Return (emotion_label, confidence)

**Reasoning:**
- This is the main prediction pipeline
- Combines preprocessing + inference + postprocessing
- Error handling prevents crashes on bad input

**Test your implementation:**
```bash
python models/emotion_model.py
# Expected: Model loads, test prediction works
```

---

## Phase 2: Smoothing Buffer (30-45 minutes)

### Concept: Circular Buffers

A circular buffer is a fixed-size array where new data overwrites the oldest:

```
Initial:     [_, _, _, _, _]  position=0
After 3:     [A, B, C, _, _]  position=3
After 5:     [A, B, C, D, E]  position=0 (wrapped!)
After 6:     [F, B, C, D, E]  position=1 (A overwritten)
```

**Why circular?**
- Fixed memory (important for embedded systems)
- O(1) insertions (no shifting)
- Perfect for sliding window operations

### Implementation

**File:** `utils/emotion_smoother.py`

#### TODO 18: Initialize Buffer

```python
def __init__(self, window_size=5, num_classes=7):
    # TODO 18: Initialize circular buffer
```

**What to implement:**
1. Store window_size and num_classes
2. Create buffer array of zeros: shape (window_size, num_classes)
3. Initialize position=0, count=0

#### TODO 19: Update Buffer

```python
def update(self, probabilities):
    # TODO 19: Add new prediction to buffer
```

**What to implement:**
1. Write probabilities at current position
2. Advance position (wrap with modulo)
3. Update count (cap at window_size)

#### TODO 20: Get Smoothed Result

```python
def get_smoothed(self):
    # TODO 20: Return averaged probabilities
```

**What to implement:**
1. Handle empty buffer (return uniform distribution)
2. Average the valid entries
3. Return averaged probabilities

**Test your implementation:**
```bash
python utils/emotion_smoother.py
# Expected: Smoothing tests pass
```

---

## Phase 3: Integration (1 hour)

### Concept: Parallel Pipelines

The face crop goes to TWO models simultaneously:

```python
cropped_face = frame[y1:y2, x1:x2]

# Path A: Identity (Semester 1) - WHO is this?
name, similarity = self.recognize_face(cropped_face)

# Path B: Emotion (Semester 2) - HOW do they feel?
emotion, confidence = self.recognize_emotion(cropped_face)

# Combine for display
label = f"{name} | {emotion}"
```

This is the power of modular design - same input, different analyses!

### Implementation

**File:** `core/face_recognizer.py`

#### TODO 21: Initialize Emotion Components

In `__init__`, after the Semester 1 initialization:

```python
# TODO 21: Initialize emotion model and smoother
```

**What to implement:**
1. Create EmotionModel instance
2. Create EmotionSmoother instance
3. Handle missing model file gracefully

#### TODO 22: Implement recognize_emotion()

New method parallel to recognize_face():

```python
def recognize_emotion(self, face_img):
    # TODO 22: Implement emotion recognition with smoothing
```

**What to implement:**
1. Get prediction from emotion model
2. Update smoother with probabilities
3. Return smoothed result

#### TODO 23: Update Display

In run_webcam0, after getting identity:

```python
# TODO 23: Add Path B (Emotion) and update display
```

**What to implement:**
1. Call recognize_emotion()
2. Update label to show both identity and emotion
3. Optionally adjust colors based on emotion

**Test your implementation:**
```bash
python core/face_recognizer.py
# Expected: Webcam shows "Name | Emotion" for each face
```

---

## Testing & Debugging

### Test 1: Emotion Model

```bash
python models/emotion_model.py
# Should: Load model, show test prediction
```

### Test 2: Smoother

```bash
python utils/emotion_smoother.py
# Should: Pass all smoothing tests
```

### Test 3: Full Integration

```bash
python core/face_recognizer.py
# Should: Show webcam with Name | Emotion labels
```

### Common Issues

**Issue 1: "Model file not found"**
- Download the model: `curl -L -o assets/mobilenet_7.onnx "URL"`
- Check file exists: `ls -lh assets/mobilenet_7.onnx`

**Issue 2: "Wrong number of classes"**
- You downloaded an 8-class model instead of 7-class
- Delete and re-download `mobilenet_7.onnx` specifically

**Issue 3: "Emotion flickering rapidly"**
- Increase smoothing window (default is 5)
- Check smoother is being updated every frame

**Issue 4: "All predictions are the same emotion"**
- Check preprocessing (normalization values)
- Verify BGR→RGB conversion
- Check input image isn't corrupted

---

## What You've Learned

### Technical Skills
- Loading and running ONNX models
- Implementing softmax for classification
- ImageNet-style preprocessing
- Circular buffers for temporal smoothing
- Parallel model inference

### Concepts
- Classification vs Verification
- Probability distributions from logits
- Why preprocessing must match training
- Temporal smoothing for stable output
- Modular pipeline design

### Industry Practices
- Using ONNX for portable models
- Defensive programming (fail fast validation)
- Edge deployment considerations
- Real-time system design

---

## Summary

You've extended the Semester 1 face recognition system with emotion classification:

1. **Phase 1:** Loaded MobileNet ONNX model for emotion classification
2. **Phase 2:** Implemented temporal smoothing for stable predictions
3. **Phase 3:** Integrated emotion into the real-time webcam loop

**Key insight:** The same face crop feeds TWO parallel pipelines - identity and emotion - demonstrating modular ML system design.

**Output format:** `"Ben (0.85) | Happiness (92%)"` - who they are AND how they feel!

---

## Appendix: Emotion Classes

The 7-class AffectNet model recognizes:

| Index | Emotion | Description |
|-------|---------|-------------|
| 0 | Anger | Furrowed brow, tight lips |
| 1 | Disgust | Wrinkled nose, raised upper lip |
| 2 | Fear | Wide eyes, raised eyebrows |
| 3 | Happiness | Smile, raised cheeks |
| 4 | Neutral | Relaxed face, no strong expression |
| 5 | Sadness | Downturned mouth, drooping eyelids |
| 6 | Surprise | Wide eyes, open mouth, raised eyebrows |

**Note:** "Contempt" (asymmetric lip raise) is NOT included in the 7-class model.
This is intentional - contempt is rare and often confused with other emotions.

---

## Next Steps

### Immediate Improvements
1. **Tune smoothing** - Adjust window size for your use case
2. **Add emotion colors** - Different box colors per emotion
3. **Log emotions** - Track emotional responses over time
4. **Multi-face tracking** - Separate smoother per tracked face

### Advanced Features
1. **Emotion transitions** - Detect when emotion changes
2. **Valence/Arousal** - 2D emotion space instead of categories
3. **Face attention** - Are they looking at the camera?
4. **Blink detection** - Combined with emotion for engagement

---

**Next:** Start with Phase 1 - open `models/emotion_model.py` and implement TODO 14!
