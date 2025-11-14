# Implementation Reference Guide

This document provides specific code examples and patterns for implementing the new approach.

---

## Quick Reference: What Changes Where

| Component | Action | File | Complexity |
|-----------|--------|------|------------|
| Dependencies | Add insightface | `requirements.txt` | ⭐ Easy |
| Model Loading | New approach | `src/models/face_model.py` | ⭐⭐ Medium |
| Model Download | New file | `src/models/download_model.py` | ⭐ Easy |
| Embedding Gen | Simplify | `src/utils/generate_embeddings.py` | ⭐⭐ Medium |
| Webcam | Simplify loading | `src/inference/webcam_recognition.py` | ⭐ Easy |
| Config | Remove training | `configs/config.yaml` | ⭐ Easy |
| Dataset | Adapt usage | `src/data/dataset.py` | ⭐⭐ Medium |
| Augmentation | Remove training | `src/data/augmentation.py` | ⭐ Easy |
| ONNX Export | Update | `src/export/export_onnx.py` | ⭐⭐⭐ Hard |

---

## 1. Dependencies Update

### File: `requirements.txt`

```python
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Computer Vision
opencv-python>=4.8.0

# Face Recognition Models
insightface>=0.7.3  # NEW: Add this line
onnx>=1.14.0
onnxruntime>=1.15.0

# Data Augmentation
albumentations>=1.3.0

# Numerical Computing
numpy>=1.24.0

# Hardware Communication
pyserial>=3.5

# Configuration
pyyaml>=6.0

# Visualization & Metrics
matplotlib>=3.7.0
scikit-learn>=1.3.0

# Progress bars
tqdm>=4.65.0

# Image Processing
Pillow>=9.0.0
```

**Installation Command:**
```bash
pip install insightface>=0.7.3
```

**Platform Notes:**
- **Mac/Linux:** Should work out of the box
- **Windows:** May need Visual C++ Build Tools
- **All:** Test with: `python -c "import insightface; print(insightface.__version__)"`

---

## 2. Model Loading (NEW)

### File: `src/models/face_model.py` (NEW FILE)

```python
"""
Face embedding extraction using pretrained MobileFaceNet from InsightFace.
No training required - uses production-ready pretrained weights.
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEmbeddingModel:
    """
    Pretrained face recognition model.
    
    Architecture: MobileFaceNet (InsightFace)
    Embedding Dimension: 512
    Training Data: Millions of face images (pretrained)
    
    Args:
        model_name (str): InsightFace model pack name (default: 'buffalo_l')
        ctx_id (int): Device ID (-1 for CPU, 0+ for GPU)
    """
    
    def __init__(self, model_name='buffalo_l', ctx_id=-1):
        """
        Initialize face embedding model.
        
        Example:
            model = FaceEmbeddingModel()  # CPU mode
            model = FaceEmbeddingModel(ctx_id=0)  # GPU mode
        """
        print(f"Loading {model_name} model...")
        
        # TODO: Initialize InsightFace FaceAnalysis app
        # This loads the pretrained model from InsightFace model zoo
        # Hint: self.app = FaceAnalysis(name=model_name)
        #       self.app.prepare(ctx_id=ctx_id)
        
        print("✅ Model loaded successfully")
    
    def extract_embedding(self, face_img):
        """
        Extract 512-dimensional embedding from face image.
        
        Args:
            face_img (np.ndarray): Face image in BGR format [H, W, 3]
                                   (OpenCV default format)
        
        Returns:
            embedding (np.ndarray): L2-normalized embedding [512]
                                   Returns None if no face detected
        
        Example:
            >>> face = cv2.imread('person.jpg')
            >>> embedding = model.extract_embedding(face)
            >>> print(embedding.shape)  # (512,)
            >>> print(np.linalg.norm(embedding))  # ~1.0 (normalized)
        """
        # TODO: Convert BGR to RGB (InsightFace expects RGB)
        # Hint: face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # TODO: Detect face and extract embedding
        # Hint: faces = self.app.get(face_rgb)
        #       if len(faces) > 0:
        #           embedding = faces[0].embedding  # Already normalized
        #           return embedding
        
        # TODO: Return None if no face detected
        pass
    
    def extract_embedding_batch(self, face_imgs):
        """
        Extract embeddings from multiple face images.
        
        Args:
            face_imgs (list): List of face images [N x [H, W, 3]]
        
        Returns:
            embeddings (np.ndarray): Embeddings [N, 512]
        
        Example:
            >>> faces = [cv2.imread(f'person_{i}.jpg') for i in range(5)]
            >>> embeddings = model.extract_embedding_batch(faces)
            >>> print(embeddings.shape)  # (5, 512)
        """
        embeddings = []
        for face_img in face_imgs:
            emb = self.extract_embedding(face_img)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            return None
        
        return np.array(embeddings)


def test_model():
    """Test model loading and embedding extraction"""
    print("Testing FaceEmbeddingModel...")
    
    # Create dummy face image
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Initialize model
    model = FaceEmbeddingModel()
    
    # Extract embedding
    embedding = model.extract_embedding(dummy_face)
    
    if embedding is not None:
        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Embedding norm: {np.linalg.norm(embedding):.4f}")
        print("✅ Test passed!")
    else:
        print("⚠️  No face detected in dummy image (expected)")


if __name__ == '__main__':
    test_model()
```

**Key Changes from Old Approach:**
- ❌ No ResNet-18 loading
- ❌ No freezing layers
- ❌ No custom embedding layer
- ❌ No ArcFace head
- ✅ Simple: Just load pretrained model and extract embeddings

---

## 3. Model Download Helper (NEW)

### File: `src/models/download_model.py` (NEW FILE)

```python
"""
Download and verify InsightFace pretrained models.
"""
import os
from pathlib import Path


def download_insightface_model(model_name='buffalo_l', model_dir='models/pretrained'):
    """
    Download InsightFace model pack.
    
    Args:
        model_name (str): Model pack name ('buffalo_l', 'buffalo_s', etc.)
        model_dir (str): Directory to save model
    
    Returns:
        model_path (str): Path to downloaded model
    """
    print(f"Downloading {model_name} model...")
    
    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # InsightFace will automatically download on first use
    # Just importing and initializing triggers download
    from insightface.app import FaceAnalysis
    
    app = FaceAnalysis(name=model_name, root=model_dir)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    print(f"✅ Model downloaded to {model_dir}")
    print(f"✅ Model ready to use")
    
    return model_dir


def verify_model(model_dir='models/pretrained'):
    """Verify model exists and is loadable"""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(root=model_dir)
        app.prepare(ctx_id=-1)
        print("✅ Model verification successful")
        return True
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False


if __name__ == '__main__':
    # Download default model
    download_insightface_model()
    
    # Verify it works
    verify_model()
```

---

## 4. Configuration Update

### File: `configs/config.yaml`

```yaml
# Facial Recognition Configuration (NEW APPROACH)

# Data Configuration
data:
  dataset_path: "data/raw/Dataset"
  image_size: 112  # MobileFaceNet uses 112x112 (was 224)
  num_classes: 9

# Model Configuration (SIMPLIFIED)
model:
  provider: "insightface"
  model_name: "buffalo_l"  # InsightFace model pack
  embedding_dim: 512
  model_dir: "models/pretrained"

# Removed sections:
# - freeze_backbone (no training)
# - arcface parameters (no training)
# - training parameters (no training)
# - optimizer parameters (no training)

# Paths
paths:
  model_dir: "models/pretrained"
  reference_embeddings: "models/reference_embeddings.npy"
  label_names: "models/label_names.txt"
  export_dir: "models/exported"
  logs_dir: "logs"

# Inference Configuration
inference:
  similarity_threshold: 0.6  # Cosine similarity threshold [0, 1]
  confidence_threshold: 0.7   # Face detection confidence
  
# Hardware Configuration
hardware:
  device: "cpu"  # "cpu" or "cuda"
  arduino_port: "/dev/ttyUSB0"
  arduino_baud_rate: 9600

# Face Detection (YuNet - unchanged)
face_detection:
  model_path: "assets/face_detection_yunet_2023mar.onnx"
  input_size: [320, 320]
  score_threshold: 0.7
  nms_threshold: 0.3
```

**What Changed:**
- ✅ Removed all training-related parameters
- ✅ Simplified model config
- ✅ Added InsightFace-specific settings
- ✅ Kept inference and hardware config

---

## 5. Embedding Generation (SIMPLIFIED)

### File: `src/utils/generate_embeddings.py`

```python
"""
Generate reference embeddings for all people in dataset.
Uses pretrained MobileFaceNet - NO TRAINING REQUIRED.
"""
import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

from src.models.face_model import FaceEmbeddingModel


def generate_reference_embeddings(config_path='configs/config.yaml', 
                                   num_images_per_person=10):
    """
    Extract reference embeddings for each person.
    
    Strategy:
        1. Load pretrained model (no training!)
        2. For each person in dataset:
           - Load multiple images
           - Extract embeddings
           - Average them
           - Store as reference
    
    Args:
        config_path (str): Path to config file
        num_images_per_person (int): How many images to average
    """
    print("="*70)
    print("Generating Reference Embeddings")
    print("="*70)
    
    # TODO: Load configuration
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    
    # TODO: Initialize pretrained model
    # model = FaceEmbeddingModel(
    #     model_name=config['model']['model_name'],
    #     ctx_id=-1  # CPU mode
    # )
    
    # TODO: Get dataset path
    # dataset_path = Path(config['data']['dataset_path'])
    
    # TODO: Find all person folders
    # person_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    
    reference_embeddings = {}
    label_names = []
    
    # TODO: For each person
    # for person_folder in tqdm(person_folders, desc="Processing people"):
    #     person_name = person_folder.name
    #     label_names.append(person_name)
    #     
    #     # Get all images for this person
    #     image_files = list(person_folder.glob('*.png'))[:num_images_per_person]
    #     
    #     person_embeddings = []
    #     for img_path in image_files:
    #         # Load image
    #         img = cv2.imread(str(img_path))
    #         
    #         # Extract embedding
    #         embedding = model.extract_embedding(img)
    #         
    #         if embedding is not None:
    #             person_embeddings.append(embedding)
    #     
    #     # Average embeddings for this person
    #     if len(person_embeddings) > 0:
    #         avg_embedding = np.mean(person_embeddings, axis=0)
    #         
    #         # Re-normalize (important!)
    #         avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    #         
    #         reference_embeddings[person_name] = avg_embedding
    
    # TODO: Convert to array [num_people, 512]
    # embeddings_array = np.array([reference_embeddings[name] for name in label_names])
    
    # TODO: Save reference embeddings
    # output_path = config['paths']['reference_embeddings']
    # np.save(output_path, embeddings_array)
    
    # TODO: Save label names
    # labels_path = config['paths']['label_names']
    # with open(labels_path, 'w') as f:
    #     for name in label_names:
    #         f.write(f"{name}\n")
    
    print("\n✅ Reference embeddings generated!")
    # print(f"✅ Saved to: {output_path}")
    # print(f"✅ Shape: {embeddings_array.shape}")
    # print(f"✅ People: {label_names}")


if __name__ == '__main__':
    generate_reference_embeddings()
```

**Complexity Comparison:**
- **Old:** Load checkpoint → complex state dict loading → model setup
- **New:** Just `FaceEmbeddingModel()` - that's it!

---

## 6. Webcam Inference (SIMPLIFIED)

### File: `src/inference/webcam_recognition.py`

```python
"""
Real-time face recognition using webcam.
Combines YuNet detection with InsightFace MobileFaceNet embeddings.
"""
import cv2
import numpy as np
import yaml
from pathlib import Path

from src.models.face_model import FaceEmbeddingModel


class FaceRecognizer:
    """Real-time face recognizer"""
    
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize recognizer"""
        
        # TODO: Load config
        # with open(config_path, 'r') as f:
        #     self.config = yaml.safe_load(f)
        
        print("Loading model...")
        
        # TODO: Load pretrained model (MUCH SIMPLER!)
        # self.model = FaceEmbeddingModel(
        #     model_name=self.config['model']['model_name']
        # )
        
        # TODO: Load reference embeddings
        # embeddings_path = self.config['paths']['reference_embeddings']
        # self.reference_embeddings = np.load(embeddings_path)
        
        # TODO: Load label names
        # labels_path = self.config['paths']['label_names']
        # with open(labels_path, 'r') as f:
        #     self.label_names = [line.strip() for line in f]
        
        # TODO: Initialize YuNet face detector
        # detector_config = self.config['face_detection']
        # self.detector = cv2.FaceDetectorYN.create(
        #     model=detector_config['model_path'],
        #     config="",
        #     input_size=tuple(detector_config['input_size']),
        #     score_threshold=detector_config['score_threshold'],
        #     nms_threshold=detector_config['nms_threshold']
        # )
        
        # TODO: Get similarity threshold
        # self.similarity_threshold = self.config['inference']['similarity_threshold']
        
        print("✅ Recognizer ready")
    
    def recognize_face(self, face_img):
        """
        Recognize a face by comparing embedding to references.
        
        Args:
            face_img: Face image (BGR format)
        
        Returns:
            name (str): Person name or "Unknown"
            similarity (float): Similarity score [0, 1]
        """
        # TODO: Extract embedding from face
        # embedding = self.model.extract_embedding(face_img)
        # if embedding is None:
        #     return "Unknown", 0.0
        
        # TODO: Compute cosine similarities with all references
        # similarities = embedding @ self.reference_embeddings.T
        # This gives similarity with each reference person
        
        # TODO: Find best match
        # max_idx = np.argmax(similarities)
        # max_similarity = similarities[max_idx]
        
        # TODO: Check threshold
        # if max_similarity >= self.similarity_threshold:
        #     return self.label_names[max_idx], max_similarity
        # else:
        #     return "Unknown", max_similarity
        
        pass
    
    def run_webcam(self, camera_id=0):
        """Run real-time recognition on webcam"""
        # TODO: Implementation (similar to old approach)
        # Main difference: model loading is simpler
        pass


def main():
    """Main entry point"""
    # TODO: Create recognizer
    # recognizer = FaceRecognizer()
    
    # TODO: Run webcam
    # recognizer.run_webcam()
    
    pass


if __name__ == '__main__':
    main()
```

**Key Difference:**
```python
# OLD (complex):
checkpoint = torch.load('models/checkpoints/best_model.pth')
model = ResNetArcFace(num_classes=9, embedding_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# NEW (simple):
model = FaceEmbeddingModel()
```

---

## 7. Dataset Adaptation (PRESERVE STUDENT CODE)

### File: `src/data/dataset.py`

```python
"""
Dataset utilities for face recognition.
In the NEW approach, this is used for embedding generation, not training.
"""
import os
import cv2
from pathlib import Path


class FaceDataset:
    """
    Simple face dataset loader.
    Used to iterate through images for embedding generation.
    """
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        
        # PRESERVE STUDENT CODE - This is what they wrote:
        self.image_paths = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        
        # PRESERVE STUDENT CODE - Folder iteration
        dataset_path = self.root_dir
        names = sorted([folder.name.lower() 
                 for folder in dataset_path.iterdir() 
                 if folder.is_dir()])
        
        for id, name in enumerate(names):
            self.label_to_name[id] = name
            self.name_to_label[name] = id
              
        for person in names:
            folder = dataset_path / person
            image_paths_person = [
                str(img)
                for img in folder.glob("*.png")
            ]
            self.image_paths.extend(image_paths_person)

            for i in range(len(image_paths_person)):
                self.labels.append(self.name_to_label[person])
        
        print(f"Dataset: {len(self.image_paths)} images, {len(self.label_to_name)} people")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load single image"""
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        label = self.labels[idx]
        return image, label
    
    def get_images_by_person(self, person_name, max_images=10):
        """
        Get all images for a specific person.
        Useful for embedding generation.
        """
        label_id = self.name_to_label[person_name]
        indices = [i for i, l in enumerate(self.labels) if l == label_id]
        
        images = []
        for idx in indices[:max_images]:
            img, _ = self.__getitem__(idx)
            images.append(img)
        
        return images
    
    def get_all_people(self):
        """Get list of all person names"""
        return list(self.name_to_label.keys())
```

**Changes:**
- ✅ Preserved student's data loading logic
- ✅ Removed DataLoader/batching (not needed for inference)
- ✅ Added helper methods for embedding generation
- ✅ Removed augmentation transforms (handled differently now)

---

## 8. Augmentation Simplification

### File: `src/data/augmentation.py`

```python
"""
Image preprocessing for inference.
In NEW approach, only validation transforms needed (no training).
"""
import cv2
import numpy as np


def preprocess_face(face_img, target_size=(112, 112)):
    """
    Preprocess face image for MobileFaceNet.
    
    Args:
        face_img (np.ndarray): BGR face image
        target_size (tuple): Output size (default: 112x112 for MobileFaceNet)
    
    Returns:
        preprocessed (np.ndarray): Preprocessed image
    """
    # Resize
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    face_normalized = face_rgb.astype(np.float32) / 255.0
    
    # InsightFace models expect RGB in [0, 255]
    # Or normalized depends on model - check documentation
    
    return face_rgb  # Return RGB


def preprocess_batch(face_imgs, target_size=(112, 112)):
    """Preprocess batch of faces"""
    return [preprocess_face(img, target_size) for img in face_imgs]
```

**Changes:**
- ✅ Removed all training augmentations (flip, brightness, etc.)
- ✅ Only keep resize and normalization
- ✅ Much simpler!

---

## 9. ONNX Export (OPTIONAL)

### File: `src/export/export_onnx.py`

```python
"""
Export InsightFace model to ONNX for Jetson deployment.

Note: InsightFace models can be exported to ONNX, but it's more complex
than the old PyTorch export. For Jetson, you might:
1. Use InsightFace Python directly (easier)
2. Export to ONNX (better performance with TensorRT)
"""
import onnx
import numpy as np


def export_insightface_to_onnx(output_path='models/exported/mobilefacenet.onnx'):
    """
    Export InsightFace model to ONNX.
    
    This is more complex than PyTorch export because InsightFace
    uses custom model formats. Options:
    
    Option 1: Use InsightFace ONNX models directly
    - InsightFace provides pre-exported ONNX models
    - Download from their model zoo
    
    Option 2: Export yourself (advanced)
    - Requires understanding InsightFace internals
    """
    print("InsightFace ONNX export:")
    print("  Option 1: Download pre-exported ONNX from InsightFace model zoo")
    print("  Option 2: Use InsightFace Python API on Jetson (simpler)")
    
    # TODO: For students, recommend Option 2 (use Python API directly)
    # This avoids ONNX export complexity
    
    pass


def test_onnx_inference(onnx_path):
    """Test ONNX model inference"""
    import onnxruntime as ort
    
    # TODO: Load and test ONNX model
    pass


if __name__ == '__main__':
    print("⚠️  ONNX export for InsightFace is optional")
    print("✅ Recommended: Use InsightFace Python API on Jetson directly")
```

**Changes:**
- ✅ Simplified - recommend using Python API on Jetson
- ✅ ONNX export is optional (not required)
- ✅ Pre-exported ONNX models available from InsightFace

---

## 10. Test Scripts

### File: `test_embedding_extraction.py` (NEW)

```python
"""
Test embedding extraction with pretrained model.
"""
import numpy as np
import cv2
from src.models.face_model import FaceEmbeddingModel


def test_embedding_extraction():
    """Test basic embedding extraction"""
    print("="*70)
    print("Testing Embedding Extraction")
    print("="*70)
    
    # Initialize model
    print("\n1. Loading model...")
    model = FaceEmbeddingModel()
    print("   ✅ Model loaded")
    
    # Test with single image
    print("\n2. Testing single image...")
    test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    embedding = model.extract_embedding(test_img)
    
    if embedding is not None:
        print(f"   ✅ Embedding shape: {embedding.shape}")
        print(f"   ✅ Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"   ✅ Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    else:
        print("   ⚠️  No face detected (expected for random image)")
    
    # Test with batch
    print("\n3. Testing batch extraction...")
    test_imgs = [test_img for _ in range(5)]
    embeddings = model.extract_embedding_batch(test_imgs)
    
    if embeddings is not None:
        print(f"   ✅ Batch shape: {embeddings.shape}")
    
    # Test similarity computation
    print("\n4. Testing similarity computation...")
    emb1 = np.random.randn(512)
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = np.random.randn(512)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    similarity = np.dot(emb1, emb2)
    print(f"   ✅ Random embeddings similarity: {similarity:.4f}")
    print(f"   ✅ Same embedding similarity: {np.dot(emb1, emb1):.4f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == '__main__':
    test_embedding_extraction()
```

---

## Common Implementation Patterns

### Pattern 1: Loading Pretrained Model

```python
# ALWAYS DO THIS:
from src.models.face_model import FaceEmbeddingModel

model = FaceEmbeddingModel(
    model_name='buffalo_l',  # or 'buffalo_s' for faster
    ctx_id=-1  # CPU mode, use 0 for GPU
)

# NEVER DO THIS (old approach):
# checkpoint = torch.load(...)
# model = ResNetArcFace(...)
# model.load_state_dict(...)
```

### Pattern 2: Extracting Embeddings

```python
# For single image:
face_img = cv2.imread('face.jpg')  # BGR format
embedding = model.extract_embedding(face_img)  # Returns [512] or None

# For multiple images:
face_imgs = [cv2.imread(f'face_{i}.jpg') for i in range(5)]
embeddings = model.extract_embedding_batch(face_imgs)  # Returns [N, 512]
```

### Pattern 3: Computing Similarities

```python
# Cosine similarity between two embeddings:
similarity = np.dot(emb1, emb2)  # Since embeddings are normalized

# Similarity with reference database:
reference_db = np.load('reference_embeddings.npy')  # [num_people, 512]
query_emb = model.extract_embedding(face_img)  # [512]

similarities = query_emb @ reference_db.T  # [num_people]
best_match = np.argmax(similarities)
best_score = similarities[best_match]
```

### Pattern 4: Threshold-Based Recognition

```python
SIMILARITY_THRESHOLD = 0.6  # Tune based on your data

if best_score >= SIMILARITY_THRESHOLD:
    person_name = label_names[best_match]
    print(f"Recognized: {person_name} (confidence: {best_score:.2f})")
else:
    print(f"Unknown person (best match: {best_score:.2f})")
```

---

## Debugging Tips

### Issue 1: Model Download Fails

```python
# Manual download:
from insightface.app import FaceAnalysis
import os

os.environ['INSIGHTFACE_HOME'] = './models/pretrained'
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)
```

### Issue 2: No Face Detected

```python
# Check image format:
print(face_img.shape)  # Should be (H, W, 3)
print(face_img.dtype)  # Should be uint8
print(face_img.min(), face_img.max())  # Should be [0, 255]

# Try preprocessing:
face_img = cv2.resize(face_img, (112, 112))
face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
```

### Issue 3: Low Similarity Scores

```python
# Check embedding normalization:
print(np.linalg.norm(embedding))  # Should be ~1.0

# Check reference embeddings:
print(reference_embeddings.shape)  # Should be [num_people, 512]
print(np.linalg.norm(reference_embeddings, axis=1))  # All should be ~1.0
```

---

## Next Steps

1. **Start with dependencies**: Update `requirements.txt` and test installation
2. **Create model loading**: Implement `src/models/face_model.py`
3. **Test model loading**: Run test script to verify
4. **Update embedding generation**: Simplify to use pretrained model
5. **Update inference**: Simplify model loading
6. **Update documentation**: Reflect new workflow

---

## Need Help?

- See `TRANSITION_PLAN.md` for detailed migration steps
- See `BEFORE_AFTER_COMPARISON.md` for side-by-side comparison
- See `ANALYSIS_SUMMARY.md` for high-level overview

**Ready to implement?** Start with File #1 (dependencies) and work your way through!

