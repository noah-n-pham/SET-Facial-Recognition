"""
Face Recognition Model - Phase 1

Wraps InsightFace's pretrained MobileFaceNet model for easy embedding extraction.
No training needed - we use a model already trained on millions of faces.

Time: 30-45 minutes
TODOs: 3
"""

import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEmbeddingModel:
    """
    Loads pretrained MobileFaceNet and extracts face embeddings.
    
    An embedding is a 512-number representation of a face.
    Similar faces → similar embeddings (compared using dot product).
    """
    
    def __init__(self, model_name='buffalo_l', device='cpu'):
        """Initialize pretrained face recognition model."""
        
        print(f"Loading {model_name} model on {device}...")
        
        # TODO 1: Initialize InsightFace FaceAnalysis model
        # ==================================================
        # Import FaceAnalysis from insightface.app
        # Create a FaceAnalysis instance with name=model_name
        # Call .prepare() with ctx_id (-1 for CPU, 0 for GPU) and det_size=(224, 224)
        # Store the instance as self.app
        #
        # Hints:
        # - Check if device == 'cpu' to determine ctx_id
        # - First run will download model files (~100MB)
        # - See InsightFace documentation for FaceAnalysis usage
        
        ctx_id = -1 if device == 'cpu' else 0
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=(224, 224))
        
        print("✅ Model loaded!\n")
    
    def extract_embedding(self, face_img):
        """
        Extract 512-dimensional embedding from face image.
        
        Args:
            face_img: Face image in BGR format (OpenCV default)
        
        Returns:
            embedding: [512] normalized array, or None if no face detected
        """
        
        # TODO 2: Convert BGR to RGB
        # ==========================
        # OpenCV loads images in BGR format, but InsightFace expects RGB.
        # Use cv2.cvtColor() to convert the color space.
        # Store result as face_rgb.
        #
        # Hint: Look up cv2.COLOR_BGR2RGB conversion code
        
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # TODO 3: Detect face and extract embedding
        # ==========================================
        # Call self.app.get() with the RGB image to detect faces and extract embeddings.
        # The method returns a list of detected faces.
        # If no faces found, return None.
        # Otherwise, get the first face's embedding (faces[0].embedding).
        # Return the embedding.
        #
        # Hints:
        # - Check if the returned list is empty
        # - Each face object has an .embedding attribute
        # - InsightFace already L2-normalizes embeddings
        
        faces = self.app.get(face_rgb)
        
        if len(faces) > 0:
            embedding = faces[0].embedding
            # Explicitly normalize to unit length for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        
        return None
    
    def extract_embedding_batch(self, face_imgs):
        """Extract embeddings from multiple images (helper method)."""
        embeddings = []
        for img in face_imgs:
            emb = self.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        return np.array(embeddings) if embeddings else None


# Test code
if __name__ == '__main__':
    print("="*70)
    print("Testing Face Embedding Model")
    print("="*70)
    
    try:
        model = FaceEmbeddingModel(device='cpu')
        print("✅ Model loaded successfully\n")
        
        # Test embedding extraction
        test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        embedding = model.extract_embedding(test_img)
        
        if embedding is None:
            print("⚠️  No face in random image (expected)")
            print("✅ Function works - returns None correctly\n")
        else:
            print(f"✅ Embedding extracted: shape {embedding.shape}")
            print(f"   Norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)\n")
        
        print("="*70)
        print("✅ Phase 1 Complete!")
        print("="*70)
        print("\nNext: data/face_capture.py (Phase 2A)")
        
    except NotImplementedError as e:
        print(f"❌ {e}")
        print("\n💡 Implement the TODOs above")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure insightface is installed: pip install insightface")
