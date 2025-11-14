"""
Real-Time Face Recognition - Phase 3

Complete recognition system that:
1. Detects faces in webcam frames
2. Extracts embeddings
3. Compares with reference database
4. Displays results with bounding boxes

Time: 2-3 hours
TODOs: 4
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import time

sys.path.append('.')
from models.face_model import FaceEmbeddingModel
from utils.face_detector import FaceDetector


class FaceRecognizer:
    """Real-time face recognition system."""
    
    def __init__(self, 
                 reference_path='models/reference_embeddings.npy',
                 labels_path='models/label_names.txt',
                 similarity_threshold=0.6):
        """
        Initialize recognition system.
        
        Args:
            reference_path: Path to reference embeddings
            labels_path: Path to label names
            similarity_threshold: Minimum similarity to recognize [0-1]
        """
        
        print("="*70)
        print("Face Recognition System - Initializing")
        print("="*70)
        print()
        
        self.similarity_threshold = similarity_threshold
        
        # TODO 11: Load models and reference database
        # ============================================
        # Steps:
        # 1. Create FaceDetector with conf_threshold=0.6, store as self.detector
        # 2. Create FaceEmbeddingModel with device='cpu', store as self.model
        # 3. Load reference embeddings:
        #    - Use np.load() to load from reference_path
        #    - Store as self.reference_embeddings
        # 4. Load label names:
        #    - Open labels_path and read all lines
        #    - Strip whitespace from each line
        #    - Store as list in self.label_names
        # 5. Verify consistency:
        #    - Check len(self.reference_embeddings) == len(self.label_names)
        #    - Raise ValueError if mismatch
        # 6. Print summary with known people and threshold
        #
        # Hints:
        # - Use open() with 'r' mode to read file
        # - Use line.strip() to remove whitespace
        # - Arrays loaded with np.load() are already numpy arrays
        
        raise NotImplementedError("TODO 11: Load models and database")
    
    def recognize_face(self, face_img):
        """
        Recognize face by comparing with reference database.
        
        Args:
            face_img: Cropped face image (BGR)
        
        Returns:
            (name, similarity): Tuple of name and confidence score
        """
        
        # TODO 12: Extract embedding and find best match
        # ===============================================
        # Steps:
        # 1. Extract embedding from face_img using self.model.extract_embedding()
        # 2. If embedding is None, return ("Unknown", 0.0)
        # 3. Compute similarities with all references:
        #    - Use matrix multiplication: embedding @ self.reference_embeddings.T
        #    - This gives similarity score for each person
        # 4. Find best match:
        #    - Use np.argmax() to get index of highest similarity
        #    - Get similarity value at that index
        #    - Get name from self.label_names at that index
        # 5. Check threshold:
        #    - If similarity >= self.similarity_threshold: return (name, similarity)
        #    - Otherwise: return ("Unknown", similarity)
        # 6. Convert similarity to float before returning
        #
        # Hints:
        # - @ operator does matrix multiplication
        # - .T transposes the array
        # - Embeddings are normalized, so dot product = cosine similarity
        
        raise NotImplementedError("TODO 12: Implement recognition")
    
    def run_webcam(self, camera_id=0):
        """Run real-time recognition on webcam. Press ESC to exit."""
        
        print("="*70)
        print("Starting Real-Time Recognition")
        print("="*70)
        print("Press ESC to exit\n")
        
        # TODO 13: Implement webcam capture and recognition loop
        # =======================================================
        # Steps:
        # 1. Open webcam with cv2.VideoCapture(camera_id)
        # 2. Check if opened successfully
        # 3. Create window with cv2.namedWindow()
        # 4. Initialize FPS tracking variables (fps=0, frame_count=0, start_time)
        # 
        # 5. Main loop (try-except-finally for cleanup):
        #    While True:
        #    a. Read frame from camera
        #    b. Flip frame horizontally
        #    c. Detect faces using self.detector.detect()
        #    
        #    d. For each detected face (x, y, w, h):
        #       - Add 10% padding around face
        #       - Ensure coordinates are within frame bounds
        #       - Crop face region from frame
        #       - Call self.recognize_face() on cropped face
        #       - Choose color: green if recognized, red if unknown
        #       - Draw rectangle around face
        #       - Create label with name and similarity score
        #       - Draw filled rectangle for label background
        #       - Put text label above face
        #    
        #    e. Calculate FPS (every 10 frames)
        #    f. Display FPS on frame
        #    g. Show frame with cv2.imshow()
        #    h. Check for ESC key (27) to exit
        # 
        # 6. In finally block: release camera and destroy windows
        # 7. Print summary with frame count and average FPS
        #
        # Hints:
        # - Use cv2.flip(frame, 1) for mirror effect
        # - Padding: int(0.1 * max(w, h))
        # - Use max(0, x) and min(x, frame_width) for bounds
        # - Green=(0,255,0), Red=(0,0,255) in BGR
        # - cv2.getTextSize() for label dimensions
        # - cv2.FILLED for filled rectangle
        # - time.time() for FPS calculation
        
        raise NotImplementedError("TODO 13: Implement webcam loop")


# Run recognizer
if __name__ == '__main__':
    reference_path = Path('models/reference_embeddings.npy')
    labels_path = Path('models/label_names.txt')
    
    if not reference_path.exists() or not labels_path.exists():
        print("❌ Reference database not found!")
        print("   Run core/generate_embeddings.py first")
        exit(1)
    
    try:
        recognizer = FaceRecognizer(
            reference_path=str(reference_path),
            labels_path=str(labels_path),
            similarity_threshold=0.6
        )
        recognizer.run_webcam()
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
