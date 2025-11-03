"""
Real-time face recognition using webcam.
Combines YuNet detection with trained ResNet-18 + ArcFace model.
"""
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path

from src.models.resnet_arcface import ResNetArcFace
from src.data.augmentation import get_val_transforms


class FaceRecognizer:
    """Real-time face recognizer"""
    
    def __init__(self, model_path, embeddings_path, labels_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.transform = get_val_transforms()
        
        print("Loading model...")
        
        # TODO: Load trained model
        # 1. Create ResNetArcFace with parameters from config
        # 2. Load checkpoint with torch.load()
        # 3. Load state_dict from checkpoint
        # 4. Move model to device
        # 5. Set model to eval mode
        
        # TODO: Load reference embeddings
        # Use np.load() to load embeddings_path
        # Store as self.reference_embeddings
        
        # TODO: Load label names
        # Open labels_path and read lines
        # Strip whitespace from each line
        # Store as self.label_names list
        
        # TODO: Initialize YuNet face detector
        # Use cv2.FaceDetectorYN.create() with:
        #   - model path: "assets/face_detection_yunet_2023mar.onnx"
        #   - config: ""
        #   - input_size: (320, 320)
        #   - score_threshold: 0.7
        #   - nms_threshold: 0.3
        #   - top_k: 5000
        # Store as self.detector
        
        # TODO: Get similarity threshold from config
        # Store as self.similarity_threshold
        
    def recognize_face(self, face_img):
        """
        Recognize a face by comparing embedding to references.
        
        Args:
            face_img: Cropped face image (BGR format from OpenCV)
        
        Returns:
            name (str): Recognized person name or "Unknown"
            similarity (float): Similarity score [0, 1]
        """
        # TODO: Preprocess face image
        # 1. Convert BGR to RGB with cv2.cvtColor()
        # 2. Apply self.transform (pass as image= parameter)
        # 3. Extract tensor with ['image'] key
        # 4. Add batch dimension with .unsqueeze(0)
        # 5. Move to device
        
        # TODO: Extract embedding
        # Use torch.no_grad() context
        # Call model.extract_embedding()
        # Convert to numpy: .cpu().numpy()[0]
        
        # TODO: Compute cosine similarity with all references
        # Use matrix multiplication: embedding @ reference_embeddings.T
        # This gives similarity score with each reference
        
        # TODO: Find best match
        # Use .argmax() to find index of highest similarity
        # Get the max similarity value
        
        # TODO: Check if similarity exceeds threshold
        # If yes: return self.label_names[max_idx], max_similarity
        # If no: return "Unknown", max_similarity
        pass
    
    def run_webcam(self, camera_id=0):
        """Run real-time recognition on webcam"""
        # TODO: Open webcam with cv2.VideoCapture(camera_id)
        
        window_name = "Face Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nStarting webcam recognition...")
        print("Press ESC to exit\n")
        
        # TODO: Main loop
        # while True:
        #     1. Read frame from camera
        #     2. Flip frame horizontally for mirror effect
        #     3. Get frame dimensions
        #     4. Update detector input size
        #     5. Detect faces
        #     6. For each detected face:
        #        a. Extract bounding box coordinates
        #        b. Add padding (10% of face size)
        #        c. Ensure coordinates within frame bounds
        #        d. Extract face ROI
        #        e. Call recognize_face()
        #        f. Draw bounding box (green if recognized, red if unknown)
        #        g. Draw label with name and similarity score
        #     7. Show frame
        #     8. Check for ESC key (27) to exit
        
        # TODO: Release camera and close windows
        pass


def main():
    # TODO: Load config from configs/config.yaml
    
    # TODO: Create FaceRecognizer with paths to:
    # - model checkpoint
    # - reference embeddings
    # - label names
    # - config
    
    # TODO: Run webcam recognition
    pass


if __name__ == '__main__':
    main()

