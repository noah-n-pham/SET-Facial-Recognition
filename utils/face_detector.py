"""
Face Detector - Helper for Phase 2A and 3

Wraps YuNet face detector for finding faces in images.
Returns bounding boxes [x, y, width, height] for each detected face.

Time: 15-20 minutes
TODOs: 2
"""

import cv2


class FaceDetector:
    """Detects faces in images/video and returns bounding boxes."""
    
    def __init__(self, 
                 model_path='assets/face_detection_yunet_2023mar.onnx',
                 conf_threshold=0.7):
        """
        Initialize YuNet face detector.
        
        Args:
            model_path: Path to YuNet ONNX model
            conf_threshold: Minimum confidence [0-1] to keep detection
        """
        
        self.conf_threshold = conf_threshold
        print(f"Initializing face detector (threshold: {conf_threshold})...")
        
        # TODO 4: Create YuNet face detector
        # ===================================
        # Use cv2.FaceDetectorYN.create() to create the detector.
        # Pass these parameters:
        # - model: the model_path
        # - config: empty string ""
        # - input_size: tuple (320, 320)
        # - score_threshold: conf_threshold
        # - nms_threshold: 0.3
        # - top_k: 5000
        # Store as self.detector
        #
        # Hint: Look up cv2.FaceDetectorYN.create() in OpenCV docs
        
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=conf_threshold,
            nms_threshold=0.3,
            top_k=5000
        )
        
        print("✅ Face detector ready\n")
    
    def detect(self, frame):
        """
        Detect faces in frame.
        
        Args:
            frame: Image in BGR format
        
        Returns:
            List of bounding boxes [[x, y, w, h], ...] or [] if no faces
        """
        
        # TODO 5: Detect faces and return bounding boxes
        # ===============================================
        # Steps:
        # 1. Get frame dimensions using frame.shape[:2]
        # 2. Update detector input size: call self.detector.setInputSize() with (width, height)
        # 3. Detect faces: call self.detector.detect(frame) - returns (status, faces)
        # 4. If faces is None, return empty list []
        # 5. Loop through faces array and extract bounding box from each:
        #    - Get x, y, w, h from face[:4] (first 4 elements)
        #    - Convert to integers
        #    - Append [x, y, w, h] to results list
        # 6. Return the results list
        #
        # Hints:
        # - setInputSize() must be called before detect()
        # - Use .astype(int) to convert to integers
        # - face[:4] gets first 4 elements (x, y, width, height)
        
        # Step 1: Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Step 2: Update detector input size
        self.detector.setInputSize((frame_width, frame_height))
        
        # Step 3: Detect faces
        _, faces = self.detector.detect(frame)
        
        # Step 4: If no faces, return empty list
        if faces is None:
            return []
        
        # Step 5: Extract bounding boxes from all detected faces
        results = []
        for face in faces:
            x, y, w, h = map(int, face[:4])
            results.append([x, y, w, h])
        
        # Step 6: Return results
        return results


# Test code
if __name__ == '__main__':
    print("="*70)
    print("Testing Face Detector with Webcam")
    print("="*70)
    print("Press ESC to exit\n")
    
    try:
        detector = FaceDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            exit(1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = detector.detect(frame)
            
            # Draw boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Detection Test', frame)
            
            if cv2.waitKey(1) == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Detection test complete!")
        
    except NotImplementedError as e:
        print(f"❌ {e}")
