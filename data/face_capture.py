"""
Face Photo Capture - Phase 2A

Captures multiple photos of each person using webcam.
Photos saved to data/raw/Dataset/person_name/ folder.

Time: 15-20 minutes + photo capture time
TODOs: 2
"""

import cv2
import sys
from pathlib import Path

sys.path.append('.')
from utils.face_detector import FaceDetector


class FaceCapture:
    """Webcam-based photo capture with real-time face detection feedback."""
    
    def __init__(self, output_dir='data/raw/Dataset', camera_id=0):
        """
        Initialize capture tool.
        
        Args:
            output_dir: Base directory for photos (data/raw/Dataset/person_name/)
            camera_id: Camera device ID (0 = default webcam)
        """
        
        self.output_dir = Path(output_dir)
        self.camera_id = camera_id
        
        # TODO 6: Initialize webcam and detector
        # =======================================
        # Steps:
        # 1. Create output directory: use self.output_dir.mkdir() with parents=True, exist_ok=True
        # 2. Create FaceDetector with conf_threshold=0.5, store as self.detector
        # 3. Open webcam: create cv2.VideoCapture with self.camera_id, store as self.cap
        # 4. Check if camera opened: if not self.cap.isOpened(), raise RuntimeError
        # 5. Set camera resolution: use self.cap.set() for CAP_PROP_FRAME_WIDTH (640) and HEIGHT (480)
        #
        # Hints:
        # - Use Path.mkdir() for creating directories
        # - cv2.CAP_PROP_FRAME_WIDTH and cv2.CAP_PROP_FRAME_HEIGHT are OpenCV constants
        
        raise NotImplementedError("TODO 6: Initialize webcam and detector")
        
        print(f"✅ Capture tool ready\n")
    
    def capture_person(self, person_name, num_photos=20):
        """
        Capture photos of one person.
        
        Controls: SPACE=capture, ESC=stop
        
        Args:
            person_name: Person's name (folder name)
            num_photos: Number of photos to capture
        """
        
        print(f"\n📸 Capturing {num_photos} photos for: {person_name}")
        print("   SPACE=capture, ESC=finish\n")
        
        # TODO 7: Implement capture loop
        # ===============================
        # Steps:
        # 1. Create person directory:
        #    - Clean name: person_name.lower().replace(' ', '_')
        #    - Create folder: self.output_dir / clean_name
        #    - Use mkdir(exist_ok=True)
        # 
        # 2. Count existing photos to continue numbering
        # 
        # 3. Create OpenCV window for display
        # 
        # 4. Main loop (while photo_count < num_photos):
        #    - Read frame from self.cap
        #    - Flip frame horizontally: cv2.flip(frame, 1)
        #    - Detect faces using self.detector.detect()
        #    - Draw bounding boxes on faces (green if 1 face, orange if multiple)
        #    - Display photo count: "Photos: X/Y"
        #    - Show warning if not exactly 1 face detected
        #    - Show frame with cv2.imshow()
        #    - Check for key press:
        #      * ESC (27): break
        #      * SPACE (32): save photo if exactly 1 face
        #        - Save to: person_dir / f'{person_name}_{photo_count}.png'
        #        - Increment photo_count
        # 
        # 5. Close window
        # 6. Print summary
        #
        # Hints:
        # - cv2.waitKey(1) & 0xFF gets key press
        # - Use cv2.rectangle() to draw boxes
        # - Use cv2.imwrite() to save images
        # - Color format is BGR: green=(0,255,0), orange=(0,165,255), red=(0,0,255)
        
        raise NotImplementedError("TODO 7: Implement capture loop")
    
    def capture_multiple_people(self):
        """Interactive mode: capture photos for multiple people."""
        print("="*70)
        print("Face Capture - Multiple People")
        print("="*70)
        print("Enter each person's name, then capture their photos.")
        print("Press ENTER without name to finish.\n")
        
        while True:
            name = input("Person name (or ENTER to finish): ").strip()
            if not name:
                break
            
            try:
                self.capture_person(name, num_photos=20)
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted")
                break
        
        print("\n✅ Capture complete!")
        print("Next: python core/generate_embeddings.py\n")
    
    def __del__(self):
        """Cleanup camera"""
        if hasattr(self, 'cap'):
            self.cap.release()
            cv2.destroyAllWindows()


# Run capture tool
if __name__ == '__main__':
    try:
        capturer = FaceCapture()
        capturer.capture_multiple_people()
    except NotImplementedError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
