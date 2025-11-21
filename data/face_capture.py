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
        
        # Step 1: Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Create FaceDetector
        self.detector = FaceDetector(conf_threshold=0.5)
        
        # Step 3: Open webcam
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Step 4: Check if camera opened
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Step 5: Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
        
        # Step 1: Create person directory
        clean_name = person_name.lower().replace(' ', '_')
        person_dir = self.output_dir / clean_name
        person_dir.mkdir(exist_ok=True)
        
        # Step 2: Count existing photos to continue numbering
        existing_photos = list(person_dir.glob('*.png'))
        photo_count = len(existing_photos)
        
        # Step 3: Create OpenCV window
        window_name = f"Capture: {person_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Step 4: Main capture loop
        while photo_count < num_photos:
            # Read and flip frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detector.detect(frame)
            num_faces = len(faces)
            
            # Determine box color based on face count
            if num_faces == 1:
                box_color = (0, 255, 0)  # Green - good for capture
                status_msg = "✓ Ready to capture"
            elif num_faces > 1:
                box_color = (0, 165, 255)  # Orange - multiple faces
                status_msg = "⚠ Multiple faces detected"
            else:
                box_color = (0, 0, 255)  # Red - no faces
                status_msg = "⚠ No face detected"
            
            # Draw bounding boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Display photo count
            cv2.putText(frame, f'Photos: {photo_count}/{num_photos}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display status message
            cv2.putText(frame, status_msg, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("   Capture stopped by user")
                break
            elif key == 32:  # SPACE
                if num_faces == 1:
                    # Save photo
                    photo_path = person_dir / f'{clean_name}_{photo_count}.png'
                    cv2.imwrite(str(photo_path), frame)
                    print(f"   📷 Captured: {photo_path.name}")
                    photo_count += 1
                else:
                    print(f"   ⚠️  Cannot capture: {num_faces} faces detected (need exactly 1)")
        
        # Step 5: Close window
        cv2.destroyWindow(window_name)
        
        # Step 6: Print summary
        print(f"\n✅ Captured {photo_count} photos for {person_name}")
        print(f"   Saved to: {person_dir}\n")
    
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
        # camera_id=1 is typically built-in webcam on macOS (camera_id=0 may be iPhone with Continuity Camera)
        capturer = FaceCapture(camera_id=1)
        capturer.capture_multiple_people()
    except NotImplementedError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
