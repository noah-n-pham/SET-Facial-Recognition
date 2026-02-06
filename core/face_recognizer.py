"""
Real-Time Face Recognition - Phase 3
+ Emotion Recognition Integration - Semester 2, Phase 3

Complete recognition system that:
1. Detects faces in webcam frames
2. Extracts embeddings (Identity - Semester 1)
3. Compares with reference database (Identity - Semester 1)
4. Classifies emotion (Emotion - Semester 2)
5. Displays results with bounding boxes

Semester 1 TODOs: 11-13
Semester 2 TODOs: 21-23
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from pathlib import Path
import sys
import time

sys.path.append('.')
from models.face_model import FaceEmbeddingModel
from utils.face_detector import FaceDetector

# Semester 2 imports - Emotion Recognition
from models.emotion_model import EmotionModel, EMOTIONS
from utils.emotion_smoother import EmotionSmoother


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

        self.framesListForEmotions = []
        self.counter = 0

        self.emotion_model = EmotionModel()
        print("✅ Emotion Model loaded successfully\n")
        
        # Test with random image (won't give meaningful results, just tests the pipeline)
        self.test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.emotion, self.confidence = self.emotion_model.predict(self.test_img)
        
        self.similarity_threshold = similarity_threshold
        
        # TODO 11: Load models and reference database
        # ============================================
        # Steps:
        # 1. Create FaceDetector with conf_threshold=0.6, store as self.detector
        self.detector = FaceDetector(conf_threshold=similarity_threshold)
        # 2. Create FaceEmbeddingModel with device='cpu', store as self.model
        self.model = FaceEmbeddingModel(device = 'cpu')
        # 3. Load reference embeddings:
        #    - Use np.load() to load from reference_path
        #    - Store as self.reference_embeddings
        self.reference_embeddings = np.load(reference_path)
        # 4. Load label names:
        #    - Open labels_path and read all lines
        self.label_names = []
        with open(labels_path, "r") as file:
            for i in range(len(self.reference_embeddings)):
                self.label_names.append(file.readline().strip())

        for each in self.label_names:
            print(each)
        #    - Strip whitespace from each line
        #    - Store as list in self.label_names
        # 5. Verify consistency:
        #    - Check len(self.reference_embeddings) == len(self.label_names)
        #    - Raise ValueError if mismatch
        if not (len(self.reference_embeddings) == len(self.label_names)):
            print(len(self.reference_embeddings))
            print(len(self.label_names))
            raise ValueError("Mismatch between embeddings and labels")
                # 6. Print summary with known people and threshold
        #
        # Hints:
        # - Use open() with 'r' mode to read file
        # - Use line.strip() to remove whitespace
        # - Arrays loaded with np.load() are already numpy arrays
        
        #raise NotImplementedError("TODO 11: Load models and database")
        
        # ====================================================================
        # SEMESTER 2: Emotion Recognition Integration
        # ====================================================================
        
        # TODO 21: Initialize emotion model and smoother
        # ===============================================
        # Steps:
        # 1. Create EmotionModel instance:
        #    - model_path='assets/enet_b0_8.onnx'
        #    - Store as self.emotion_model
        #
        # 2. Create EmotionSmoother instance:
        #    - window_size=5 (average over 5 frames)
        #    - num_classes=8 (8 emotions)
        #    - Store as self.emotion_smoother
        #
        # 3. Print confirmation message
        #
        # Why smoothing?
        # - Emotion predictions are noisy frame-to-frame
        # - Averaging over 5 frames provides stable output
        # - Similar to how video stabilization works
        #
        # Hints:
        # - EmotionModel is imported from models.emotion_model
        # - EmotionSmoother is imported from utils.emotion_smoother
        # - Use try/except to handle missing model file gracefully
        
        #raise NotImplementedError("TODO 21: Initialize emotion model and smoother")
    
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
        # Steps
        # 2. If embedding is None, return ("Unknown", 0.0)
        embedding = self.model.extract_embedding(face_img=face_img)
        if embedding is None:
            return ("Unknown", 0.0)
        # 3. Compute similarities with all references:
        #    - Use matrix multiplication: embedding @ self.reference_embeddings.T
        #    - This gives similarity score for each person
        similarities = []

        for each in self.reference_embeddings:
            similarities.append(embedding @ each)
        # 4. Find best match:
        #    - Use np.argmax() to get index of highest similarity
        #    - Get similarity value at that index
        #    - Get name from self.label_names at that index
        max_index = np.argmax(similarities)
        similarity = similarities[max_index]
        # 5. Check threshold:
        #    - If similarity >= self.similarity_threshold: return (name, similarity)
        #    - Otherwise: return ("Unknown", similarity)
        if similarity >= self.similarity_threshold:
            return (self.label_names[max_index], similarity)
        else:
            return ("Unknown", similarity)
        # 6. Convert similarity to float before returning
        #
        # Hints:
        # - @ operator does matrix multiplication
        # - .T transposes the array
        # - Embeddings are normalized, so dot product = cosine similarity
    
    # ========================================================================
    # SEMESTER 2: Emotion Recognition Method
    # ========================================================================
    
    def recognize_emotion(self, face_img):
        """
        Recognize emotion from face image.
        
        This runs IN PARALLEL with recognize_face() - same input, different output.
        
        Args:
            face_img: Cropped face image (BGR) - same as recognize_face input
        
        Returns:
            (emotion, confidence): Tuple of emotion label and probability
        """
        
        # TODO 22: Implement emotion recognition with smoothing
        # ======================================================
        # Steps:
        # 1. Get raw prediction from emotion model:
        #    - Call self.emotion_model.predict(face_img)
        #    - This returns (emotion_label, confidence)
        #    - But we need the full probability vector for smoothing!
        #
        # 2. For proper smoothing, we need probabilities:
        #    - Preprocess: input_tensor = self.emotion_model.preprocess(face_img)
        #    - Run inference: logits = self.emotion_model.session.run(...)
        #    - Apply softmax: probs = self.emotion_model.softmax(logits)
        #
        # 3. Update smoother with probabilities:
        #    - Call self.emotion_smoother.update(probs)
        #
        # 4. Get smoothed prediction:
        #    - Call self.emotion_smoother.get_emotion(EMOTIONS)
        #    - This returns (emotion_label, smoothed_confidence)
        #
        # 5. Return the smoothed result
        #
        # Error handling:
        # - Wrap in try/except
        # - On error, return ("Unknown", 0.0)
        #
        # Why smooth here instead of in EmotionModel?
        # - Smoothing is a TEMPORAL operation (across frames)
        # - The model only sees one frame at a time
        # - The recognizer sees the stream of frames
        #
        # Alternative simpler approach (if above is too complex):
        # - Just call self.emotion_model.predict(face_img)
        # - Skip smoothing for now, add it later
        # - This still works but may flicker more
        
        raise NotImplementedError("TODO 22: Implement emotion recognition")
    
    def run_webcam0(self, camera_id=0):
        #Run real-time recognition on webcam. Press ESC to exit.
        
        print("="*70)
        print("Starting Real-Time Recognition")
        print("="*70)
        print("Press ESC to exit\n")
        
        # TODO 13: Implement webcam capture and recognition loop
        # =======================================================
        # Steps:
        
        self.cap = cv2.VideoCapture(camera_id)
        if self.cap.isOpened():
            self.window = cv2.namedWindow("Face Recognizer Window")
            fps = 0
            frame_count = 0
            start_time = time.time()
            while(True):
                ret,frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame,1)
                faces = self.detector.detect(frame)
                for (x,y,w,h) in faces:
                    padding = int(0.1 * max(w, h))
                    x1 = max(0,x-padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y1 = max(0,y-padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    cropped_face = frame[y1:y2,x1:x2]
                    
                    # Path A: Identity Recognition (Semester 1)
                    name, similarity = self.recognize_face(cropped_face)
                    
                    # TODO 23: Add Path B (Emotion) and update display
                    # =================================================
                    # Steps:
                    # 1. Call emotion recognition:
                    #    - emotion, emotion_conf = self.recognize_emotion(cropped_face)
                    #
                    # 2. Update the label to show both identity and emotion:
                    #    - Old: label = f'{name} ({similarity:.2f})'
                    #    - New: label = f'{name} | {emotion}'
                    #    - Or with confidence: f'{name} ({similarity:.2f}) | {emotion} ({emotion_conf:.0%})'
                    #
                    # 3. Optional: Draw emotion on second line
                    #    - cv2.putText for name on line 1 (y1-10)
                    #    - cv2.putText for emotion on line 2 (y1-30)
                    #
                    # 4. Optional: Color-code by emotion
                    #    - Happy: green
                    #    - Angry/Fear: red
                    #    - Neutral: gray
                    #    - etc.
                    #
                    # For now, keep it simple - just add emotion to the label!
                    #
                    # Note: The cropped_face goes to BOTH pipelines:
                    #   - recognize_face() -> Identity (who is this?)
                    #   - recognize_emotion() -> Emotion (how do they feel?)
                    # This is the "parallel processing" architecture from the plan.
                    
                    color = (0,0,0)
                    if name != "Unknown":
                        color = (0,255,0)
                    else:
                        color = (255,0,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    label = f'{name} ({similarity:.2f})'
                    # draw rectange if needed cv2.rectangle
                    cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                frame_count+=1
                if(frame_count % 10 == 0):
                    elapsed = time.time() - start_time
                    fps = 10 / elapsed
                    start_time = time.time()
                cv2.putText(frame,f'FPS: {fps:.1f}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                cv2.imshow("Face Recognizer Window", frame)
                if(cv2.waitKey(1) & 0xFF == 27):
                    break
            self.cap.release()
            cv2.destroyAllWindows()

    def run_webcam1(self, camera_id=0):

        print("="*70)
        print("Starting Real-Time Recognition")
        print("="*70)
        print("Press ESC to exit\n")

        people_dict = []

        def withinThreshold(originalValue, newValue, threshold):
            return abs(originalValue - newValue) <= threshold

        self.cap = cv2.VideoCapture(camera_id)
        if self.cap.isOpened():
            self.window = cv2.namedWindow("Face Recognizer Window")
            fps = 0
            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                faces = self.detector.detect(frame)

                for (x, y, w, h) in faces:

                    threshold = 15
                    matched_index = None

                    # Try to match with existing tracked faces
                    for i, each in enumerate(people_dict):
                        if withinThreshold(each["x"], x, threshold) and withinThreshold(each["y"], y, threshold):
                            matched_index = i
                            break

                    # Crop face with padding
                    padding = int(0.1 * max(w, h))
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    cropped_face = frame[y1:y2, x1:x2]

                    # Recognize or reuse previous identity
                    if matched_index is None:
                        # New face → recognize
                        name, similarity = self.recognize_face(cropped_face)
                        people_dict.append({
                            "x": x,
                            "y": y,
                            "name": name,
                            "similarity": similarity,
                            "keep": True
                        })

                    else:
                        person = people_dict[matched_index]
                        person["x"] = x
                        person["y"] = y
                        person["keep"] = True

                        if person["name"] == "Unknown":
                            # Unknown face → keep trying to recognize every frame
                            name, similarity = self.recognize_face(cropped_face)
                            person["name"] = name
                            person["similarity"] = similarity
                        else:
                            # Known face → reuse stored identity
                            name = person["name"]
                            similarity = person["similarity"]

                    # Draw bounding box + label
                    color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} ({similarity:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Cleanup: remove faces not seen this frame
                for i in range(len(people_dict) - 1, -1, -1):
                    if not people_dict[i]["keep"]:
                        del people_dict[i]
                    else:
                        people_dict[i]["keep"] = False

                # FPS calculation
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = 10 / elapsed
                    start_time = time.time()

                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                cv2.imshow("Face Recognizer Window", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            self.cap.release()
            cv2.destroyAllWindows()


    def run_webcam2(self, camera_id=0):
    
        import numpy as np
        from filterpy.kalman import KalmanFilter

        print("="*70)
        print("Starting Real-Time Recognition (Kalman Tracking)")
        print("="*70)
        print("Press ESC to exit\n")

    # ---------------------------------------------------------
    # Create a Kalman filter for a new face track
    # ---------------------------------------------------------
        def create_kalman(x, y, w, h):
            kf = KalmanFilter(dim_x=7, dim_z=4)

            # State transition matrix
            kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
            ])

        # Measurement matrix
            kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
            ])

        # Convert bounding box to Kalman state
            cx = x + w/2
            cy = y + h/2
            s = w * h
            r = w / float(h)

            kf.x[:4] = np.array([[cx], [cy], [s], [r]])
            kf.P *= 10
            return kf

    # ---------------------------------------------------------
    # IoU for matching detections to predicted tracks
    # ---------------------------------------------------------
        def iou(bb1, bb2):
            x1, y1, w1, h1 = bb1
            x2, y2, w2, h2 = bb2

            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)

            inter = max(0, xb - xa) * max(0, yb - ya)
            union = w1*h1 + w2*h2 - inter
            return inter / union if union > 0 else 0

    # ---------------------------------------------------------
    # Track structure
    # ---------------------------------------------------------
        tracks = []
        next_id = 0

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        cv2.namedWindow("Face Recognizer Window")

        fps = 0
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            detections = self.detector.detect(frame)

        # ---------------------------------------------------------
        # 1. PREDICT all tracks
        # ---------------------------------------------------------
            for tr in tracks:
                tr["kf"].predict()
                tr["ttl"] -= 1

        # ---------------------------------------------------------
        # 2. MATCH detections to tracks using IoU
        # ---------------------------------------------------------
            unmatched_dets = []
            unmatched_tracks = list(range(len(tracks)))

            for det in detections:
                x, y, w, h = det
                best_iou = 0
                best_track = None

                for ti in unmatched_tracks:
                    kf = tracks[ti]["kf"]

                # Convert Kalman state to bounding box
                    cx, cy, s, r = kf.x[:4].reshape(-1)
                    pw = np.sqrt(s * r)
                    ph = s / pw
                    px = cx - pw/2
                    py = cy - ph/2

                    track_box = (px, py, pw, ph)
                    det_box = (x, y, w, h)

                    score = iou(det_box, track_box)
                    if score > best_iou:
                        best_iou = score
                        best_track = ti

                if best_iou > 0.3:
                # Match found
                    unmatched_tracks.remove(best_track)

                # Update Kalman filter
                    cx = x + w/2
                    cy = y + h/2
                    s = w*h
                    r = w/float(h)
                    tracks[best_track]["kf"].update([cx, cy, s, r])

                # Re-recognize if unknown
                    if tracks[best_track]["name"] == "Unknown":
                        name, sim = self.recognize_face(frame[y:y+h, x:x+w])
                        tracks[best_track]["name"] = name
                        tracks[best_track]["sim"] = sim

                    tracks[best_track]["ttl"] = 10
                else:
                    unmatched_dets.append(det)

        # ---------------------------------------------------------
        # 3. CREATE new tracks for unmatched detections
        # ---------------------------------------------------------
            for (x, y, w, h) in unmatched_dets:
                name, sim = self.recognize_face(frame[y:y+h, x:x+w])
                kf = create_kalman(x, y, w, h)

                tracks.append({
                "id": next_id,
                "kf": kf,
                "name": name,
                "sim": sim,
                "ttl": 10
                })
                next_id += 1

        # ---------------------------------------------------------
        # 4. REMOVE dead tracks
        # ---------------------------------------------------------
            tracks = [t for t in tracks if t["ttl"] > 0]

        # ---------------------------------------------------------
        # 5. DRAW results
        # ---------------------------------------------------------
            for tr in tracks:
                cx, cy, s, r = tr["kf"].x[:4].reshape(-1)
                w = np.sqrt(s * r)
                h = s / w
                x = int(cx - w/2)
                y = int(cy - h/2)

                color = (0,255,0) if tr["name"] != "Unknown" else (0,0,255)
                cv2.rectangle(frame, (x,y), (x+int(w), y+int(h)), color, 2)
                label = f'ID {tr["id"]}: {tr["name"]} ({tr["sim"]:.2f})'
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = 10 / elapsed
                start_time = time.time()

            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            cv2.imshow("Face Recognizer Window", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_webcam3(self, camera_id=0):
        import numpy as np
        from filterpy.kalman import KalmanFilter
        from concurrent.futures import ThreadPoolExecutor

        print("="*70)
        print("Starting Real-Time Recognition (Kalman + MT + Smoothing)")
        print("="*70)
        print("Press ESC to exit\n")

        def create_kalman(x, y, w, h):
                kf = KalmanFilter(dim_x=7, dim_z=4)

                kf.F = np.array([
                    [1,0,0,0,1,0,0],
                    [0,1,0,0,0,1,0],
                    [0,0,1,0,0,0,1],
                    [0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,1]
                ])

                kf.H = np.array([
                    [1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0]
                ])

                cx = x + w/2
                cy = y + h/2
                s = max(1, w * h)
                r = max(0.1, w / float(h))

                kf.x[:4] = np.array([[cx], [cy], [s], [r]])
                kf.P *= 10
                return kf

        def iou(bb1, bb2):
                x1, y1, w1, h1 = bb1
                x2, y2, w2, h2 = bb2

                xa = max(x1, x2)
                ya = max(y1, y2)
                xb = min(x1 + w1, x2 + w2)
                yb = min(y1 + h1, y2 + h2)

                inter = max(0, xb - xa) * max(0, yb - ya)
                union = w1*h1 + w2*h2 - inter
                return inter / union if union > 0 else 0

        def clamp_box(x, y, w, h, frame):
                H, W = frame.shape[:2]

                w = max(1, w)
                h = max(1, h)

                x = max(0, min(x, W - 1))
                y = max(0, min(y, H - 1))

                w = min(w, W - x)
                h = min(h, H - y)

                return int(x), int(y), int(w), int(h)

        def color_for_id(track_id):
                rng = np.random.RandomState(track_id * 9973 + 12345)
                return int(rng.randint(50, 255)), int(rng.randint(50, 255)), int(rng.randint(50, 255))

        tracks = []
        next_id = 0

        cap = cv2.VideoCapture(camera_id)
        cv2.namedWindow("Face Recognizer Window")

        fps = 0
        frame_count = 0
        start_time = time.time()

        alpha = 0.6

        with ThreadPoolExecutor(max_workers=4) as executor:
                while True:
                        ret, frame = cap.read()
                        if not ret:
                                break

                        frame = cv2.flip(frame, 1)
                        detections = self.detector.detect(frame)

                        for tr in tracks:
                                tr["kf"].predict()
                                tr["ttl"] -= 1

                        unmatched_dets = []
                        unmatched_tracks = list(range(len(tracks)))

                        unknown_jobs = []
                        new_jobs = []

                        for det in detections:
                                x, y, w, h = det
                                x, y, w, h = clamp_box(x, y, w, h, frame)

                                best_iou = 0
                                best_track = None

                                for ti in unmatched_tracks:
                                        kf = tracks[ti]["kf"]
                                        cx, cy, s, r = kf.x[:4].reshape(-1)

                                        pw = max(1, np.sqrt(abs(s * r)))
                                        ph = max(1, abs(s) / pw)

                                        px = cx - pw/2
                                        py = cy - ph/2

                                        px, py, pw, ph = clamp_box(int(px), int(py), int(pw), int(ph), frame)

                                        track_box = (px, py, pw, ph)
                                        det_box = (x, y, w, h)

                                        score = iou(det_box, track_box)
                                        if score > best_iou:
                                                best_iou = score
                                                best_track = ti

                                if best_iou > 0.3 and best_track is not None:
                                        unmatched_tracks.remove(best_track)

                                        cx = x + w/2
                                        cy = y + h/2
                                        s = max(1, w*h)
                                        r = max(0.1, w/float(h))

                                        tracks[best_track]["kf"].update([cx, cy, s, r])
                                        tracks[best_track]["ttl"] = 10

                                        crop = frame[y:y+h, x:x+w]
                                        if crop.size == 0:
                                                continue

                                        if tracks[best_track]["name"] == "Unknown":
                                                unknown_jobs.append((best_track, crop))
                                else:
                                        unmatched_dets.append((x, y, w, h))

                        if unknown_jobs:
                                crops_unknown = [c for (_, c) in unknown_jobs]
                                results_unknown = list(executor.map(self.recognize_face, crops_unknown))
                                for (track_idx, _), (name, sim) in zip(unknown_jobs, results_unknown):
                                        tracks[track_idx]["name"] = name
                                        tracks[track_idx]["sim"] = sim

                        if unmatched_dets:
                                crops_new = [frame[y:y+h, x:x+w] for (x, y, w, h) in unmatched_dets]
                                crops_new = [c for c in crops_new if c.size > 0]
                                results_new = list(executor.map(self.recognize_face, crops_new))

                                idx = 0
                                for (x, y, w, h) in unmatched_dets:
                                        crop = frame[y:y+h, x:x+w]
                                        if crop.size == 0:
                                                continue

                                        name, sim = results_new[idx]
                                        idx += 1

                                        kf = create_kalman(x, y, w, h)
                                        tracks.append({
                                            "id": next_id,
                                            "kf": kf,
                                            "name": name,
                                            "sim": sim,
                                            "ttl": 10,
                                            "draw_box": None,
                                            "recent_frames": [crop.copy()],
                                            "frames_position": 1
                                        })
                                        next_id += 1

                        tracks = [t for t in tracks if t["ttl"] > 0]

                        for tr in tracks:
                                cx, cy, s, r = tr["kf"].x[:4].reshape(-1)

                                w = max(1, np.sqrt(abs(s * r)))
                                h = max(1, abs(s) / w)

                                x = int(cx - w/2)
                                y = int(cy - h/2)

                                x, y, w, h = clamp_box(x, y, int(w), int(h), frame)

                                if tr["draw_box"] is None:
                                        tr["draw_box"] = np.array([x, y, w, h], dtype=np.float32)
                                else:
                                        current = np.array([x, y, w, h], dtype=np.float32)
                                        tr["draw_box"] = alpha * tr["draw_box"] + (1 - alpha) * current

                                dx, dy, dw, dh = tr["draw_box"].astype(int)

                                color = color_for_id(tr["id"])
                                cv2.rectangle(frame, (dx, dy), (dx+dw, dy+dh), color, 2)
                                
                                # Predict emotion using current frame's crop (every frame)
                                emotion_label = ""
                                current_crop = frame[dy:dy+dh, dx:dx+dw]
                                if current_crop.size > 0:
                                    emotion_label, emotion_conf = self.emotion_model.predict(current_crop)
                                
                                label = f'ID {tr["id"]}: {tr["name"]} ({tr.get("sim", 0):.2f}) | {emotion_label}'
                                cv2.putText(frame, label, (dx, dy-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                


                        frame_count += 1
                        if frame_count % 10 == 0:
                                elapsed = time.time() - start_time
                                fps = 10 / elapsed
                                start_time = time.time()

                        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

                        cv2.imshow("Face Recognizer Window", frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                                break

        cap.release()
        cv2.destroyAllWindows()

    def run_webcam(self, camera_id = 0, method_num = 1):
        if method_num == 1:
            self.run_webcam1(camera_id=camera_id)
        elif method_num == 2:
            self.run_webcam2(camera_id=camera_id)
        elif method_num == 3:
            self.run_webcam3(camera_id=camera_id)
        elif method_num == 0:
            self.run_webcam0(camera_id=camera_id)

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
        # camera_id=0 is default camera (may be iPhone with Continuity Camera)
        # camera_id=1 is typically built-in webcam on macOS
        # Change to camera_id=1 if iPhone is activating instead of computer webcam
        recognizer.run_webcam(method_num=3)
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
