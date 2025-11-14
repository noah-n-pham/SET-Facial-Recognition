"""
Jetson Nano Inference (Phase 4 - Optional)

This module adapts the face recognition system for deployment on Jetson Nano.
It uses the same core recognition logic but optimized for edge hardware.

Phase: 4 - Hardware Deployment (Optional)
Time: 2-3 hours
Prerequisites: Phases 1-3 complete

Note: This is the same as desktop recognition but can be optimized further
with ONNX export and TensorRT for faster inference on Jetson.

For now, students can use the core/face_recognizer.py directly on Jetson!
The pretrained InsightFace model works great on ARM architecture.

To deploy:
1. Copy project to Jetson: scp -r Facial-Recognition/ jetson@<IP>:~/
2. SSH to Jetson: ssh jetson@<IP>
3. Install dependencies: cd Facial-Recognition && pip3 install -r requirements.txt
4. Run recognition: python3 core/face_recognizer.py

Arduino integration can be added by importing serial communication
in the face_recognizer.py file.
"""

print("""
Jetson Deployment Instructions:
================================

Option 1: Use core/face_recognizer.py directly (Recommended)
- Works out of the box on Jetson
- Good performance (15-20 FPS)
- No additional setup needed

Option 2: ONNX Optimization (Advanced)
- Export model to ONNX for TensorRT optimization
- Potential 2-4x speedup
- Requires additional setup

For most students, Option 1 is sufficient!
""")

