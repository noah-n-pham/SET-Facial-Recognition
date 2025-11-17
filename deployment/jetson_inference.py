"""
Jetson Nano Deployment - Phase 4 (Optional)

This phase is for students who want to deploy their face recognition system
to a Jetson Nano for standalone operation.

Time: 2-3 hours
Prerequisites: Phases 1-3 complete
Hardware: Jetson Nano, compatible webcam
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║                    JETSON NANO DEPLOYMENT GUIDE                     ║
╚════════════════════════════════════════════════════════════════════╝

Phase 4 is OPTIONAL but provides hands-on experience with edge deployment.

════════════════════════════════════════════════════════════════════
📦 DEPLOYMENT APPROACH
════════════════════════════════════════════════════════════════════

GOOD NEWS: You can use your existing code directly on Jetson Nano!
The face_recognizer.py you built in Phase 3 works out-of-the-box.

Why? 
- InsightFace supports ARM architecture
- MobileFaceNet is optimized for edge devices
- No GPU needed (but can use Jetson GPU if available)

════════════════════════════════════════════════════════════════════
🚀 DEPLOYMENT STEPS
════════════════════════════════════════════════════════════════════

Step 1: Transfer Project to Jetson
───────────────────────────────────

On your laptop:
```bash
# Compress the project (exclude venv and data)
tar -czf facial-recognition.tar.gz \\
    --exclude='venv' \\
    --exclude='data/raw' \\
    models/ core/ utils/ data/ configs/ requirements.txt

# Copy to Jetson (replace with your Jetson IP)
scp facial-recognition.tar.gz jetson@192.168.1.10:~/

# Also copy your reference database
scp models/reference_embeddings.npy jetson@192.168.1.10:~/
scp models/label_names.txt jetson@192.168.1.10:~/
```

Step 2: Setup on Jetson
────────────────────────

SSH to Jetson:
```bash
ssh jetson@192.168.1.10

# Extract project
cd ~
tar -xzf facial-recognition.tar.gz

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Move reference files
mv reference_embeddings.npy models/
mv label_names.txt models/
```

Step 3: Test on Jetson
───────────────────────

```bash
# Run the same recognition script!
python3 core/face_recognizer.py
```

That's it! Your code should work immediately.

════════════════════════════════════════════════════════════════════
⚡ OPTIONAL: GPU ACCELERATION
════════════════════════════════════════════════════════════════════

To use Jetson's GPU for faster inference:

1. Modify core/face_recognizer.py:
   Change: FaceEmbeddingModel(device='cpu')
   To: FaceEmbeddingModel(device='cuda')

2. Make sure CUDA is available:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

Expected speedup: 1.5-2x faster inference

════════════════════════════════════════════════════════════════════
🔌 ARDUINO INTEGRATION (Advanced)
════════════════════════════════════════════════════════════════════

Want to control LEDs/servos based on face recognition?

See: arduino/face_recognition_controller/face_recognition_controller.ino

High-level approach:
1. Connect Arduino to Jetson via USB
2. Arduino listens on serial port
3. Modify face_recognizer.py to send messages:
   
   ```python
   import serial
   ser = serial.Serial('/dev/ttyUSB0', 9600)
   
   # After recognition:
   if name != "Unknown":
       ser.write(f"PERSON:{name}\\n".encode())
   else:
       ser.write(b"UNKNOWN\\n")
   ```

4. Arduino receives messages and controls hardware

════════════════════════════════════════════════════════════════════
🐛 TROUBLESHOOTING
════════════════════════════════════════════════════════════════════

Issue: "No module named 'insightface'"
→ Make sure you activated venv: source venv/bin/activate
→ Install: pip install insightface

Issue: "Camera not found"
→ Check camera: ls /dev/video*
→ Try different camera_id in face_recognizer.py

Issue: "Slow inference (< 10 FPS)"
→ Switch to GPU: device='cuda'
→ Reduce camera resolution in configs/config.yaml
→ Consider ONNX optimization (see below)

Issue: "Permission denied for /dev/ttyUSB0"
→ Add user to dialout group: sudo usermod -a -G dialout $USER
→ Logout and login again

════════════════════════════════════════════════════════════════════
🔧 ADVANCED: ONNX OPTIMIZATION
════════════════════════════════════════════════════════════════════

For maximum performance (2-4x speedup), export model to ONNX:

This is ADVANCED and optional. The Python API is already fast!

If interested:
1. Export InsightFace model to ONNX format
2. Use TensorRT for optimization
3. Load ONNX model in inference code

See: InsightFace documentation for ONNX export

════════════════════════════════════════════════════════════════════
✅ SUCCESS CRITERIA
════════════════════════════════════════════════════════════════════

You've successfully deployed when:
- [✓] Jetson recognizes faces in real-time (10+ FPS)
- [✓] Recognition accuracy matches laptop results
- [✓] System runs standalone (no laptop needed)
- [✓] (Optional) Arduino responds to face recognition

════════════════════════════════════════════════════════════════════
🎓 WHAT YOU'VE LEARNED
════════════════════════════════════════════════════════════════════

✅ Edge device deployment
✅ Cross-architecture compatibility (x86 → ARM)
✅ Hardware interfacing (serial communication)
✅ Performance optimization for embedded systems
✅ Real-world production deployment

════════════════════════════════════════════════════════════════════

🎉 Congratulations! You've built AND deployed a face recognition system!

""")

# If running this file directly, show the guide
if __name__ == "__main__":
    print("\n💡 This is a deployment guide, not executable code.")
    print("   Follow the instructions above to deploy to Jetson Nano.")
    print("\n📚 For more details, see LEARNING_GUIDE.md Phase 4\n")
