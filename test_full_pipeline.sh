#!/bin/bash
# Full pipeline test script

echo "========================================"
echo "Full Pipeline Test"
echo "========================================"

# TODO: Check camera
# List /dev/video* devices
# Print ✅ if found, ❌ if not

# TODO: Check Arduino
# List /dev/ttyUSB* and /dev/ttyACM* devices
# Print ✅ if found, ⚠️ if not

# TODO: Check model files
# Check if models/exported/face_recognition.onnx exists
# Check if models/reference_embeddings.npy exists
# Print ✅ for each found file, ❌ if missing

# TODO: Test Python dependencies
# Run: python3 -c "import cv2, numpy, onnxruntime, serial"
# Print ✅ if successful, ❌ if import error

# TODO: Run inference
# Execute: python3 src/inference/jetson_inference.py
# Print instructions: "Press ESC to exit"

echo -e "\n========================================"
echo "Test Complete"
echo "========================================"

