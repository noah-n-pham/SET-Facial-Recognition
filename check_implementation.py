"""
Implementation Progress Checker

Run this script to see which TODOs you've completed and what's remaining.
"""
import os
import re
from pathlib import Path

# Files that need implementation (with TODOs)
TODO_FILES = {
    "Phase 1: Environment Setup": [
        "test_gpu.py",
        "verify_dataset.py",
    ],
    "Phase 2: Dataset": [
        "src/data/dataset.py",
        "visualize_augmentations.py",
    ],
    "Phase 3: Model & Training": [
        "src/models/resnet_arcface.py",
        "src/models/losses.py",
        "src/training/train.py",
        "quick_overfit_test.py",
    ],
    "Phase 4: Inference": [
        "src/utils/generate_embeddings.py",
        "src/inference/webcam_recognition.py",
    ],
    "Phase 5: Jetson Deployment": [
        "src/export/export_onnx.py",
        "src/inference/jetson_inference.py",
    ],
    "Phase 6: Arduino Integration": [
        "arduino/face_recognition_controller/face_recognition_controller.ino",
        "tools/find_arduino.py",
        "test_full_pipeline.sh",
    ],
}

def count_todos_in_file(filepath):
    """Count remaining TODO comments in a file"""
    if not os.path.exists(filepath):
        return None, "File not found"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Count TODO comments
            todos = len(re.findall(r'#\s*TODO:|//\s*TODO:', content, re.IGNORECASE))
            return todos, None
    except Exception as e:
        return None, str(e)

def check_implementation():
    """Check implementation progress across all phases"""
    print("="*70)
    print("🔍 FACIAL RECOGNITION IMPLEMENTATION PROGRESS CHECKER")
    print("="*70)
    
    total_todos = 0
    completed_files = 0
    total_files = 0
    
    for phase, files in TODO_FILES.items():
        print(f"\n📌 {phase}")
        print("-" * 70)
        
        for file in files:
            total_files += 1
            todo_count, error = count_todos_in_file(file)
            
            if error:
                print(f"  ❌ {file:45s} - {error}")
            elif todo_count == 0:
                print(f"  ✅ {file:45s} - COMPLETE!")
                completed_files += 1
            else:
                print(f"  🔧 {file:45s} - {todo_count} TODOs remaining")
                total_todos += todo_count
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Files completed: {completed_files}/{total_files}")
    print(f"Remaining TODOs: {total_todos}")
    print(f"Progress: {completed_files/total_files*100:.1f}%")
    
    # Check for additional items
    print("\n" + "="*70)
    print("📁 ADDITIONAL CHECKS")
    print("="*70)
    
    # Check if output directories exist
    dirs_to_check = ["models/checkpoints", "models/exported", "logs"]
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path:30s} - exists")
        else:
            print(f"  ❌ {dir_path:30s} - needs to be created")
    
    # Check if model checkpoint exists
    checkpoint_path = "models/checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"  ✅ {checkpoint_path:30s} - exists ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  {checkpoint_path:30s} - not trained yet")
    
    # Check if reference embeddings exist
    embeddings_path = "models/reference_embeddings.npy"
    if os.path.exists(embeddings_path):
        print(f"  ✅ {embeddings_path:30s} - exists")
    else:
        print(f"  ⚠️  {embeddings_path:30s} - not generated yet")
    
    # Check if ONNX model exists
    onnx_path = "models/exported/face_recognition.onnx"
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✅ {onnx_path:30s} - exists ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  {onnx_path:30s} - not exported yet")
    
    print("\n" + "="*70)
    
    if completed_files == total_files:
        print("🎉 CONGRATULATIONS! All implementations complete!")
        print("   Next steps:")
        print("   1. Train your model: python src/training/train.py")
        print("   2. Generate embeddings: python src/utils/generate_embeddings.py")
        print("   3. Test inference: python src/inference/webcam_recognition.py")
        print("   4. Export to ONNX: python src/export/export_onnx.py")
        print("   5. Deploy to Jetson and test with Arduino")
    else:
        print("📝 Keep going! Check QUICK_START.md for next steps.")
        print(f"   {total_files - completed_files} files remaining")
    
    print("="*70)

if __name__ == "__main__":
    check_implementation()

