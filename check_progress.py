"""
Face Recognition + Emotion Recognition Progress Checker

Run this script to see which TODOs you've completed and what's remaining.
This tracks progress through the TODO-based learning approach.

Semester 1: Face Recognition (Identity)
Semester 2: Emotion Recognition (Classification)
"""
import os
import re
from pathlib import Path

# ============================================================================
# SEMESTER 1: Face Recognition (Identity)
# ============================================================================
TODO_FILES_SEM1 = {
    "Sem 1 - Phase 1: Load Pretrained Model": [
        "models/face_model.py",
    ],
    "Sem 1 - Phase 2A: Capture Face Photos": [
        "utils/face_detector.py",
        "data/face_capture.py",
    ],
    "Sem 1 - Phase 2B: Generate Reference Database": [
        "core/generate_embeddings.py",
    ],
    "Sem 1 - Phase 3: Real-Time Recognition": [
        "core/face_recognizer.py",
    ],
}

# ============================================================================
# SEMESTER 2: Emotion Recognition (Classification)
# ============================================================================
TODO_FILES_SEM2 = {
    "Sem 2 - Phase 1: Emotion Model": [
        "models/emotion_model.py",
    ],
    "Sem 2 - Phase 2: Smoothing Buffer": [
        "utils/emotion_smoother.py",
    ],
    "Sem 2 - Phase 3: Integration": [
        "core/face_recognizer.py",  # Has both Sem 1 and Sem 2 TODOs
    ],
}

# Combined for backward compatibility
TODO_FILES = {**TODO_FILES_SEM1, **TODO_FILES_SEM2}

# Expected outputs at each phase
EXPECTED_OUTPUTS = {
    # Semester 1
    "Reference Database": "models/reference_embeddings.npy",
    "Label Names": "models/label_names.txt",
    "Face Photos": "data/raw/Dataset",
    # Semester 2
    "Emotion Model": "assets/mobilenet_7.onnx",
}


def count_todos_in_file(filepath):
    """Count remaining TODO comments in a file"""
    if not os.path.exists(filepath):
        return None, "File not found"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for NotImplementedError or TODO comments
            not_implemented = len(re.findall(r'raise NotImplementedError', content))
            todo_comments = len(re.findall(r'#\s*TODO\s+\d+:', content, re.IGNORECASE))
            return max(not_implemented, todo_comments), None
    except Exception as e:
        return None, str(e)


def check_implementation():
    """Check implementation progress across all phases"""
    print("="*70)
    print("🔍 FACE + EMOTION RECOGNITION PROGRESS CHECKER")
    print("="*70)
    print("\nThis checker tracks your progress through the TODO-based learning.")
    print("• Semester 1: Face Recognition → LEARNING_GUIDE.md")
    print("• Semester 2: Emotion Recognition → SEMESTER_2_GUIDE.md\n")
    
    total_todos = 0
    completed_files = 0
    total_files = 0
    phase_status = {}
    
    for phase, files in TODO_FILES.items():
        print(f"\n📌 {phase}")
        print("-" * 70)
        
        phase_complete = True
        
        for file in files:
            total_files += 1
            todo_count, error = count_todos_in_file(file)
            
            if error:
                print(f"  ❌ {file:45s} - {error}")
                phase_complete = False
            elif todo_count == 0:
                print(f"  ✅ {file:45s} - COMPLETE!")
                completed_files += 1
            else:
                print(f"  🔧 {file:45s} - {todo_count} TODOs remaining")
                total_todos += todo_count
                phase_complete = False
        
        phase_status[phase] = phase_complete
    
    # Summary
    print("\n" + "="*70)
    print("📊 IMPLEMENTATION SUMMARY")
    print("="*70)
    print(f"Files completed: {completed_files}/{total_files}")
    print(f"Remaining TODOs: {total_todos}")
    print(f"Progress: {completed_files/total_files*100:.1f}%")
    
    # Check for outputs/artifacts
    print("\n" + "="*70)
    print("📁 OUTPUT FILES & ARTIFACTS")
    print("="*70)
    
    for name, path in EXPECTED_OUTPUTS.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                # Count subdirectories (people)
                people = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if people:
                    print(f"  ✅ {name:30s} - exists ({len(people)} people)")
                else:
                    print(f"  ⚠️  {name:30s} - exists but empty")
            else:
                # File exists
                size_kb = os.path.getsize(path) / 1024
                print(f"  ✅ {name:30s} - exists ({size_kb:.1f} KB)")
        else:
            print(f"  ⏳ {name:30s} - not created yet")
    
    # Next steps
    print("\n" + "="*70)
    print("🎯 NEXT STEPS")
    print("="*70)
    
    if completed_files == 0:
        print("\n📚 Getting Started:")
        print("   1. Read LEARNING_GUIDE.md (concepts and instructions)")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Start Phase 1: Open models/face_model.py")
        print("   4. Implement the 3 TODOs with guidance from LEARNING_GUIDE.md")
        
    elif phase_status.get("Phase 1: Load Pretrained Model", False):
        if not phase_status.get("Phase 2A: Capture Face Photos", False):
            print("\n📸 Phase 1 Complete! Next:")
            print("   1. Open utils/face_detector.py and implement TODOs")
            print("   2. Open data/face_capture.py and implement TODOs")
            print("   3. Run: python data/face_capture.py")
            print("   4. Capture 20 photos for each team member")
            
        elif not phase_status.get("Phase 2B: Generate Reference Database", False):
            print("\n💾 Phase 2A Complete! Next:")
            print("   1. Open core/generate_embeddings.py")
            print("   2. Implement the TODOs")
            print("   3. Run: python core/generate_embeddings.py")
            print("   4. This creates your reference database")
            
        elif not phase_status.get("Phase 3: Real-Time Recognition", False):
            print("\n🎥 Phase 2B Complete! Next:")
            print("   1. Open core/face_recognizer.py")
            print("   2. Implement the TODOs")
            print("   3. Run: python core/face_recognizer.py")
            print("   4. Test real-time recognition!")
            
    elif completed_files == total_files:
        print("\n🎉 CONGRATULATIONS! All implementations complete!")
        print("\n   Your face + emotion recognition system is ready!")
        print("\n   ✅ What you've built:")
        print("      Semester 1:")
        print("      - Pretrained face embedding model (MobileFaceNet)")
        print("      - Face detection system (YuNet)")
        print("      - Reference database with team faces")
        print("      - Real-time identity recognition")
        print("      Semester 2:")
        print("      - Emotion classification model (MobileNet)")
        print("      - Temporal smoothing for stable output")
        print("      - Parallel identity + emotion pipeline")
        print("\n   🚀 Optional next steps:")
        print("      - Deploy to Jetson Nano (see LEARNING_GUIDE.md Phase 4)")
        print("      - Integrate with Arduino for physical actions")
        print("      - Tune thresholds in configs/config.yaml")
        print("      - Add emotion-based robot behaviors")
        
    else:
        # Check if Semester 1 is complete
        sem1_complete = all(
            phase_status.get(phase, False) 
            for phase in TODO_FILES_SEM1.keys()
        )
        
        if sem1_complete:
            print("\n🎓 Semester 1 Complete! Moving to Semester 2...")
            print("\n   📖 Read SEMESTER_2_GUIDE.md for instructions")
            print("\n   Next steps:")
            print("   1. Download emotion model:")
            print('      curl -L -o assets/mobilenet_7.onnx \\')
            print('        "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/mobilenet_7.onnx?raw=true"')
            print("   2. Open models/emotion_model.py")
            print("   3. Implement TODOs 14-17")
        else:
            print(f"\n📝 Keep going! {total_files - completed_files} files remaining")
            print("   Follow LEARNING_GUIDE.md for Semester 1 instructions")
    
    print("\n" + "="*70)
    print("\n💡 Tips:")
    print("   • Semester 1: Read LEARNING_GUIDE.md")
    print("   • Semester 2: Read SEMESTER_2_GUIDE.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        check_implementation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running checker: {e}")
        print("Please report this issue if it persists.")

