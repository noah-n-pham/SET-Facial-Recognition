"""
Face Recognition Progress Checker

Run this script to see which TODOs you've completed and what's remaining.
This tracks progress through the TODO-based learning approach.
"""
import os
import re
from pathlib import Path

# Files with TODOs in the new pretrained approach
TODO_FILES = {
    "Phase 1: Load Pretrained Model": [
        "models/face_model.py",
    ],
    "Phase 2A: Capture Face Photos": [
        "utils/face_detector.py",
        "data/face_capture.py",
    ],
    "Phase 2B: Generate Reference Database": [
        "core/generate_embeddings.py",
    ],
    "Phase 3: Real-Time Recognition": [
        "core/face_recognizer.py",
    ],
}

# Expected outputs at each phase
EXPECTED_OUTPUTS = {
    "Reference Database": "models/reference_embeddings.npy",
    "Label Names": "models/label_names.txt",
    "Face Photos": "data/raw/Dataset",
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
    print("🔍 FACIAL RECOGNITION PROGRESS CHECKER")
    print("="*70)
    print("\nThis checker tracks your progress through the TODO-based learning.")
    print("Follow LEARNING_GUIDE.md for step-by-step instructions.\n")
    
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
        print("\n   Your face recognition system is ready!")
        print("\n   ✅ What you've built:")
        print("      - Pretrained face embedding model")
        print("      - Face detection system")
        print("      - Reference database with team faces")
        print("      - Real-time recognition via webcam")
        print("\n   🚀 Optional next steps:")
        print("      - Deploy to Jetson Nano (see LEARNING_GUIDE.md Phase 4)")
        print("      - Integrate with Arduino for physical actions")
        print("      - Tune similarity threshold in configs/config.yaml")
        print("      - Add more people to your database")
        
    else:
        print(f"\n📝 Keep going! {total_files - completed_files} files remaining")
        print("   Follow LEARNING_GUIDE.md for detailed instructions")
    
    print("\n" + "="*70)
    print("\n💡 Tip: Read LEARNING_GUIDE.md for concept explanations and guidance!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        check_implementation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running checker: {e}")
        print("Please report this issue if it persists.")

