"""
Installation Test Script

Run this immediately after 'pip install -r requirements.txt' to verify
all dependencies are correctly installed.

This should take ~5 seconds to run.
"""

import sys

print("="*70)
print("🔍 Testing Installation")
print("="*70)
print()

# Test each critical dependency
dependencies = []

# 1. Test NumPy
try:
    import numpy as np
    version = np.__version__
    dependencies.append(("NumPy", "✅", version))
    print(f"✅ NumPy {version}")
except ImportError as e:
    dependencies.append(("NumPy", "❌", str(e)))
    print(f"❌ NumPy - {e}")

# 2. Test OpenCV
try:
    import cv2
    version = cv2.__version__
    dependencies.append(("OpenCV", "✅", version))
    print(f"✅ OpenCV {version}")
except ImportError as e:
    dependencies.append(("OpenCV", "❌", str(e)))
    print(f"❌ OpenCV - {e}")

# 3. Test InsightFace
try:
    import insightface
    version = insightface.__version__
    dependencies.append(("InsightFace", "✅", version))
    print(f"✅ InsightFace {version}")
except ImportError as e:
    dependencies.append(("InsightFace", "❌", str(e)))
    print(f"❌ InsightFace - {e}")

# 4. Test ONNX Runtime
try:
    import onnxruntime
    version = onnxruntime.__version__
    dependencies.append(("ONNX Runtime", "✅", version))
    print(f"✅ ONNX Runtime {version}")
except ImportError as e:
    dependencies.append(("ONNX Runtime", "❌", str(e)))
    print(f"❌ ONNX Runtime - {e}")

# 5. Test PyYAML
try:
    import yaml
    version = yaml.__version__
    dependencies.append(("PyYAML", "✅", version))
    print(f"✅ PyYAML {version}")
except ImportError as e:
    dependencies.append(("PyYAML", "❌", str(e)))
    print(f"❌ PyYAML - {e}")

# 6. Test PIL/Pillow
try:
    from PIL import Image
    import PIL
    version = PIL.__version__
    dependencies.append(("Pillow", "✅", version))
    print(f"✅ Pillow {version}")
except ImportError as e:
    dependencies.append(("Pillow", "❌", str(e)))
    print(f"❌ Pillow - {e}")

# 7. Test tqdm
try:
    import tqdm
    version = tqdm.__version__
    dependencies.append(("tqdm", "✅", version))
    print(f"✅ tqdm {version}")
except ImportError as e:
    dependencies.append(("tqdm", "❌", str(e)))
    print(f"❌ tqdm - {e}")

print()
print("="*70)
print("📊 Summary")
print("="*70)

# Count successes and failures
successes = sum(1 for _, status, _ in dependencies if status == "✅")
failures = sum(1 for _, status, _ in dependencies if status == "❌")

print(f"\n✅ Installed: {successes}/{len(dependencies)}")
print(f"❌ Missing: {failures}/{len(dependencies)}")

if failures == 0:
    print("\n🎉 All dependencies installed successfully!")
    print("\n✅ You're ready to start the project!")
    print("\nNext steps:")
    print("   1. Open LEARNING_GUIDE.md")
    print("   2. Read the concepts section")
    print("   3. Start Phase 1: models/face_model.py")
    print("\nOr run: python check_progress.py")
else:
    print("\n⚠️  Some dependencies are missing!")
    print("\nTo fix:")
    print("   1. Make sure you're in a virtual environment")
    print("   2. Run: pip install -r requirements.txt")
    print("   3. Run this test again: python test_installation.py")
    
    print("\nMissing packages:")
    for name, status, info in dependencies:
        if status == "❌":
            print(f"   ❌ {name}")

print("\n" + "="*70)

# Exit with appropriate code
sys.exit(0 if failures == 0 else 1)

