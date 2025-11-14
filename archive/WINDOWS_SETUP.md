# Windows Setup Guide

## ⚠️ Common Windows Installation Issues

If you're getting "Could not find torch" errors on Windows, follow this guide.

---

## Step 1: Check Python Version

```powershell
python --version
```

**Requirements:**
- ✅ Python 3.8, 3.9, 3.10, 3.11, or 3.12
- ✅ 64-bit version (not 32-bit)
- ❌ Python 3.7 or older won't work
- ❌ Python 3.13+ not yet fully supported

**Check if 64-bit:**
```powershell
python -c "import struct; print(struct.calcsize('P') * 8)"
```
Should print `64` (not `32`)

---

## Step 2: Install Correct Python (if needed)

If your Python version is wrong:

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11.x** (recommended)
3. **Important**: Check "Add Python to PATH" during installation
4. Choose "Install for all users"
5. Restart PowerShell/Command Prompt after installation

Verify:
```powershell
python --version
# Should show: Python 3.11.x
```

---

## Step 3: Create Virtual Environment

```powershell
# Navigate to project folder
cd path\to\Facial-Recognition

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# You should see (venv) in your prompt
```

---

## Step 4: Install Dependencies

```powershell
# Upgrade pip first (important!)
python -m pip install --upgrade pip

# Install PyTorch with CPU support (Windows needs explicit URL)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**Why the special PyTorch command?**
- Windows PyPI doesn't host PyTorch directly
- Must use PyTorch's own index URL
- `--index-url` tells pip where to find PyTorch packages

---

## Step 5: Verify Installation

```powershell
# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Run GPU test
python test_gpu.py
# Expected: Shows CPU or GPU status
```

---

## Common Windows Errors & Solutions

### Error: "python: command not found"
**Solution**: Python not in PATH
```powershell
# Use full path, or reinstall Python with "Add to PATH" checked
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe -m venv venv
```

### Error: "pip: command not found"
**Solution**: Use `python -m pip` instead
```powershell
python -m pip install --upgrade pip
```

### Error: "Could not find a version that satisfies the requirement torch"
**Solution**: Your Python is 32-bit or wrong version
1. Check: `python -c "import struct; print(struct.calcsize('P') * 8)"`
2. If prints 32, install 64-bit Python
3. Restart PowerShell after installing

### Error: "cannot be loaded because running scripts is disabled"
**Solution**: PowerShell execution policy
```powershell
# Option 1: Use Command Prompt instead
cmd
venv\Scripts\activate

# Option 2: Change policy (requires admin)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: "Microsoft Visual C++ 14.0 or greater is required"
**Solution**: Some packages need C++ compiler
1. Download "Microsoft C++ Build Tools" from microsoft.com
2. Or install Visual Studio Community (free)
3. Restart and try again

---

## GPU Support (Optional)

If you have an NVIDIA GPU and want faster training:

```powershell
# After basic installation, upgrade to GPU version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install GPU ONNX Runtime
pip install onnxruntime-gpu>=1.15.0
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA drivers installed
- Windows 10/11

Test GPU:
```powershell
python test_gpu.py
# Should show: "✅ GPU is ready for training!"
```

---

## Quick Reference

```powershell
# Full installation from scratch
python --version  # Check version
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python test_gpu.py
```

---

## Still Having Issues?

1. **Restart PowerShell** after any Python installation
2. **Use Command Prompt** instead of PowerShell if activation fails
3. **Check antivirus** - some block pip installations
4. **Use Python 3.11** - most stable for this project
5. **Ask for help** - include the error message and Python version

---

**Next Steps:** Once installation works, return to QUICK_START.md Phase 1!

