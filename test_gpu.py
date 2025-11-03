"""
Quick test to verify PyTorch can see your GPU.
If CUDA is not available, training will be very slow.
"""
import torch
import sys

def check_gpu():
    print("="*50)
    print("PyTorch GPU Check")
    print("="*50)
    
    # TODO: Print PyTorch version
    # Use torch.__version__
    
    # TODO: Check if CUDA is available
    # Use torch.cuda.is_available() - returns True/False
    
    # TODO: If CUDA is available, print:
    #   - CUDA version (torch.version.cuda)
    #   - GPU device name (torch.cuda.get_device_name(0))
    #   - GPU memory in GB (torch.cuda.get_device_properties(0).total_memory / 1e9)
    #   - Print "✅ GPU is ready for training!"
    
    # TODO: If CUDA is NOT available, print:
    #   - "⚠️  No GPU detected!"
    #   - Warn that training will be SLOW
    #   - Suggest using GPU-enabled machine
    
    print("="*50)

if __name__ == "__main__":
    check_gpu()

