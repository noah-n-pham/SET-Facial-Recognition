"""
Export trained PyTorch model to ONNX format.
ONNX is platform-independent and can be optimized with TensorRT on Jetson.
"""
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from src.models.resnet_arcface import ResNetArcFace


def export_to_onnx(checkpoint_path, output_path, embedding_dim=512, num_classes=9):
    """
    Export model to ONNX format.
    
    Args:
        checkpoint_path (str): Path to PyTorch checkpoint (.pth)
        output_path (str): Output path for ONNX model (.onnx)
        embedding_dim (int): Embedding dimension
        num_classes (int): Number of classes
    """
    print("="*70)
    print("Exporting Model to ONNX")
    print("="*70)
    
    # TODO: Setup device (use CPU for export compatibility)
    
    # TODO: Create model
    # Initialize ResNetArcFace with pretrained=False (already trained)
    
    # TODO: Load checkpoint weights
    # Use torch.load() with map_location=device
    # Load model_state_dict from checkpoint
    # Set model to eval mode
    
    # TODO: Create dummy input tensor
    # Shape should match real input: [1, 3, 224, 224]
    # Use torch.randn()
    
    print(f"\nExporting model...")
    
    # TODO: Export to ONNX
    # Use torch.onnx.export() with:
    #   - model
    #   - dummy_input
    #   - output_path
    #   - input_names=['input']
    #   - output_names=['embeddings', 'logits']
    #   - dynamic_axes for batch dimension flexibility
    #   - opset_version=11 (compatible with most runtimes)
    #   - do_constant_folding=True (optimization)
    
    print(f"✅ Exported to: {output_path}")
    
    print("\nVerifying ONNX model...")
    
    # TODO: Load and verify ONNX model
    # Use onnx.load() and onnx.checker.check_model()
    # Print "✅ ONNX model is valid"
    
    print("\nTesting ONNX inference...")
    
    # TODO: Test inference with ONNX Runtime
    # 1. Create InferenceSession with output_path
    # 2. Prepare input dictionary: {input_name: dummy_input.numpy()}
    # 3. Run inference: session.run(None, input_dict)
    # 4. Print output shapes
    
    print("\nComparing PyTorch vs ONNX outputs...")
    
    # TODO: Compare outputs between PyTorch and ONNX
    # 1. Get PyTorch output with torch.no_grad()
    # 2. Get ONNX output (already computed above)
    # 3. Compute max absolute difference for embeddings and logits
    # 4. Print differences
    # 5. Check if differences < 1e-4 (should match closely)


if __name__ == '__main__':
    export_to_onnx(
        checkpoint_path='models/checkpoints/best_model.pth',
        output_path='models/exported/face_recognition.onnx',
        embedding_dim=512,
        num_classes=9
    )

