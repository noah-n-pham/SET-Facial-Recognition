"""
Test script for verifying loss computation.
"""
import torch
from src.models.losses import ArcFaceLoss

criterion = ArcFaceLoss(margin=0.5, scale=64.0)
dummy_logits = torch.randn(4, 9)  # 4 samples, 9 classes
dummy_labels = torch.tensor([0, 1, 2, 3])

loss = criterion(dummy_logits, dummy_labels)
print(f"Loss: {loss.item():.4f}")  # Should be positive scalar

