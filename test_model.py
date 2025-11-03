"""
Test script for verifying model architecture.
Tests both model creation and parameter freezing.
"""
from src.models.resnet_arcface import ResNetArcFace, count_parameters
import torch

print("="*70)
print("Model Architecture Test")
print("="*70)

# Test 1: Create model with frozen backbone
print("\n1. Creating model with frozen backbone...")
model = ResNetArcFace(num_classes=9, embedding_dim=512, freeze_backbone=True)

# Test 2: Count parameters
print("\n2. Counting parameters...")
total_params = count_parameters(model)
print(f"   Total parameters: {total_params:,}")

# Test 3: Count trainable vs frozen parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {frozen_params:,}")
print(f"   Trainable %: {trainable_params/total_params*100:.1f}%")

# Expected values for frozen backbone approach:
# Total: ~11M, Trainable: ~264K, Frozen: ~10.7M
expected_trainable_range = (200_000, 300_000)
if expected_trainable_range[0] <= trainable_params <= expected_trainable_range[1]:
    print("   ✅ Backbone appears to be FROZEN (correct!)")
else:
    print("   ⚠️  Unexpected trainable count - check backbone freezing")

# Test 4: Test forward pass
print("\n3. Testing forward pass...")
dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2
embeddings, logits = model(dummy_input)
print(f"   Input shape: {dummy_input.shape}")
print(f"   Embeddings shape: {embeddings.shape}")  # Should be [2, 512]
print(f"   Logits shape: {logits.shape}")  # Should be [2, 9]

if embeddings.shape == (2, 512) and logits.shape == (2, 9):
    print("   ✅ Output shapes correct!")
else:
    print("   ❌ Output shapes incorrect")

# Test 5: Verify embeddings are normalized
embedding_norms = torch.norm(embeddings, p=2, dim=1)
print(f"\n4. Embedding L2 norms: {embedding_norms}")
if torch.allclose(embedding_norms, torch.ones(2), atol=1e-5):
    print("   ✅ Embeddings are L2-normalized!")
else:
    print("   ⚠️  Embeddings should be L2-normalized")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
