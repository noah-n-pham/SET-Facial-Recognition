"""
Quick test: Train on just 5 images to verify model can overfit.
If accuracy reaches 100%, your training pipeline works correctly.
"""
import torch
import torch.optim as optim
from src.data.dataset import FaceDataset
from src.models.resnet_arcface import ResNetArcFace
from src.models.losses import ArcFaceLoss

# Create tiny dataset (first 5 images only)
dataset = FaceDataset('data/raw/Dataset', split='train')

# TODO: Reduce dataset to first 5 images
# Set dataset.image_paths = dataset.image_paths[:5]
# Set dataset.labels = dataset.labels[:5]

# TODO: Create DataLoader with batch_size=5, shuffle=True

# TODO: Setup model, criterion, optimizer
# device = cuda or cpu
# model = ResNetArcFace, move to device
# criterion = ArcFaceLoss
# optimizer = Adam with lr=0.001

print("Overfit test: Training on 5 images...")

# TODO: Training loop for 50 iterations
# for i in range(50):
#     for images, labels in loader:
#         1. Move to device
#         2. Forward pass
#         3. Compute loss
#         4. Backward pass
#         5. Update weights
#         6. Calculate accuracy
#         7. Every 10 iterations, print loss and accuracy

# TODO: Print final message
# If accuracy reached 100%, print success message

