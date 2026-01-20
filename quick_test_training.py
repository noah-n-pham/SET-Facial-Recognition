"""
Quick overfit test: Train on a tiny subset to verify the pipeline works.
This should reach 95-100% accuracy quickly if everything is implemented correctly.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import FaceDataset
from src.models.resnet_arcface import ResNetArcFace, count_parameters
from src.models.losses import ArcFaceLoss

print("="*70)
print("Quick Overfit Test - Training on 20 images")
print("="*70)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create tiny dataset (just 20 images)
dataset = FaceDataset('data/raw/Dataset', split='train', train_ratio=0.8)
dataset.image_paths = dataset.image_paths[:20]
dataset.labels = dataset.labels[:20]
print(f"Dataset size: {len(dataset)} images")

# Create DataLoader
loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

# Setup model
model = ResNetArcFace(num_classes=9, embedding_dim=512, pretrained=True, freeze_backbone=True)
model = model.to(device)

trainable, total = count_parameters(model)
print(f"Trainable params: {trainable:,} / {total:,}\n")

# Setup training
criterion = ArcFaceLoss(margin=0.5, scale=64.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training for 30 epochs...")
print("-"*70)

# Training loop
for epoch in range(1, 31):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total_samples = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        embeddings, logits = model(images, labels)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        epoch_loss += loss.item()
        _, predicted = logits.max(1)
        total_samples += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total_samples
    avg_loss = epoch_loss / len(loader)
    
    if epoch % 5 == 0 or accuracy >= 95.0:
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 98.0:
        print(f"\n🎉 SUCCESS! Reached {accuracy:.2f}% accuracy at epoch {epoch}")
        print("✅ Training pipeline is working correctly!")
        break
else:
    print(f"\n⚠️  Final accuracy: {accuracy:.2f}%")
    print("Note: If accuracy is >90%, the pipeline works but may need more epochs")

print("="*70)

