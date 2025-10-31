import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# -------------------------------
# 1️⃣ Set up device (GPU if available)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2️⃣ Data transforms and loaders
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    ),
])

train_dataset = datasets.ImageFolder(root="data/train", transform=transform)
val_dataset   = datasets.ImageFolder(root="data/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------------------
# 3️⃣ Load pretrained ResNet model
# -------------------------------
model = models.resnet18(weights="IMAGENET1K_V1")

# Modify the final layer for your number of classes
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# -------------------------------
# 4️⃣ Define loss function & optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()              # classification loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer
# (You could also use SGD with momentum if you prefer)

# -------------------------------
# 5️⃣ Training loop
# -------------------------------
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()  # set to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total

    print(f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")

    # -------------------------------
    # 🔍 Validation step
    # -------------------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * val_correct / val_total
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

# -------------------------------
# 6️⃣ Save the trained model
# -------------------------------
torch.save(model.state_dict(), "resnet18_custom.pth")
print("Model saved to resnet18_custom.pth ✅")

