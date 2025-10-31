import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image

# --- Load your trained model ---
model = models.resnet18(pretrained=False)
num_classes = 2  # 🔁 change this to however many classes you trained on
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_custom.pth", map_location=torch.device('cpu')))
model.eval()

# --- Load class names (optional) ---
try:
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = [f"Class {i}" for i in range(num_classes)]

# --- Define transforms for incoming frames ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ResNet expected normalization
        std=[0.229, 0.224, 0.225]
    )
])

# --- Start webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame for model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = transform(pil_img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        label = class_names[preds.item()]

    # Display label
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("ResNet18 Webcam Inference", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


