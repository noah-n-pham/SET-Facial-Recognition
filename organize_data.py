import os
import shutil
import random

# Set paths
source_dir = "Dataset"
target_dir = "data"
train_ratio = 0.8  # 80% training, 20% validation

# Create target structure
for split in ['train', 'val']:
    split_path = os.path.join(target_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Process each class folder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class folders in train and val
    for split, split_images in [('train', train_images), ('val', val_images)]:
        split_class_dir = os.path.join(target_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy2(src, dst)

print("✅ Dataset reorganized successfully.")