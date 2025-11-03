"""
Test script for verifying FaceDataset works correctly.
"""
import yaml
from src.data.dataset import FaceDataset

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

train_ds = FaceDataset(config['data']['dataset_path'], split='train')
val_ds = FaceDataset(config['data']['dataset_path'], split='val')

print(f"\nClass names: {train_ds.get_class_names()}")
img, label = train_ds[0]
print(f"Sample - Image shape: {img.shape}, Label: {label}, Range: [{img.min():.2f}, {img.max():.2f}]")

