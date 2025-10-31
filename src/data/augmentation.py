"""
Data augmentation transforms for facial recognition training.

This module defines Albumentations pipelines for training and validation.
Augmentations are applied on-the-fly during training, not saved to disk.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    """
    Get training augmentation pipeline with aggressive augmentations.
    Applied randomly during training to increase dataset diversity.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(std_range=(0.05, 0.05), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomGamma(p=0.3),
        A.CLAHE(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Get validation augmentation pipeline with minimal augmentations.
    Only resize and normalize, no random augmentations.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

