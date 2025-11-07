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
        # Geometric transforms
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        
        # Color and lighting transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        
        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        
        # Normalization
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

