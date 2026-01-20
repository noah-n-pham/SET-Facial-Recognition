"""
PyTorch Dataset for loading face images with augmentation.
Splits data into train/val and applies different transforms to each.
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .augmentation import get_train_transforms, get_val_transforms


class FaceDataset(Dataset):
    """
    Face recognition dataset with on-the-fly augmentation.
    
    Args:
        root_dir (str): Path to Dataset folder (e.g., 'data/raw/Dataset')
        split (str): 'train' or 'val'
        train_ratio (float): Fraction of data for training (default: 0.8)
        random_seed (int): Seed for reproducible splits (default: 42)
    """
    
    def __init__(self, root_dir, split='train', train_ratio=0.8, random_seed=42):
        self.root_dir = Path(root_dir)
        self.split = split
        
        # TODO: Initialize empty lists for storing data
        self.image_paths = [] 
        self.labels = [] 
        self.label_to_name = {}
        self.name_to_label = {}
        
        # TODO: Build dataset by looping through person folders
        # Steps:
        # 1. Get all subdirectories in root_dir (these are person folders)
        dataset_path = self.root_dir
        names = sorted([folder.name.lower() 
                 for folder in dataset_path.iterdir() 
                 if folder.is_dir()]) #a
        
        for id, name in enumerate(names):
            self.label_to_name[id] = name #b
            self.name_to_label[name] = id #c 
              
        for person in names:
            folder = dataset_path / person
            image_paths_person = [
                str(img)
                for img in folder.glob("*.png")  # matches .png 
            ]
            self.image_paths.extend(image_paths_person)

            for i in range(len(image_paths_person)):
                self.labels.append(self.name_to_label[person]) #adds the matching person index to the labels list
        # 2. Sort them alphabetically for consistent label assignment
        # 3. For each folder (with enumerate to get label_id):
        #    a. Get person name from folder.name
        #    b. Store in label_to_name[label_id] = person_name
        #    c. Store in name_to_label[person_name] = label_id
        #    d. Find all .png images in this folder
        #    e. For each image, append str(image_path) to self.image_paths
        #    f. For each image, append label_id to self.labels
        
        
        # TODO: Split into train/val sets
        # Use sklearn.model_selection.train_test_split with:
        #   - X=self.image_paths, y=self.labels
        #   - train_size=train_ratio
        #   - random_state=random_seed
        #   - stratify=self.labels (keeps class distribution balanced)
        # This returns: train_paths, val_paths, train_labels, val_labels
        # Based on self.split, keep only train or val data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.image_paths,
            self.labels,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=self.labels
        )
        
        if self.split == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        else:
            self.image_paths = val_paths
            self.labels = val_labels

        # TODO: Set transforms based on split
        # If split is 'train', use get_train_transforms()
        # If split is 'val', use get_val_transforms()
        if self.split == 'train':
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()
        
        # TODO: Print summary
        # Print: "{split} dataset: {num_images} images, {num_classes} classes"
        print(f"{self.split} dataset: {len(self.image_paths)} images, {len(self.label_to_name)} classes")
    
    def __len__(self):
        """Return total number of images in this split"""
        # TODO: Return length of self.image_paths
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and transform one image.
        
        Returns:
            image (torch.Tensor): Shape [3, 224, 224], normalized
            label (int): Class ID (0-8)
        """
        # Load image at index idx
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation transforms
        image = self.transform(image=image)['image']
        
        # Get corresponding label
        label = self.labels[idx]
        
        return image, label
    
    def get_class_names(self):
        """Return list of person names in order of label IDs"""
        return [self.label_to_name[i] for i in range(len(self.label_to_name))]


def create_dataloaders(config):
    """
    Create train and validation DataLoaders.
    
    Args:
        config (dict): Config dictionary from config.yaml
    
    Returns:
        train_loader, val_loader, class_names
    """
    # Extract parameters from config
    dataset_path = config['data']['dataset_path']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    train_split = config['data']['train_split']
    
    # Create train and val datasets
    train_dataset = FaceDataset(dataset_path, split='train', train_ratio=train_split)
    val_dataset = FaceDataset(dataset_path, split='val', train_ratio=train_split)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class names from train_dataset
    class_names = train_dataset.get_class_names()
    
    return train_loader, val_loader, class_names
