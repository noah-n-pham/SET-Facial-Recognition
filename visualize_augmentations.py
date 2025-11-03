"""
Visualize augmented samples to verify transforms are reasonable.
"""
import yaml
import matplotlib.pyplot as plt
import torch
from src.data.dataset import FaceDataset

def visualize_augmentations(dataset, num_samples=5):
    """Show original and augmented versions of images"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Get same image twice (will get different augmentations)
        img1, label = dataset[i]
        img2, _ = dataset[i]
        
        # TODO: Denormalize images for display
        # Images are normalized with:
        #   mean = [0.485, 0.456, 0.406]
        #   std = [0.229, 0.224, 0.225]
        # To denormalize: img = img * std + mean
        # Create tensors for mean and std with shape [3, 1, 1]
        # Apply denormalization to img1 and img2
        
        # TODO: Convert tensors to numpy arrays for plotting
        # 1. Permute from [C, H, W] to [H, W, C] using .permute(1, 2, 0)
        # 2. Convert to numpy with .numpy()
        # 3. Clip values to [0, 1] range
        
        # TODO: Plot both versions
        # Use axes[0, i].imshow() for first augmentation
        # Use axes[1, i].imshow() for second augmentation
        # Set titles showing person label
        # Turn off axis ticks with axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('logs/augmentation_samples.png')
    print("✅ Saved visualization to logs/augmentation_samples.png")

if __name__ == "__main__":
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    
    train_ds = FaceDataset(config['data']['dataset_path'], split='train')
    visualize_augmentations(train_ds, num_samples=5)

