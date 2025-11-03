"""
Generate reference embeddings for all people in dataset.
These will be used for real-time recognition via similarity comparison.
"""
import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

from src.models.resnet_arcface import ResNetArcFace
from src.data.dataset import FaceDataset


def generate_reference_embeddings(checkpoint_path, output_dir='models'):
    """
    Extract one embedding per person for reference database.
    
    Args:
        checkpoint_path (str): Path to trained model checkpoint
        output_dir (str): Directory to save embeddings and labels
    """
    print("="*70)
    print("Generating Reference Embeddings")
    print("="*70)
    
    # TODO: Load config from configs/config.yaml
    
    # TODO: Setup device (cuda or cpu)
    
    # TODO: Initialize model
    # Create ResNetArcFace with parameters from config
    
    # TODO: Load trained checkpoint
    # Use torch.load(checkpoint_path, map_location=device)
    # Load state dict into model
    # Move model to device
    # Set model to eval mode
    
    # TODO: Load validation dataset
    # Create FaceDataset with split='val'
    # Get class names from dataset
    
    # TODO: Extract one embedding per person
    # Strategy: For each label_id (0 to num_classes-1):
    #   1. Find all indices in val_dataset that have this label
    #      Loop through val_dataset.labels and collect matching indices
    #   2. Take first 5 images for this person (or fewer if not available)
    #   3. For each image:
    #      a. Get image tensor from dataset[idx]
    #      b. Add batch dimension with .unsqueeze(0)
    #      c. Move to device
    #      d. Extract embedding with model.extract_embedding()
    #      e. Convert to numpy and store
    #   4. Average the embeddings for this person
    #   5. Re-normalize the averaged embedding (divide by L2 norm)
    #   6. Add to list of reference embeddings
    
    # TODO: Convert reference embeddings list to numpy array
    # Should have shape [num_classes, embedding_dim] = [9, 512]
    
    # TODO: Save reference embeddings
    # Use np.save(f'{output_dir}/reference_embeddings.npy', embeddings_array)
    
    # TODO: Save class names to text file
    # Write one name per line to '{output_dir}/label_names.txt'
    
    # TODO: Print confirmation with shape and class list
    print(f"\n✅ Saved reference embeddings")
    print(f"✅ Saved label names")


if __name__ == '__main__':
    generate_reference_embeddings(
        checkpoint_path='models/checkpoints/best_model.pth',
        output_dir='models'
    )

