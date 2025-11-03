"""
Main training script for face recognition model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.resnet_arcface import ResNetArcFace, count_parameters
from src.models.losses import ArcFaceLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Returns:
        avg_loss (float): Average loss over epoch
        accuracy (float): Training accuracy
    """
    model.train()
    
    # TODO: Initialize tracking variables
    # running_loss, correct, total (all start at 0)
    
    # TODO: Create progress bar using tqdm
    # pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    # TODO: Loop through batches
    # for images, labels in pbar:
    #     1. Move images and labels to device
    #     2. Forward pass: get embeddings and logits from model
    #     3. Compute loss using criterion(logits, labels)
    #     4. Zero gradients: optimizer.zero_grad()
    #     5. Backward pass: loss.backward()
    #     6. Update weights: optimizer.step()
    #     7. Update metrics:
    #        - Add loss.item() to running_loss
    #        - Get predictions: _, predicted = logits.max(1)
    #        - Count correct: predicted.eq(labels).sum().item()
    #        - Update total count
    #     8. Update progress bar with current loss and accuracy
    #        Use pbar.set_postfix({'loss': ..., 'acc': ...})
    
    # TODO: Compute and return average loss and accuracy
    # avg_loss = running_loss / len(dataloader)
    # accuracy = correct / total
    pass


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate model on validation set.
    
    Returns:
        avg_loss (float): Average validation loss
        accuracy (float): Validation accuracy
    """
    model.eval()
    
    # TODO: Initialize tracking variables (same as training)
    
    # TODO: Disable gradient computation
    # Use: with torch.no_grad():
    
    # TODO: Loop through validation batches (similar to training)
    # Key difference: No optimizer.zero_grad(), loss.backward(), optimizer.step()
    # Just: forward pass → compute loss → update metrics
    
    # TODO: Return average loss and accuracy
    pass


def main():
    """Main training function"""
    print("="*70)
    print("Face Recognition Training")
    print("="*70)
    
    # TODO: Load configuration from configs/config.yaml
    # Use yaml.safe_load()
    
    # TODO: Setup device
    # Use torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Print which device is being used
    
    # TODO: Create dataloaders using create_dataloaders(config)
    # Returns: train_loader, val_loader, class_names
    # Print class names
    
    # TODO: Initialize model
    # Create ResNetArcFace with parameters from config:
    #   - num_classes from config['data']['num_classes']
    #   - embedding_dim from config['model']['embedding_dim']
    #   - pretrained from config['model']['pretrained']
    #   - freeze_backbone from config['model']['freeze_backbone']
    # Move model to device
    # Print parameter count using count_parameters(model)
    
    # TODO: Count trainable vs frozen parameters
    # This helps verify backbone freezing worked correctly
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_params = sum(p.numel() for p in model.parameters())
    # Print both counts to show the difference
    # With frozen backbone: ~11M total, ~4M trainable (only embedding + head)
    
    # TODO: Create loss criterion
    # Create ArcFaceLoss with:
    #   - margin from config['arcface']['margin']
    #   - scale from config['arcface']['scale']
    
    # TODO: Create optimizer (IMPORTANT: only for trainable parameters)
    # When backbone is frozen, we should only optimize trainable parameters
    # Option 1 (simpler): Pass model.parameters() - optimizer will skip frozen params automatically
    # Option 2 (explicit): Filter trainable params first
    #   trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Create Adam optimizer with:
    #   - model.parameters() or trainable_params
    #   - lr from config['training']['learning_rate']
    #   - weight_decay from config['training']['weight_decay']
    
    # TODO: Create learning rate scheduler
    # Check config['training']['lr_scheduler']:
    #   - If 'step': use StepLR with step_size and gamma from config
    #   - If 'cosine': use CosineAnnealingLR with T_max=epochs
    
    # TODO: Initialize training loop variables
    # best_val_acc = 0.0
    # Create checkpoint directory from config['paths']['checkpoint_dir']
    # Use Path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    # TODO: Training loop
    # for epoch in range(1, config['training']['epochs'] + 1):
    #     1. Call train_one_epoch() to get train_loss, train_acc
    #     2. Call validate() to get val_loss, val_acc
    #     3. Call scheduler.step() to update learning rate
    #     4. Print epoch summary:
    #        - Epoch number
    #        - Train loss and accuracy
    #        - Val loss and accuracy
    #        - Current learning rate (optimizer.param_groups[0]['lr'])
    #     5. Check if val_acc > best_val_acc:
    #        - Update best_val_acc
    #        - Save checkpoint using torch.save():
    #          Save dictionary with: epoch, model_state_dict, 
    #          optimizer_state_dict, val_acc, class_names
    #        - Print confirmation message
    #     6. Print separator line
    
    # TODO: Print final summary
    # Print best validation accuracy achieved
    print("="*70)


if __name__ == '__main__':
    main()

