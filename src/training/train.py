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
    
    # Initialize tracking variables
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar using tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    # Loop through batches
    for images, labels in pbar:
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass: get embeddings and logits from model
        embeddings, logits = model(images, labels)
        
        # Compute loss using criterion
        loss = criterion(logits, labels)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar with current loss and accuracy
        accuracy = 100. * correct / total
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})
    
    # Compute and return average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate model on validation set.
    
    Returns:
        avg_loss (float): Average validation loss
        accuracy (float): Validation accuracy
    """
    model.eval()
    
    # Initialize tracking variables
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        
        # Loop through validation batches
        for images, labels in pbar:
            # Move to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            embeddings, logits = model(images, labels)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})
    
    # Return average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    """Main training function"""
    print("="*70)
    print("Face Recognition Training")
    print("="*70)
    
    # Load configuration from configs/config.yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, class_names = create_dataloaders(config)
    print(f"Classes: {class_names}")
    
    # Initialize model
    model = ResNetArcFace(
        num_classes=config['data']['num_classes'],
        embedding_dim=config['model']['embedding_dim'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    model = model.to(device)
    
    # Count trainable vs frozen parameters
    trainable_params, total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100.*trainable_params/total_params:.1f}%)")
    print(f"Frozen parameters: {total_params - trainable_params:,} ({100.*(total_params-trainable_params)/total_params:.1f}%)")
    
    # Create loss criterion
    criterion = ArcFaceLoss(
        margin=config['arcface']['margin'],
        scale=config['arcface']['scale']
    )
    
    # Create optimizer (only for trainable parameters)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    if config['training']['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=config['training']['lr_gamma']
        )
    elif config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # Initialize training loop variables
    best_val_acc = 0.0
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    # Training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{config['training']['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, checkpoint_path)
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        print("="*70)
    
    # Print final summary
    print(f"\n🎉 Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()

