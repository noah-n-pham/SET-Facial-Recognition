"""
ResNet-18 backbone with ArcFace classification head for face recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNetArcFace(nn.Module):
    """
    Face recognition model with embedding extraction.
    Uses FROZEN BACKBONE + TRAINABLE HEAD approach for efficient transfer learning.
    
    Architecture:
        Input [B, 3, 224, 224] → ResNet-18 backbone (FROZEN) → Embedding [B, 512] (TRAINABLE)
        → ArcFace head (TRAINABLE) → Logits [B, num_classes]
    
    Why freeze the backbone?
    - ResNet-18 pretrained on ImageNet already learned excellent visual features
    - Freezing prevents overfitting on small face datasets
    - Trains much faster (fewer parameters to update)
    - Uses less GPU memory
    - Only the task-specific layers (embedding + head) need to adapt to faces
    
    Args:
        num_classes (int): Number of people to recognize (9 in your case)
        embedding_dim (int): Dimension of face embeddings (default: 512)
        pretrained (bool): Use ImageNet pretrained weights (default: True)
        freeze_backbone (bool): Freeze ResNet-18 backbone weights (default: True)
    """
    
    def __init__(self, num_classes=9, embedding_dim=512, pretrained=True, freeze_backbone=True):
        super(ResNetArcFace, self).__init__()
        
        # TODO: Load pretrained ResNet-18 and extract feature layers
        # 1. Load resnet18(pretrained=pretrained) from torchvision
        # 2. Remove the final fully connected layer
        #    Hint: Use nn.Sequential(*list(resnet.children())[:-1])
        #    This gives features of shape [B, 512, 1, 1]
        # 3. Store as self.features
        
        # TODO: Freeze the backbone if freeze_backbone is True
        # This implements transfer learning with frozen features
        # Loop through self.features.parameters() and set requires_grad = False
        # Why? Pretrained ResNet already learned good visual features from ImageNet.
        # We only need to train the embedding layer and ArcFace head for face recognition.
        # This is faster, uses less memory, and prevents overfitting on small datasets.
        #
        # if freeze_backbone:
        #     for param in self.features.parameters():
        #         param.requires_grad = False
        
        # TODO: Create embedding layer
        # 1. Create nn.Linear(512, embedding_dim) - stores as self.embedding
        # 2. Create nn.BatchNorm1d(embedding_dim) - stores as self.bn
        # BatchNorm stabilizes training by normalizing activations
        # Note: These layers ARE trainable (not frozen)
        
        # TODO: Create ArcFace classification head
        # Create nn.Linear(embedding_dim, num_classes, bias=False)
        # Store as self.fc
        # Note: No bias because we'll use cosine similarity
        # Note: This layer IS trainable (not frozen)
        
    def forward(self, x, labels=None):
        """
        Forward pass through model.
        
        Args:
            x (torch.Tensor): Input images [B, 3, 224, 224]
            labels (torch.Tensor, optional): Ground truth labels for training
        
        Returns:
            embeddings (torch.Tensor): L2-normalized embeddings [B, embedding_dim]
            logits (torch.Tensor): Classification scores [B, num_classes]
        """
        # TODO: Extract features from ResNet backbone
        # 1. Pass x through self.features
        # 2. Result will be [B, 512, 1, 1]
        # 3. Flatten to [B, 512] using .view(features.size(0), -1)
        
        # TODO: Generate embeddings
        # 1. Pass features through self.embedding layer
        # 2. Pass through self.bn (batch normalization)
        # 3. L2 normalize using F.normalize(embeddings, p=2, dim=1)
        #    This ensures all embeddings have unit length (important for cosine similarity)
        
        # TODO: Generate classification logits
        # Pass normalized embeddings through self.fc
        
        # TODO: Return (embeddings, logits)
        pass
    
    def extract_embedding(self, x):
        """
        Extract only embeddings (for inference).
        
        Args:
            x (torch.Tensor): Input images [B, 3, 224, 224]
        
        Returns:
            embeddings (torch.Tensor): L2-normalized embeddings [B, embedding_dim]
        """
        # TODO: Call forward() and return only the embeddings
        # Hint: embeddings, _ = self.forward(x)
        pass


def count_parameters(model):
    """Count trainable parameters"""
    # TODO: Sum all trainable parameters
    # Loop through model.parameters()
    # Check if p.requires_grad
    # Sum p.numel() for each trainable parameter
    pass

