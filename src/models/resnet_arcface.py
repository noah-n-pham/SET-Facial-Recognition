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
        
        # Load pretrained ResNet-18 and extract feature layers
        resnet = resnet18(pretrained=pretrained)
        # Remove the final fully connected layer
        # This gives features of shape [B, 512, 1, 1]
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze the backbone if freeze_backbone is True
        # This implements transfer learning with frozen features
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            print(f"✅ Backbone frozen - ResNet-18 weights will not be updated during training")
        
        # Create embedding layer
        self.embedding = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Create ArcFace classification head
        self.fc = nn.Linear(embedding_dim, num_classes, bias=False)
        
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
        # Extract features from ResNet backbone
        features = self.features(x)  # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten to [B, 512]
        
        # Generate embeddings
        embeddings = self.embedding(features)
        embeddings = self.bn(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
        
        # Generate classification logits
        logits = self.fc(embeddings)
        
        return embeddings, logits
    
    def extract_embedding(self, x):
        """
        Extract only embeddings (for inference).
        
        Args:
            x (torch.Tensor): Input images [B, 3, 224, 224]
        
        Returns:
            embeddings (torch.Tensor): L2-normalized embeddings [B, embedding_dim]
        """
        embeddings, _ = self.forward(x)
        return embeddings


def count_parameters(model):
    """Count trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

