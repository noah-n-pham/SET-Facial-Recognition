"""
ArcFace loss for improved face recognition.
Paper: https://arxiv.org/abs/1801.07698
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    
    Formula: L = -log(exp(s * cos(θ_yi + m)) / (exp(s * cos(θ_yi + m)) + Σ exp(s * cos(θ_j))))
    
    Where:
        - θ_yi is the angle between embedding and target class weight
        - m is the angular margin (makes classification harder)
        - s is the feature scale (enlarges decision boundaries)
    
    Args:
        margin (float): Angular margin in radians (default: 0.5)
        scale (float): Feature scale factor (default: 64.0)
    """
    
    def __init__(self, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        
        # Precompute cos(margin) and sin(margin) for numerical stability
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        # Precompute threshold values for stability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # Create CrossEntropyLoss criterion
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels):
        """
        Compute ArcFace loss.
        
        Args:
            logits (torch.Tensor): Cosine similarities [B, num_classes]
                Note: These are already cosine similarities because 
                embeddings and FC weights are L2-normalized
            labels (torch.Tensor): Ground truth class IDs [B]
        
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        # Clamp cosine values to avoid numerical issues
        cosine = logits.clamp(-1, 1)
        
        # Compute sine from cosine: sine = sqrt(1 - cosine^2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Apply angle addition formula: cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Create one-hot mask for target classes
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        
        # Replace target class logits with margin-added version
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Apply feature scale
        output = output * self.scale
        
        # Compute cross-entropy loss
        loss = self.criterion(output, labels)
        
        return loss


class SimplifiedLoss(nn.Module):
    """
    Simplified loss that just uses CrossEntropy on logits.
    Use this first to verify training works, then switch to ArcFace.
    """
    
    def __init__(self):
        super(SimplifiedLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        """Simple cross-entropy loss"""
        return self.criterion(logits, labels)

