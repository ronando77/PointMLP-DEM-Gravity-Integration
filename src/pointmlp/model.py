import torch
import torch.nn as nn
import torch.nn.functional as F


class PointMLP(nn.Module):
    """PointMLP model for point cloud classification and segmentation
    
    Based on: "Rethinking Network Design and Local Geometry in Point Cloud: 
    A Simple Residual MLP Framework" (ICLR 2022)
    """
    
    def __init__(self, num_classes=40, num_points=1024, in_channels=3):
        """Initialize PointMLP model
        
        Args:
            num_classes: Number of output classes
            num_points: Number of points in point cloud
            in_channels: Number of input channels (default 3 for xyz)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.in_channels = in_channels
        
        # Placeholder for backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Point cloud tensor of shape (B, N, C) where B=batch, N=num_points, C=channels
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Extract features per point
        features = self.backbone(x)  # (B, N, 256)
        
        # Global max pooling
        global_features = torch.max(features, dim=1)[0]  # (B, 256)
        
        # Classification
        logits = self.classifier(global_features)  # (B, num_classes)
        
        return logits


class PointMLPSegmentation(nn.Module):
    """PointMLP model for point cloud segmentation"""
    
    def __init__(self, num_classes=50, num_points=2048, in_channels=3):
        """Initialize PointMLP segmentation model
        
        Args:
            num_classes: Number of output classes for segmentation
            num_points: Number of points in point cloud
            in_channels: Number of input channels
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        
        # Backbone encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        
        # Segmentation head with point-wise predictions
        self.segmentation_head = nn.Sequential(
            nn.Linear(256 + 256, 128),  # Concatenate local + global features
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass for segmentation
        
        Args:
            x: Point cloud tensor of shape (B, N, C)
            
        Returns:
            Segmentation logits of shape (B, N, num_classes)
        """
        # Extract local features per point
        local_features = self.encoder(x)  # (B, N, 256)
        
        # Extract global features
        global_features = torch.max(local_features, dim=1)[0]  # (B, 256)
        global_features = global_features.unsqueeze(1).expand_as(local_features)  # (B, N, 256)
        
        # Concatenate and predict per-point labels
        combined = torch.cat([local_features, global_features], dim=-1)  # (B, N, 512)
        logits = self.segmentation_head(combined)  # (B, N, num_classes)
        
        return logits
