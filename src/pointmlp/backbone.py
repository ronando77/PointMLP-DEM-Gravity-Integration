import torch
import torch.nn as nn


class PointMLPBackbone(nn.Module):
    """PointMLP backbone for feature extraction"""
    
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512]):
        """Initialize PointMLP backbone
        
        Args:
            in_channels: Number of input channels
            hidden_dims: List of hidden dimensions for each layer
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        
        # Build MLP layers
        layers = []
        prev_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """Extract features from point cloud
        
        Args:
            x: Point cloud tensor of shape (B, N, C)
            
        Returns:
            Features of shape (B, N, hidden_dims[-1])
        """
        # Reshape for MLP
        B, N, C = x.shape
        x = x.reshape(B * N, C)
        
        # Extract features
        features = self.mlp(x)
        
        # Reshape back
        features = features.reshape(B, N, -1)
        
        return features
