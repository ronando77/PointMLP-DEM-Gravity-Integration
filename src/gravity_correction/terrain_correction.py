import torch
import torch.nn as nn
import numpy as np


class TerrainCorrection(nn.Module):
    """Deep learning-based terrain correction for gravity data"""
    
    def __init__(self, dem_resolution=30, depth_layers=5):
        """Initialize terrain correction module
        
        Args:
            dem_resolution: Resolution of DEM in meters
            depth_layers: Number of depth layers for correction
        """
        super().__init__()
        self.dem_resolution = dem_resolution
        self.depth_layers = depth_layers
        
        # DEM encoder for feature extraction
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Terrain correction predictor
        self.correction_predictor = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, depth_layers, 3, 1, 1),
        )
    
    def forward(self, dem, gravity_fft=None):
        """Compute terrain correction
        
        Args:
            dem: DEM tensor of shape (B, 1, H, W)
            gravity_fft: Optional FFT of gravity anomaly
            
        Returns:
            Terrain correction tensor of shape (B, depth_layers, H, W)
        """
        # Extract DEM features
        dem_features = self.dem_encoder(dem)
        
        # Predict terrain corrections for each depth
        corrections = self.correction_predictor(dem_features)
        
        # Upsample back to original resolution
        corrections = torch.nn.functional.interpolate(
            corrections, size=dem.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        return corrections
