import torch
import torch.nn as nn
import numpy as np


class AdaptiveTerrainCorrection(nn.Module):
    """Adaptive terrain correction using deep learning
    
    Adapts correction parameters based on terrain complexity
    """
    
    def __init__(self, dem_resolution=30):
        """Initialize adaptive terrain correction
        
        Args:
            dem_resolution: DEM resolution in meters
        """
        super().__init__()
        self.dem_resolution = dem_resolution
        
        # Terrain complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output complexity score [0, 1]
        )
        
        # Adaptive parameter predictor
        self.param_predictor = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, 3, 1, 1),  # 5 adaptive parameters
        )
        
        # Correction layer
        self.correction_layer = nn.Sequential(
            nn.Conv2d(5, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
    
    def forward(self, dem, gravity_data=None):
        """Apply adaptive terrain correction
        
        Args:
            dem: DEM tensor (B, 1, H, W)
            gravity_data: Optional gravity anomaly data (B, 1, H, W)
            
        Returns:
            Corrected gravity data and correction values
        """
        # Analyze terrain complexity
        complexity = self.complexity_analyzer(dem)
        
        # Predict adaptive parameters
        adaptive_params = self.param_predictor(dem)
        
        # Generate terrain correction
        corrections = self.correction_layer(adaptive_params)
        
        # If gravity data provided, apply correction
        if gravity_data is not None:
            corrected_gravity = gravity_data - corrections
            return corrected_gravity, corrections, complexity
        
        return corrections, complexity
    
    def get_complexity_score(self, dem):
        """Get terrain complexity score
        
        Args:
            dem: DEM tensor
            
        Returns:
            Complexity score (0-1)
        """
        complexity = self.complexity_analyzer(dem)
        return complexity.item()
