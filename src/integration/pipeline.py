import torch
import numpy as np
from ..pointmlp import PointMLP
from ..dem_fusion import PIMSRModel
from ..gravity_correction import AdaptiveTerrainCorrection


class IntegrationPipeline:
    """Unified integration pipeline for all three components"""
    
    def __init__(self, device='cpu'):
        """Initialize integration pipeline
        
        Args:
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Initialize models
        self.pointmlp = PointMLP(num_classes=40).to(device)
        self.dem_fusion = PIMSRModel().to(device)
        self.gravity_correction = AdaptiveTerrainCorrection().to(device)
    
    def process_point_cloud(self, point_cloud):
        """Process point cloud for classification
        
        Args:
            point_cloud: Point cloud tensor (B, N, 3)
            
        Returns:
            Classification logits
        """
        point_cloud = point_cloud.to(self.device)
        with torch.no_grad():
            logits = self.pointmlp(point_cloud)
        return logits
    
    def fuse_dem_sar(self, dem_lr, sar_hr):
        """Fuse low-res DEM with high-res SAR
        
        Args:
            dem_lr: Low-resolution DEM
            sar_hr: High-resolution SAR
            
        Returns:
            Super-resolved DEM
        """
        dem_lr = dem_lr.to(self.device)
        sar_hr = sar_hr.to(self.device)
        with torch.no_grad():
            dem_sr = self.dem_fusion(dem_lr, sar_hr)
        return dem_sr
    
    def correct_gravity_terrain(self, dem, gravity_data=None):
        """Apply adaptive terrain correction to gravity data
        
        Args:
            dem: Digital Elevation Model
            gravity_data: Optional gravity anomaly data
            
        Returns:
            Corrected gravity data and correction values
        """
        dem = dem.to(self.device)
        if gravity_data is not None:
            gravity_data = gravity_data.to(self.device)
        
        with torch.no_grad():
            results = self.gravity_correction(dem, gravity_data)
        return results
    
    def full_pipeline(self, point_cloud, dem_lr, sar_hr, gravity_data=None):
        """Run complete pipeline
        
        Args:
            point_cloud: Point cloud data
            dem_lr: Low-resolution DEM
            sar_hr: High-resolution SAR
            gravity_data: Optional gravity anomaly
            
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Point cloud classification
        results['point_classification'] = self.process_point_cloud(point_cloud)
        
        # DEM-SAR fusion
        results['dem_sr'] = self.fuse_dem_sar(dem_lr, sar_hr)
        
        # Gravity terrain correction
        gravity_results = self.correct_gravity_terrain(results['dem_sr'].squeeze(0).unsqueeze(1), gravity_data)
        if gravity_data is not None:
            results['corrected_gravity'] = gravity_results[0]
            results['terrain_correction'] = gravity_results[1]
            results['complexity'] = gravity_results[2]
        else:
            results['terrain_correction'] = gravity_results[0]
            results['complexity'] = gravity_results[1]
        
        return results
    
    def save_checkpoint(self, path):
        """Save all models
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'pointmlp': self.pointmlp.state_dict(),
            'dem_fusion': self.dem_fusion.state_dict(),
            'gravity_correction': self.gravity_correction.state_dict(),
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load all models
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.pointmlp.load_state_dict(checkpoint['pointmlp'])
        self.dem_fusion.load_state_dict(checkpoint['dem_fusion'])
        self.gravity_correction.load_state_dict(checkpoint['gravity_correction'])
