"""Configuration management"""

import yaml
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """Configuration class for the integration pipeline"""
    
    # Point Cloud Processing
    pointcloud_num_points: int = 1024
    pointcloud_num_classes: int = 40
    
    # DEM Fusion
    dem_upscale_factor: int = 4
    dem_patch_size: int = 64
    
    # Gravity Correction
    gravity_dem_resolution: int = 30
    gravity_depth_layers: int = 5
    
    # Training
    batch_size: int = 8
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = 'cuda'
    
    # Paths
    data_dir: str = './data'
    output_dir: str = './output'
    checkpoint_dir: str = './checkpoints'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pointcloud_num_points': self.pointcloud_num_points,
            'pointcloud_num_classes': self.pointcloud_num_classes,
            'dem_upscale_factor': self.dem_upscale_factor,
            'dem_patch_size': self.dem_patch_size,
            'gravity_dem_resolution': self.gravity_dem_resolution,
            'gravity_depth_layers': self.gravity_depth_layers,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'device': self.device,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load config from YAML file
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            Config instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
