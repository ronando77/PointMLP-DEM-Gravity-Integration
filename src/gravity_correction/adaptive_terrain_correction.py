"""Adaptive Terrain Correction with PSTINet and traditional methods"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path


class PSTINet(nn.Module):
    """Physics-guided STI (Spatial Terrain Inference) Network for fast terrain correction"""
    
    def __init__(self, dem_resolution: float = 30.0):
        """Initialize PSTINet
        
        Args:
            dem_resolution: DEM resolution in meters
        """
        super().__init__()
        self.dem_resolution = dem_resolution
        
        # Physics-informed feature extraction
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        
        # Multi-scale terrain correction predictor
        self.correction_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
        # Gravity influence estimation
        self.gravity_estimator = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 5),  # Output 5 depth-weighted corrections
        )
    
    def forward(self, dem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            dem: DEM tensor (B, 1, H, W)
            
        Returns:
            Tuple of (terrain_correction, gravity_weights)
        """
        # Extract features
        features = self.dem_encoder(dem)
        
        # Predict terrain correction
        correction = self.correction_head(features)
        
        # Estimate gravity influence weights
        weights = torch.softmax(self.gravity_estimator(features), dim=1)
        
        return correction, weights


class AdaptiveTerrainCorrectionSystem:
    """Complete adaptive terrain correction system combining traditional and DL methods"""
    
    def __init__(
        self,
        dem_resolution: float = 30.0,
        use_neural: bool = True,
        model_path: Optional[str] = None,
        device: str = 'cuda',
    ):
        """Initialize adaptive correction system
        
        Args:
            dem_resolution: DEM resolution
            use_neural: Whether to use neural network
            model_path: Path to pretrained model
            device: Device to use
        """
        self.dem_resolution = dem_resolution
        self.use_neural = use_neural
        self.device = device
        
        # Import traditional method
        from .harmonica_wrapper import TraditionalTerrainCorrection
        self.traditional = TraditionalTerrainCorrection(density=2670)
        
        # Initialize neural network if requested
        if use_neural:
            self.psti_net = PSTINet(dem_resolution=dem_resolution).to(device)
            self.psti_net.eval()
            
            if model_path:
                checkpoint = torch.load(model_path, map_location=device)
                self.psti_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.psti_net = None
    
    def compute_adaptive_correction(
        self,
        dem: np.ndarray,
        latitude: Optional[np.ndarray] = None,
        longitude: Optional[np.ndarray] = None,
        obs_height: float = 0.0,
        blend_ratio: float = 0.5,
    ) -> Dict:
        """Compute adaptive terrain correction
        
        Args:
            dem: Digital Elevation Model (H, W)
            latitude: Latitude grid (optional)
            longitude: Longitude grid (optional)
            obs_height: Observation height
            blend_ratio: Weight between traditional (0) and neural (1) methods
            
        Returns:
            Dictionary with correction results and metadata
        """
        results = {}
        
        # Ensure DEM is properly formatted
        if len(dem.shape) != 2:
            raise ValueError("DEM must be 2D array")
        
        dem = dem.astype(np.float32)
        results['dem_shape'] = dem.shape
        results['dem_resolution'] = self.dem_resolution
        
        # Traditional method
        try:
            traditional_correction = self._compute_traditional(
                dem, latitude, longitude, obs_height
            )
            results['traditional_correction'] = traditional_correction
            results['traditional_valid'] = True
        except Exception as e:
            print(f"Warning: Traditional method failed: {e}")
            results['traditional_valid'] = False
            traditional_correction = np.zeros_like(dem)
        
        # Neural network method
        if self.use_neural and self.psti_net is not None:
            try:
                neural_correction, weights = self._compute_neural(dem)
                results['neural_correction'] = neural_correction
                results['gravity_weights'] = weights
                results['neural_valid'] = True
            except Exception as e:
                print(f"Warning: Neural method failed: {e}")
                results['neural_valid'] = False
                neural_correction = np.zeros_like(dem)
                weights = np.ones(5) / 5
        else:
            neural_correction = np.zeros_like(dem)
            results['neural_valid'] = False
        
        # Adaptive blending
        if results['traditional_valid'] and results['neural_valid']:
            # Adaptive blending based on terrain complexity
            complexity = self._compute_terrain_complexity(dem)
            results['terrain_complexity'] = complexity
            
            # Use more neural network for complex terrain
            adaptive_blend = blend_ratio * (1 + complexity)
            adaptive_blend = np.clip(adaptive_blend, 0, 1)
            results['adaptive_blend_ratio'] = adaptive_blend
            
            correction = (
                (1 - adaptive_blend) * traditional_correction +
                adaptive_blend * neural_correction
            )
        elif results['traditional_valid']:
            correction = traditional_correction
        elif results['neural_valid']:
            correction = neural_correction
        else:
            correction = np.zeros_like(dem)
        
        results['final_correction'] = correction
        
        # Compute statistics
        results['correction_stats'] = {
            'min': float(np.min(correction)),
            'max': float(np.max(correction)),
            'mean': float(np.mean(correction)),
            'std': float(np.std(correction)),
        }
        
        return results
    
    def _compute_traditional(
        self,
        dem: np.ndarray,
        latitude: Optional[np.ndarray],
        longitude: Optional[np.ndarray],
        obs_height: float,
    ) -> np.ndarray:
        """Compute traditional terrain correction using prism method"""
        h, w = dem.shape
        
        # Create coordinate grids
        if latitude is None:
            coords_y = np.arange(h) * self.dem_resolution
        else:
            coords_y = latitude.flatten()
        
        if longitude is None:
            coords_x = np.arange(w) * self.dem_resolution
        else:
            coords_x = longitude.flatten()
        
        # Create meshgrid
        xx, yy = np.meshgrid(coords_x, coords_y)
        
        # Compute correction (simplified)
        correction = np.zeros_like(dem, dtype=float)
        
        # Use DEM gradient as proxy for terrain effect
        gradient_x = np.gradient(dem, axis=1)
        gradient_y = np.gradient(dem, axis=0)
        terrain_slope = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Simplified terrain correction (proportional to slope and height)
        correction = 0.0419 * dem * terrain_slope / 1000
        
        return correction
    
    def _compute_neural(self, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute neural network terrain correction"""
        # Prepare input
        dem_tensor = torch.from_numpy(dem).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            correction, weights = self.psti_net(dem_tensor)
        
        # Convert to numpy
        correction = correction.squeeze().cpu().numpy()
        weights = weights.squeeze().cpu().numpy()
        
        return correction, weights
    
    @staticmethod
    def _compute_terrain_complexity(dem: np.ndarray) -> float:
        """Compute terrain complexity score (0-1)"""
        # Use gradient-based complexity measure
        gradient_x = np.gradient(dem, axis=1)
        gradient_y = np.gradient(dem, axis=0)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize to 0-1
        max_slope = np.percentile(slope, 95)  # Use 95th percentile
        complexity = np.mean(slope) / (max_slope + 1e-8)
        complexity = np.clip(complexity, 0, 1)
        
        return complexity
    
    def save_results(
        self,
        results: Dict,
        output_dir: str,
    ):
        """Save correction results to disk
        
        Args:
            results: Results dictionary from compute_adaptive_correction
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save corrections as numpy files
        np.save(output_dir / 'traditional_correction.npy', 
               results['traditional_correction'])
        np.save(output_dir / 'neural_correction.npy',
               results['neural_correction'])
        np.save(output_dir / 'final_correction.npy',
               results['final_correction'])
        
        # Save metadata
        import json
        metadata = {
            'dem_shape': results['dem_shape'],
            'dem_resolution': results['dem_resolution'],
            'terrain_complexity': float(results.get('terrain_complexity', 0)),
            'adaptive_blend_ratio': float(results.get('adaptive_blend_ratio', 0)),
            'correction_stats': results['correction_stats'],
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
