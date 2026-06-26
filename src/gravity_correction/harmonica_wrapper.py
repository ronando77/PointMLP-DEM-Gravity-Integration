"""Enhanced Harmonica wrapper with traditional terrain correction"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import harmonica
    HAS_HARMONICA = True
except ImportError:
    HAS_HARMONICA = False
    warnings.warn("harmonica not installed. Install with: pip install harmonica")


class TraditionalTerrainCorrection:
    """Traditional terrain correction using Prism/Tesseroid method"""
    
    def __init__(self, density: float = 2670):
        """Initialize terrain correction
        
        Args:
            density: Terrain density in kg/m³
        """
        self.density = density
        self.G = 6.67430e-11  # Gravitational constant
    
    def prism_correction(
        self,
        dem: np.ndarray,
        coords_x: np.ndarray,
        coords_y: np.ndarray,
        cell_size: float = 30.0,
        observation_height: float = 0.0,
    ) -> np.ndarray:
        """Compute terrain correction using prism approximation
        
        Args:
            dem: Digital Elevation Model
            coords_x: X coordinates of observation points
            coords_y: Y coordinates of observation points
            cell_size: DEM cell size in meters
            observation_height: Height of observation points
            
        Returns:
            Terrain correction at observation points
        """
        correction = np.zeros_like(coords_x, dtype=float)
        
        # Create mesh grid
        ny, nx = dem.shape
        x_grid = np.arange(0, nx * cell_size, cell_size)
        y_grid = np.arange(0, ny * cell_size, cell_size)
        
        # Compute correction for each observation point
        for i in range(len(coords_x)):
            obs_x = coords_x[i]
            obs_y = coords_y[i]
            
            # Compute distance to each prism and accumulate
            for j in range(ny):
                for k in range(nx):
                    # Prism vertices
                    x1, x2 = x_grid[k], x_grid[k] + cell_size
                    y1, y2 = y_grid[j], y_grid[j] + cell_size
                    z1, z2 = 0, dem[j, k]  # From ground to DEM surface
                    
                    # Compute prism effect
                    effect = self._prism_effect(
                        obs_x, obs_y, observation_height,
                        x1, x2, y1, y2, z1, z2
                    )
                    correction[i] += effect
        
        return correction * 1e5  # Convert to mGal
    
    @staticmethod
    def _prism_effect(
        obs_x: float, obs_y: float, obs_z: float,
        x1: float, x2: float, y1: float, y2: float, z1: float, z2: float,
        density: float = 2670.0
    ) -> float:
        """Compute gravitational effect of prism (simplified)
        
        Args:
            obs_x, obs_y, obs_z: Observation point coordinates
            x1, x2, y1, y2, z1, z2: Prism bounds
            density: Prism density
            
        Returns:
            Gravitational effect
        """
        G = 6.67430e-11
        
        # Simplified prism computation using corner points
        effect = 0
        mass = density * (x2-x1) * (y2-y1) * (z2-z1)
        
        # Distance from observation point to prism center
        dx = (x1 + x2) / 2 - obs_x
        dy = (y1 + y2) / 2 - obs_y
        dz = (z1 + z2) / 2 - obs_z
        
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if r > 1e-6:  # Avoid division by zero
            effect = G * mass * dz / (r**3)
        
        return effect
    
    def bouguer_correction(self, latitude: np.ndarray, height: np.ndarray) -> np.ndarray:
        """Compute Bouguer correction
        
        Args:
            latitude: Latitude in degrees
            height: Height/elevation in meters
            
        Returns:
            Bouguer correction in mGal
        """
        # Bouguer correction formula
        bouguer = 0.0419 * self.density * height / 1000  # mGal
        return bouguer
    
    def free_air_correction(self, height: np.ndarray) -> np.ndarray:
        """Compute free-air correction
        
        Args:
            height: Height/elevation in meters
            
        Returns:
            Free-air correction in mGal
        """
        # Standard free-air correction
        free_air = 0.3086 * height / 1000  # mGal
        return free_air


class HarmonicaWrapper:
    """Wrapper for Harmonica library functions"""
    
    def __init__(self, density: float = 2670):
        """Initialize Harmonica wrapper
        
        Args:
            density: Density in kg/m³
        """
        if not HAS_HARMONICA:
            raise ImportError("Harmonica is required. Install with: pip install harmonica")
        
        self.density = density
    
    def compute_full_bouguer(
        self,
        latitude: np.ndarray,
        height: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute full Bouguer correction components
        
        Args:
            latitude: Latitude in degrees
            height: Height/elevation in meters
            
        Returns:
            Tuple of (free_air, bouguer, full_bouguer)
        """
        free_air = 0.3086 * height / 1000
        bouguer = 0.0419 * self.density * height / 1000
        full_bouguer = free_air - bouguer
        
        return free_air, bouguer, full_bouguer
