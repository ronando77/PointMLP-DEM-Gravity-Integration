"""LAS file loader for point cloud data"""

import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Optional

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    warnings.warn("laspy not installed. Install with: pip install laspy")


class LASLoader:
    """Loader for LAS/LAZ point cloud files"""
    
    def __init__(self, normalize: bool = True, sample_points: Optional[int] = None):
        """Initialize LAS loader
        
        Args:
            normalize: Whether to normalize point cloud
            sample_points: Number of points to sample (None = use all)
        """
        if not HAS_LASPY:
            raise ImportError("laspy is required. Install with: pip install laspy")
        
        self.normalize = normalize
        self.sample_points = sample_points
    
    def load(self, file_path: str) -> np.ndarray:
        """Load LAS file
        
        Args:
            file_path: Path to LAS/LAZ file
            
        Returns:
            Point cloud array of shape (N, 3) or (N, num_features)
        """
        with laspy.open(file_path) as las_file:
            las = las_file.read()
            
            # Extract XYZ coordinates
            points = np.vstack((las.x, las.y, las.z)).transpose()
            
            # Optional: extract additional features
            features = [points]
            
            # Add intensity if available
            if hasattr(las, 'intensity'):
                features.append(las.intensity.reshape(-1, 1))
            
            # Add classification if available
            if hasattr(las, 'classification'):
                features.append(las.classification.reshape(-1, 1))
            
            # Add RGB if available
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                rgb = np.vstack((las.red, las.green, las.blue)).transpose()
                features.append(rgb)
            
            # Concatenate all features
            if len(features) > 1:
                point_cloud = np.hstack(features).astype(np.float32)
            else:
                point_cloud = points.astype(np.float32)
        
        # Sample points if specified
        if self.sample_points is not None and len(point_cloud) > self.sample_points:
            indices = np.random.choice(len(point_cloud), self.sample_points, replace=False)
            point_cloud = point_cloud[indices]
        
        # Normalize
        if self.normalize:
            point_cloud = self._normalize(point_cloud)
        
        return point_cloud
    
    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """Normalize point cloud
        
        Args:
            points: Point cloud array
            
        Returns:
            Normalized point cloud
        """
        # Normalize XYZ (first 3 channels)
        xyz = points[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        furthest_distance = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / (furthest_distance + 1e-8)
        
        # Keep other features as-is or normalize separately
        if points.shape[1] > 3:
            other_features = points[:, 3:]
            # Normalize other features to [0, 1]
            other_features = (other_features - other_features.min(axis=0)) / \
                           (other_features.max(axis=0) - other_features.min(axis=0) + 1e-8)
            point_cloud = np.hstack([xyz, other_features])
        else:
            point_cloud = xyz
        
        return point_cloud
    
    def batch_load(self, directory: str, extension: str = '.las') -> list:
        """Load multiple LAS files from directory
        
        Args:
            directory: Directory containing LAS files
            extension: File extension to search for
            
        Returns:
            List of point cloud arrays
        """
        directory = Path(directory)
        las_files = list(directory.glob(f'*{extension}'))
        
        point_clouds = []
        for las_file in las_files:
            try:
                pc = self.load(str(las_file))
                point_clouds.append(pc)
            except Exception as e:
                print(f"Error loading {las_file}: {e}")
        
        return point_clouds


def load_las_batch(directory: str, num_points: int = 1024, 
                   normalize: bool = True) -> Tuple[np.ndarray, list]:
    """Convenient function to load batch of LAS files
    
    Args:
        directory: Directory containing LAS files
        num_points: Number of points to sample from each file
        normalize: Whether to normalize point clouds
        
    Returns:
        Tuple of (point_clouds_array, file_names)
    """
    loader = LASLoader(normalize=normalize, sample_points=num_points)
    directory = Path(directory)
    las_files = sorted(directory.glob('*.las')) + sorted(directory.glob('*.laz'))
    
    point_clouds = []
    file_names = []
    
    for las_file in las_files:
        try:
            pc = loader.load(str(las_file))
            point_clouds.append(pc)
            file_names.append(las_file.name)
        except Exception as e:
            print(f"Error loading {las_file}: {e}")
    
    if point_clouds:
        # Pad or truncate to num_points
        point_clouds = [_pad_or_truncate(pc, num_points) for pc in point_clouds]
        point_clouds = np.array(point_clouds)
    
    return point_clouds, file_names


def _pad_or_truncate(points: np.ndarray, num_points: int) -> np.ndarray:
    """Pad or truncate point cloud to specific size
    
    Args:
        points: Point cloud array
        num_points: Target number of points
        
    Returns:
        Point cloud with num_points points
    """
    n = len(points)
    if n > num_points:
        indices = np.random.choice(n, num_points, replace=False)
        return points[indices]
    elif n < num_points:
        indices = np.random.choice(n, num_points, replace=True)
        return points[indices]
    else:
        return points
