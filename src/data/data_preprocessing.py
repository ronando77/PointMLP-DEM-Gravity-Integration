"""Data preprocessing and augmentation utilities"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

try:
    import laspy
except ImportError:
    warnings.warn("laspy not installed. Install with: pip install laspy")


class PointCloudPreprocessor:
    """Comprehensive point cloud preprocessing"""
    
    def __init__(self, verbose: bool = False):
        """Initialize preprocessor
        
        Args:
            verbose: Print preprocessing information
        """
        self.verbose = verbose
    
    def filter_outliers(self, points: np.ndarray, method: str = 'statistical',
                       num_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        """Filter outliers using statistical method
        
        Args:
            points: Point cloud array (N, 3)
            method: 'statistical' or 'radius'
            num_neighbors: Number of neighbors to consider
            std_ratio: Standard deviation ratio threshold
            
        Returns:
            Filtered point cloud
        """
        if method == 'statistical':
            # Statistical outlier removal
            mean_dist = np.zeros(len(points))
            for i in range(len(points)):
                distances = np.linalg.norm(points - points[i], axis=1)
                distances = np.sort(distances)[1:num_neighbors+1]
                mean_dist[i] = np.mean(distances)
            
            global_mean = np.mean(mean_dist)
            global_std = np.std(mean_dist)
            threshold = global_mean + std_ratio * global_std
            
            mask = mean_dist < threshold
            if self.verbose:
                print(f"Removed {len(points) - mask.sum()} outliers")
            
            return points[mask]
        
        elif method == 'radius':
            # Radius outlier removal
            radius = std_ratio  # Use std_ratio as radius
            mask = np.zeros(len(points), dtype=bool)
            
            for i in range(len(points)):
                distances = np.linalg.norm(points - points[i], axis=1)
                neighbors = np.sum(distances < radius)
                mask[i] = neighbors >= num_neighbors
            
            if self.verbose:
                print(f"Removed {len(points) - mask.sum()} outliers")
            
            return points[mask]
    
    def downsample_voxel(
        self,
        points: np.ndarray,
        voxel_size: float = 0.1,
    ) -> np.ndarray:
        """Voxel-based downsampling
        
        Args:
            points: Point cloud array
            voxel_size: Size of voxel
            
        Returns:
            Downsampled point cloud
        """
        if points.shape[0] == 0:
            return points
        
        # Create voxel grid
        coords = np.floor(points / voxel_size).astype(int)
        unique_coords, indices = np.unique(coords, axis=0, return_index=True)
        
        downsampled = points[indices]
        
        if self.verbose:
            print(f"Downsampled from {len(points)} to {len(downsampled)} points")
        
        return downsampled
    
    def normalize(
        self,
        points: np.ndarray,
        method: str = 'center_scale',
    ) -> Tuple[np.ndarray, dict]:
        """Normalize point cloud
        
        Args:
            points: Point cloud array
            method: 'center_scale', 'zscore', or 'minmax'
            
        Returns:
            Tuple of (normalized_points, transform_info)
        """
        info = {}
        
        if method == 'center_scale':
            centroid = np.mean(points, axis=0)
            points = points - centroid
            furthest_distance = np.max(np.linalg.norm(points, axis=1))
            points = points / (furthest_distance + 1e-8)
            
            info = {
                'centroid': centroid,
                'scale': furthest_distance,
                'method': method,
            }
        
        elif method == 'zscore':
            mean = np.mean(points, axis=0)
            std = np.std(points, axis=0)
            points = (points - mean) / (std + 1e-8)
            
            info = {
                'mean': mean,
                'std': std,
                'method': method,
            }
        
        elif method == 'minmax':
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            points = (points - min_vals) / (max_vals - min_vals + 1e-8)
            
            info = {
                'min': min_vals,
                'max': max_vals,
                'method': method,
            }
        
        if self.verbose:
            print(f"Normalized using {method}")
        
        return points, info
    
    def process_las_file(
        self,
        file_path: str,
        num_points: int = 1024,
        remove_outliers: bool = True,
        downsample: bool = False,
        voxel_size: float = 0.1,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Complete preprocessing pipeline for LAS file
        
        Args:
            file_path: Path to LAS file
            num_points: Target number of points
            remove_outliers: Whether to remove outliers
            downsample: Whether to downsample
            voxel_size: Voxel size for downsampling
            normalize: Whether to normalize
            
        Returns:
            Tuple of (processed_points, metadata)
        """
        metadata = {'file': file_path}
        
        # Load LAS file
        with laspy.open(file_path) as las_file:
            las = las_file.read()
            points = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
        
        metadata['original_count'] = len(points)
        
        # Remove outliers
        if remove_outliers and len(points) > 100:
            points = self.filter_outliers(points, method='statistical')
            metadata['after_outlier_removal'] = len(points)
        
        # Downsample
        if downsample and len(points) > num_points:
            points = self.downsample_voxel(points, voxel_size=voxel_size)
            metadata['after_downsample'] = len(points)
        
        # Normalize
        if normalize:
            points, norm_info = self.normalize(points, method='center_scale')
            metadata['normalization'] = norm_info
        
        # Sample to exact number of points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            indices = np.random.choice(len(points), num_points, replace=True)
            points = points[indices]
        
        metadata['final_count'] = len(points)
        
        return points, metadata
    
    def batch_process(
        self,
        directory: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[np.ndarray, dict]]:
        """Batch process LAS files
        
        Args:
            directory: Input directory
            output_dir: Optional output directory to save processed files
            **kwargs: Arguments for process_las_file
            
        Returns:
            List of (processed_points, metadata) tuples
        """
        directory = Path(directory)
        las_files = list(directory.glob('*.las')) + list(directory.glob('*.laz'))
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        results = []
        for las_file in las_files:
            try:
                points, metadata = self.process_las_file(str(las_file), **kwargs)
                results.append((points, metadata))
                
                # Save if output_dir provided
                if output_dir:
                    output_path = output_dir / f"{las_file.stem}_processed.npy"
                    np.save(output_path, points)
                    
                    meta_path = output_dir / f"{las_file.stem}_metadata.txt"
                    with open(meta_path, 'w') as f:
                        for key, value in metadata.items():
                            f.write(f"{key}: {value}\n")
                
                if self.verbose:
                    print(f"✓ Processed {las_file.name}")
            
            except Exception as e:
                print(f"✗ Error processing {las_file.name}: {e}")
        
        return results
