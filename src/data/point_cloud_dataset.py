"""PyTorch Dataset for point cloud data"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, Tuple
import warnings

try:
    from .las_loader import LASLoader
    HAS_LAS_LOADER = True
except ImportError:
    HAS_LAS_LOADER = False


class PointCloudDataset(Dataset):
    """PyTorch Dataset for point cloud files"""
    
    def __init__(
        self,
        data_dir: str,
        num_points: int = 1024,
        extension: str = '.las',
        normalize: bool = True,
        transform: Optional[Callable] = None,
        labels: Optional[dict] = None,
        split: str = 'train',
        split_ratio: float = 0.8,
    ):
        """Initialize dataset
        
        Args:
            data_dir: Directory containing point cloud files
            num_points: Number of points per cloud
            extension: File extension (.las, .ply, .npy, etc.)
            normalize: Whether to normalize point clouds
            transform: Optional data augmentation transforms
            labels: Optional dict mapping filename to label
            split: 'train', 'val', or 'test'
            split_ratio: Train/val split ratio
        """
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.extension = extension
        self.normalize = normalize
        self.transform = transform
        self.labels = labels or {}
        
        # Find all files
        self.file_list = sorted(self.data_dir.glob(f'*{extension}'))
        
        if not self.file_list:
            raise FileNotFoundError(f"No files with extension {extension} found in {data_dir}")
        
        # Split data
        split_idx = int(len(self.file_list) * split_ratio)
        if split == 'train':
            self.file_list = self.file_list[:split_idx]
        elif split == 'val':
            self.file_list = self.file_list[split_idx:]
        elif split == 'test':
            pass  # Use all files
        
        # Initialize loader
        if extension in ['.las', '.laz']:
            if not HAS_LAS_LOADER:
                raise ImportError("laspy required for LAS files. Install: pip install laspy")
            self.loader = LASLoader(normalize=False, sample_points=None)
            self.load_func = self.loader.load
        else:
            self.load_func = self._load_default
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.file_list[idx]
        
        # Load point cloud
        points = self.load_func(str(file_path))
        
        # Ensure correct number of points
        points = self._pad_or_truncate(points)
        
        # Use only XYZ (first 3 channels)
        points = points[:, :3].astype(np.float32)
        
        # Normalize
        if self.normalize:
            points = self._normalize_points(points)
        
        # Apply transforms
        if self.transform:
            points = self.transform(points)
        
        # Convert to tensor
        points = torch.from_numpy(points).float()
        
        # Get label
        label = self.labels.get(file_path.stem, 0)
        label = torch.tensor(label, dtype=torch.long)
        
        return points, label
    
    def _pad_or_truncate(self, points: np.ndarray) -> np.ndarray:
        """Pad or truncate to num_points"""
        n = len(points)
        if n > self.num_points:
            indices = np.random.choice(n, self.num_points, replace=False)
            return points[indices]
        elif n < self.num_points:
            indices = np.random.choice(n, self.num_points, replace=True)
            return points[indices]
        else:
            return points
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize point cloud"""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / (furthest_distance + 1e-8)
        return points
    
    def _load_default(self, file_path: str) -> np.ndarray:
        """Default loader for .npy files"""
        if file_path.endswith('.npy'):
            return np.load(file_path)
        else:
            raise NotImplementedError(f"Loader not implemented for {file_path}")


class PointCloudAugmentation:
    """Data augmentation for point clouds"""
    
    def __init__(
        self,
        rotation: bool = True,
        scaling: bool = True,
        jitter: bool = True,
        dropout: bool = False,
        dropout_rate: float = 0.1,
    ):
        """Initialize augmentation
        
        Args:
            rotation: Random rotation
            scaling: Random scaling
            jitter: Add noise
            dropout: Random point dropout
            dropout_rate: Dropout probability
        """
        self.rotation = rotation
        self.scaling = scaling
        self.jitter = jitter
        self.dropout = dropout
        self.dropout_rate = dropout_rate
    
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Apply augmentation
        
        Args:
            points: Point cloud array (N, 3)
            
        Returns:
            Augmented point cloud
        """
        if self.rotation:
            points = self._random_rotation(points)
        
        if self.scaling:
            points = self._random_scaling(points)
        
        if self.jitter:
            points = self._add_jitter(points)
        
        if self.dropout:
            points = self._random_dropout(points)
        
        return points
    
    @staticmethod
    def _random_rotation(points: np.ndarray) -> np.ndarray:
        """Random rotation"""
        angles = np.random.rand(3) * 2 * np.pi
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        R = Rx @ Ry @ Rz
        return points @ R.T
    
    @staticmethod
    def _random_scaling(points: np.ndarray, scale_range: float = 0.2) -> np.ndarray:
        """Random scaling"""
        scale = 1 + np.random.randn() * scale_range
        return points * scale
    
    @staticmethod
    def _add_jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.randn(*points.shape) * sigma
        noise = np.clip(noise, -clip, clip)
        return points + noise
    
    def _random_dropout(self, points: np.ndarray) -> np.ndarray:
        """Random point dropout"""
        mask = np.random.rand(len(points)) > self.dropout_rate
        if mask.sum() == 0:
            return points
        return points[mask]
