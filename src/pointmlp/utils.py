import torch
import numpy as np


def load_point_cloud(file_path, num_points=1024):
    """Load point cloud from file
    
    Args:
        file_path: Path to point cloud file (.ply, .pcd, .npy)
        num_points: Number of points to sample
        
    Returns:
        Point cloud tensor of shape (1, num_points, 3)
    """
    if file_path.endswith('.npy'):
        points = np.load(file_path)
    else:
        raise NotImplementedError(f"File format not supported: {file_path}")
    
    # Ensure correct number of points
    if points.shape[0] > num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[indices]
    elif points.shape[0] < num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=True)
        points = points[indices]
    
    return torch.from_numpy(points).unsqueeze(0).float()


def normalize_point_cloud(points, method='zscore'):
    """Normalize point cloud
    
    Args:
        points: Point cloud tensor
        method: Normalization method ('zscore' or 'minmax')
        
    Returns:
        Normalized point cloud
    """
    if method == 'zscore':
        centroid = torch.mean(points, dim=1, keepdim=True)
        points = points - centroid
        furthest_distance = torch.max(torch.norm(points, dim=2))
        points = points / furthest_distance
    elif method == 'minmax':
        min_vals = torch.min(points, dim=1, keepdim=True)[0]
        max_vals = torch.max(points, dim=1, keepdim=True)[0]
        points = (points - min_vals) / (max_vals - min_vals + 1e-8)
    
    return points
