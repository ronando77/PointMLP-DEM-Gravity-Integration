"""Example: Point cloud classification with PointMLP"""

import torch
from src.pointmlp import PointMLP, utils


def main():
    """Run point cloud classification example"""
    
    # Initialize model
    model = PointMLP(num_classes=40, num_points=1024)
    model.eval()
    
    # Create dummy point cloud data (Batch=2, Points=1024, Channels=3)
    point_cloud = torch.randn(2, 1024, 3)
    
    # Normalize
    point_cloud = utils.normalize_point_cloud(point_cloud)
    
    # Forward pass
    with torch.no_grad():
        logits = model(point_cloud)
    
    # Get predictions
    predictions = torch.argmax(logits, dim=1)
    
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Classification logits shape: {logits.shape}")
    print(f"Predictions: {predictions}")
    print(f"Confidence scores: {torch.softmax(logits, dim=1).max(dim=1)[0]}")


if __name__ == "__main__":
    main()
