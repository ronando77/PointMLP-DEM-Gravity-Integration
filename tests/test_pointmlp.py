"""Tests for PointMLP module"""

import torch
import pytest
from src.pointmlp import PointMLP, PointMLPSegmentation


class TestPointMLP:
    """Test PointMLP classification model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = PointMLP(num_classes=40)
        assert model.num_classes == 40
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = PointMLP(num_classes=40, num_points=1024)
        point_cloud = torch.randn(2, 1024, 3)
        output = model(point_cloud)
        assert output.shape == (2, 40)
    
    def test_segmentation_model(self):
        """Test segmentation model"""
        model = PointMLPSegmentation(num_classes=50, num_points=2048)
        point_cloud = torch.randn(2, 2048, 3)
        output = model(point_cloud)
        assert output.shape == (2, 2048, 50)


if __name__ == "__main__":
    pytest.main([__file__])
