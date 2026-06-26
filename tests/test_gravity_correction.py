"""Tests for Gravity Correction module"""

import torch
import pytest
from src.gravity_correction import AdaptiveTerrainCorrection


class TestAdaptiveTerrainCorrection:
    """Test adaptive terrain correction model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = AdaptiveTerrainCorrection(dem_resolution=30)
        assert model.dem_resolution == 30
    
    def test_forward_pass_without_gravity(self):
        """Test forward pass without gravity data"""
        model = AdaptiveTerrainCorrection()
        dem = torch.randn(1, 1, 256, 256)
        corrections, complexity = model(dem)
        assert corrections.shape == (1, 1, 256, 256)
        assert 0 <= complexity.item() <= 1
    
    def test_forward_pass_with_gravity(self):
        """Test forward pass with gravity data"""
        model = AdaptiveTerrainCorrection()
        dem = torch.randn(1, 1, 256, 256)
        gravity = torch.randn(1, 1, 256, 256)
        corrected, corrections, complexity = model(dem, gravity)
        assert corrected.shape == (1, 1, 256, 256)
        assert corrections.shape == (1, 1, 256, 256)
        assert 0 <= complexity.item() <= 1
    
    def test_complexity_score(self):
        """Test complexity score computation"""
        model = AdaptiveTerrainCorrection()
        dem = torch.randn(1, 1, 256, 256)
        score = model.get_complexity_score(dem)
        assert isinstance(score, float)
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__])
