"""Tests for DEM Fusion module"""

import torch
import pytest
from src.dem_fusion import PIMSRModel


class TestPIMSRModel:
    """Test PIMSR DEM fusion model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = PIMSRModel(upscale_factor=4)
        assert model.upscale_factor == 4
    
    def test_forward_pass_4x(self):
        """Test 4x upscaling"""
        model = PIMSRModel(upscale_factor=4)
        dem_lr = torch.randn(1, 1, 64, 64)
        sar_hr = torch.randn(1, 1, 256, 256)
        output = model(dem_lr, sar_hr)
        assert output.shape == (1, 1, 256, 256)
    
    def test_forward_pass_2x(self):
        """Test 2x upscaling"""
        model = PIMSRModel(upscale_factor=2)
        dem_lr = torch.randn(1, 1, 128, 128)
        sar_hr = torch.randn(1, 1, 256, 256)
        output = model(dem_lr, sar_hr)
        assert output.shape == (1, 1, 256, 256)


if __name__ == "__main__":
    pytest.main([__file__])
