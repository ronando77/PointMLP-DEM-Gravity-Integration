"""Example: DEM-SAR fusion with PIMSR"""

import torch
from src.dem_fusion import PIMSRModel


def main():
    """Run DEM fusion example"""
    
    # Initialize model
    model = PIMSRModel(
        in_channels_dem=1,
        in_channels_sar=1,
        out_channels=1,
        upscale_factor=4
    )
    model.eval()
    
    # Create dummy data
    # LR DEM: (Batch=1, Channels=1, Height=64, Width=64)
    dem_lr = torch.randn(1, 1, 64, 64)
    
    # HR SAR: (Batch=1, Channels=1, Height=256, Width=256) - 4x upscale
    sar_hr = torch.randn(1, 1, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        dem_sr = model(dem_lr, sar_hr)
    
    print(f"LR DEM shape: {dem_lr.shape}")
    print(f"HR SAR shape: {sar_hr.shape}")
    print(f"SR DEM shape: {dem_sr.shape}")
    print(f"DEM value range: [{dem_sr.min():.4f}, {dem_sr.max():.4f}]")


if __name__ == "__main__":
    main()
