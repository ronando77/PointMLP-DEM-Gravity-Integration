"""Example: Adaptive gravity terrain correction"""

import torch
from src.gravity_correction import AdaptiveTerrainCorrection


def main():
    """Run gravity correction example"""
    
    # Initialize model
    model = AdaptiveTerrainCorrection(dem_resolution=30)
    model.eval()
    
    # Create dummy data
    # DEM: (Batch=1, Channels=1, Height=256, Width=256)
    dem = torch.randn(1, 1, 256, 256) * 1000 + 500  # Elevation 0-2000m
    
    # Gravity anomaly (optional)
    gravity_data = torch.randn(1, 1, 256, 256) * 50  # Anomaly in mGal
    
    # Forward pass with correction
    with torch.no_grad():
        corrected_gravity, corrections, complexity = model(dem, gravity_data)
    
    print(f"DEM shape: {dem.shape}")
    print(f"Gravity anomaly shape: {gravity_data.shape}")
    print(f"Corrected gravity shape: {corrected_gravity.shape}")
    print(f"Terrain correction shape: {corrections.shape}")
    print(f"Terrain complexity score: {complexity.item():.4f}")
    print(f"Correction range: [{corrections.min():.4f}, {corrections.max():.4f}] mGal")


if __name__ == "__main__":
    main()
