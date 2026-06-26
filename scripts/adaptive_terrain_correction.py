"""Adaptive terrain correction script combining traditional and neural methods"""

import argparse
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gravity_correction import AdaptiveTerrainCorrectionSystem

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Adaptive terrain correction for gravity data'
    )
    
    parser.add_argument('--dem', type=str, required=True,
                       help='Path to DEM file (GeoTIFF or NPY)')
    parser.add_argument('--dem-resolution', type=float, default=30.0,
                       help='DEM resolution in meters')
    parser.add_argument('--output-dir', type=str, default='./gravity_results',
                       help='Output directory')
    parser.add_argument('--use-neural', action='store_true',
                       help='Use neural network method')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pretrained neural model')
    parser.add_argument('--blend-ratio', type=float, default=0.5,
                       help='Blending ratio (0=traditional, 1=neural)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    return parser.parse_args()


def load_dem(dem_path: str) -> np.ndarray:
    """Load DEM from file
    
    Args:
        dem_path: Path to DEM file
        
    Returns:
        DEM array
    """
    dem_path = Path(dem_path)
    
    if dem_path.suffix == '.npy':
        dem = np.load(dem_path)
    elif dem_path.suffix in ['.tif', '.tiff']:
        if not HAS_RASTERIO:
            raise ImportError("rasterio required for GeoTIFF. Install: pip install rasterio")
        
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
    else:
        raise ValueError(f"Unsupported DEM format: {dem_path.suffix}")
    
    return dem.astype(np.float32)


def main():
    """Main function"""
    args = parse_args()
    
    print("="*70)
    print("Adaptive Terrain Correction for Gravity Data")
    print("="*70)
    print(f"DEM file: {args.dem}")
    print(f"DEM resolution: {args.dem_resolution} m")
    print(f"Use neural method: {args.use_neural}")
    print(f"Blend ratio: {args.blend_ratio}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load DEM
    print("\nLoading DEM...")
    dem = load_dem(args.dem)
    print(f"DEM shape: {dem.shape}")
    print(f"DEM range: [{dem.min():.2f}, {dem.max():.2f}]")
    
    # Create correction system
    print("\nInitializing adaptive terrain correction system...")
    system = AdaptiveTerrainCorrectionSystem(
        dem_resolution=args.dem_resolution,
        use_neural=args.use_neural,
        model_path=args.model_path,
        device=args.device,
    )
    
    # Compute adaptive correction
    print("\nComputing adaptive terrain correction...")
    results = system.compute_adaptive_correction(
        dem=dem,
        blend_ratio=args.blend_ratio,
    )
    
    # Print results
    print("\n" + "="*70)
    print("Terrain Correction Results")
    print("="*70)
    
    if results['traditional_valid']:
        print("\n✓ Traditional method: VALID")
        print(f"  Range: [{results['traditional_correction'].min():.4f}, "
              f"{results['traditional_correction'].max():.4f}] mGal")
    else:
        print("✗ Traditional method: FAILED")
    
    if results['neural_valid']:
        print("\n✓ Neural network method: VALID")
        print(f"  Range: [{results['neural_correction'].min():.4f}, "
              f"{results['neural_correction'].max():.4f}] mGal")
        print(f"  Gravity weights: {results['gravity_weights']}")
    else:
        print("\n✗ Neural network method: SKIPPED")
    
    if 'terrain_complexity' in results:
        print(f"\nTerrain complexity: {results['terrain_complexity']:.4f}")
        print(f"Adaptive blend ratio: {results['adaptive_blend_ratio']:.4f}")
    
    print("\nFinal Terrain Correction Statistics:")
    stats = results['correction_stats']
    print(f"  Min:  {stats['min']:.4f} mGal")
    print(f"  Max:  {stats['max']:.4f} mGal")
    print(f"  Mean: {stats['mean']:.4f} mGal")
    print(f"  Std:  {stats['std']:.4f} mGal")
    
    # Save results
    print("\nSaving results...")
    system.save_results(results, str(output_dir))
    
    print(f"\n✓ Results saved to {output_dir}")
    print("  - traditional_correction.npy")
    print("  - neural_correction.npy")
    print("  - final_correction.npy")
    print("  - metadata.json")
    print("="*70)


if __name__ == '__main__':
    main()
