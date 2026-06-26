"""Data preprocessing script for LAS files"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PointCloudPreprocessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess LAS point cloud data')
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing LAS files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed files')
    parser.add_argument('--num-points', type=int, default=1024,
                       help='Target number of points')
    parser.add_argument('--remove-outliers', action='store_true',
                       help='Remove outliers')
    parser.add_argument('--downsample', action='store_true',
                       help='Downsample using voxels')
    parser.add_argument('--voxel-size', type=float, default=0.1,
                       help='Voxel size for downsampling')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize point clouds')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main preprocessing function"""
    args = parse_args()
    
    print("="*60)
    print("Point Cloud Data Preprocessing")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target points: {args.num_points}")
    print(f"Remove outliers: {args.remove_outliers}")
    print(f"Downsample: {args.downsample}")
    print(f"Normalize: {args.normalize}")
    print("="*60)
    
    # Create preprocessor
    preprocessor = PointCloudPreprocessor(verbose=args.verbose)
    
    # Batch process
    print("\nProcessing files...")
    results = preprocessor.batch_process(
        directory=args.input_dir,
        output_dir=args.output_dir,
        num_points=args.num_points,
        remove_outliers=args.remove_outliers,
        downsample=args.downsample,
        voxel_size=args.voxel_size,
        normalize=args.normalize,
    )
    
    print(f"\nProcessed {len(results)} files")
    print(f"Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()
