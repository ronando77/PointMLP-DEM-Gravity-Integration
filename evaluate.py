"""Evaluation script for trained models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from src.pointmlp import PointMLP
from src.data import PointCloudDataset
from src.evaluation import Evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate PointMLP model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--num-points', type=int, default=1024,
                       help='Number of points per cloud')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num-classes', type=int, default=40,
                       help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Load model
    print("\nLoading model...")
    model = PointMLP(num_classes=args.num_classes, num_points=args.num_points)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.model_path}")
    
    # Create dataset
    print("\nLoading test data...")
    test_dataset = PointCloudDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        extension='.las',
        normalize=True,
        split='test',
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\nEvaluating...")
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(model, device=args.device)
    metrics = evaluator.evaluate(test_loader, criterion=criterion)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(evaluator.get_classification_report(metrics))
    
    # Plot results
    print("\nPlotting results...")
    evaluator.plot_confusion_matrix(
        metrics,
        save_path=str(output_dir / 'confusion_matrix.png')
    )
    
    try:
        evaluator.plot_roc_curves(
            metrics,
            save_path=str(output_dir / 'roc_curves.png')
        )
    except Exception as e:
        print(f"Could not plot ROC curves: {e}")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
