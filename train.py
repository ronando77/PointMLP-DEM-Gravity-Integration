"""Training script for PointMLP model with LAS data"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
import yaml

from src.pointmlp import PointMLP
from src.data import PointCloudDataset, PointCloudAugmentation
from src.training import Trainer, get_scheduler
from src.evaluation import Evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PointMLP model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory containing LAS files')
    parser.add_argument('--num-points', type=int, default=1024,
                       help='Number of points per cloud')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--num-classes', type=int, default=40,
                       help='Number of classes')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to load pre-trained model')
    
    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'exponential', 'linear', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Augmentation arguments
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--rotation', action='store_true',
                       help='Enable random rotation')
    parser.add_argument('--scaling', action='store_true',
                       help='Enable random scaling')
    parser.add_argument('--jitter', action='store_true',
                       help='Enable jitter')
    parser.add_argument('--dropout', action='store_true',
                       help='Enable point dropout')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multiple GPUs')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save checkpoint interval')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    print("="*50)
    print("PointMLP Training Configuration")
    print("="*50)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Device: {args.device}")
    print("="*50)
    
    # Create model
    print("\nCreating model...")
    model = PointMLP(num_classes=args.num_classes, num_points=args.num_points)
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Load pre-trained model if provided
    if args.model_path:
        print(f"Loading pre-trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data augmentation
    transform = None
    if args.augment:
        print("\nEnabling data augmentation...")
        transform = PointCloudAugmentation(
            rotation=args.rotation,
            scaling=args.scaling,
            jitter=args.jitter,
            dropout=args.dropout,
        )
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = PointCloudDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        extension='.las',
        normalize=True,
        transform=transform,
        split='train',
    )
    
    val_dataset = PointCloudDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        extension='.las',
        normalize=True,
        transform=None,
        split='val',
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        num_epochs=args.num_epochs,
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        clip_norm=args.clip_norm,
        save_interval=args.save_interval,
    )
    
    # Save final history
    history_path = Path(args.checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    evaluator = Evaluator(model, device=args.device)
    metrics = evaluator.evaluate(val_loader, criterion=criterion)
    
    print(f"\nFinal Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Validation Loss: {metrics['loss']:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    evaluator.plot_training_history(
        history,
        save_path=str(Path(args.log_dir) / 'training_history.png')
    )
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    evaluator.plot_confusion_matrix(
        metrics,
        save_path=str(Path(args.log_dir) / 'confusion_matrix.png')
    )
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
