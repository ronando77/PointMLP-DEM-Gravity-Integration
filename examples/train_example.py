"""Example training script"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pointmlp import PointMLP
from src.data import PointCloudDataset, PointCloudAugmentation
from src.training import Trainer, get_scheduler
from src.evaluation import Evaluator


def main():
    """Example training script"""
    
    # Configuration
    config = {
        'data_dir': './data/point_clouds',
        'num_points': 1024,
        'num_classes': 40,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
    }
    
    print(f"Using device: {config['device']}")
    
    # Create model
    model = PointMLP(
        num_classes=config['num_classes'],
        num_points=config['num_points']
    )
    
    # Create datasets
    dataset = PointCloudDataset(
        data_dir=config['data_dir'],
        num_points=config['num_points'],
        extension='.las',
        normalize=True,
        transform=PointCloudAugmentation(
            rotation=True,
            scaling=True,
            jitter=True,
        )
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
    )
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4,
    )
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_epochs=config['num_epochs'],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=config['device'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir'],
    )
    
    # Train
    criterion = nn.CrossEntropyLoss()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config['num_epochs'],
        scheduler=scheduler,
    )
    
    # Evaluate
    evaluator = Evaluator(model, device=config['device'])
    metrics = evaluator.evaluate(val_loader, criterion=criterion)
    
    print(f"\nFinal Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Validation Loss: {metrics['loss']:.4f}")
    
    # Plot results
    evaluator.plot_training_history(history)
    evaluator.plot_confusion_matrix(metrics)


if __name__ == '__main__':
    main()
