"""Distributed training script"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from pathlib import Path
import sys
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pointmlp import PointMLP
from src.data import PointCloudDataset, PointCloudAugmentation
from src.training.distributed import (
    DistributedTrainer, init_distributed_mode, cleanup_distributed_mode
)
from src.training import get_scheduler


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Distributed training')
    
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-points', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count())
    
    return parser.parse_args()


def train_worker(rank, world_size, args):
    """Training worker function"""
    # Initialize distributed training
    init_distributed_mode(rank=rank, world_size=world_size)
    
    if rank == 0:
        print(f"Training on GPU {rank}")
    
    # Create model
    model = PointMLP(num_classes=40, num_points=args.num_points)
    
    # Create distributed trainer
    trainer = DistributedTrainer(
        model=model,
        rank=rank,
        world_size=world_size,
        device='cuda',
    )
    
    # Create datasets
    train_dataset = PointCloudDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        extension='.las',
        normalize=True,
        split='train',
    )
    
    val_dataset = PointCloudDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        extension='.las',
        normalize=True,
        split='val',
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        trainer.model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_epochs=args.num_epochs,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        
        train_metrics = trainer.train_epoch(
            train_loader, optimizer, criterion
        )
        
        val_metrics = trainer.validate(val_loader, criterion)
        
        if scheduler:
            scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch+1:03d}.pt"
                trainer.save_checkpoint(str(checkpoint_path), epoch+1, optimizer)
    
    # Cleanup
    cleanup_distributed_mode()


def main():
    """Main function"""
    args = parse_args()
    
    print("="*60)
    print("Distributed Training")
    print("="*60)
    print(f"World size: {args.world_size}")
    print(f"Data directory: {args.data_dir}")
    print("="*60)
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    # Launch distributed training
    if args.world_size > 1:
        mp.spawn(
            train_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True,
        )
    else:
        train_worker(0, 1, args)


if __name__ == '__main__':
    main()
