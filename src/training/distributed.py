"""Distributed Training Support"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import Optional
from tqdm import tqdm


def init_distributed_mode(rank: int = 0, world_size: int = 1):
    """Initialize distributed training
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(
            backend='gloo',
            rank=rank,
            world_size=world_size,
        )


def cleanup_distributed_mode():
    """Clean up distributed training"""
    dist.destroy_process_group()


class DistributedTrainer:
    """Trainer for distributed training"""
    
    def __init__(
        self,
        model: nn.Module,
        rank: int = 0,
        world_size: int = 1,
        device: str = 'cuda',
    ):
        """Initialize distributed trainer
        
        Args:
            model: Model to train
            rank: Process rank
            world_size: Total number of processes
            device: Device type
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device if device == 'cuda' else f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        
        # Wrap model with DDP
        model = model.to(self.device)
        self.model = DDP(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=rank if torch.cuda.is_available() else None,
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        clip_norm: float = 1.0,
    ) -> dict:
        """Train one epoch with distributed training
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            clip_norm: Gradient clipping norm
            
        Returns:
            Metrics dictionary
        """
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            pbar = tqdm(train_loader, desc="Training")
        else:
            pbar = train_loader
        
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(points)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            if self.rank == 0:
                acc = total_correct / total_samples
                pbar.set_postfix({'loss': loss.item():.4f}, {'acc': acc:.4f})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> dict:
        """Validate with distributed training
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Metrics dictionary
        """
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        if self.rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
        
        with torch.no_grad():
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(points)
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                if self.rank == 0:
                    acc = total_correct / total_samples
                    pbar.set_postfix({'loss': loss.item():.4f}, {'acc': acc:.4f})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def save_checkpoint(self, path: str, epoch: int, optimizer: optim.Optimizer):
        """Save checkpoint (only on rank 0)
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer
        """
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': self.history,
            }
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, optimizer: Optional[optim.Optimizer] = None):
        """Load checkpoint
        
        Args:
            path: Path to checkpoint
            optimizer: Optimizer (optional)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0)
