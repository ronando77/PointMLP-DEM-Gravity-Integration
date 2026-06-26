"""Training script for PointMLP model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
from tqdm import tqdm
import json
from datetime import datetime


class Trainer:
    """Trainer for point cloud models"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
    ):
        """Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=str(self.log_dir / timestamp))
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
        }
        
        self.best_val_acc = 0
        self.global_step = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 100,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        clip_norm: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 5,
    ) -> Dict:
        """Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Number of training epochs
            scheduler: Learning rate scheduler
            clip_norm: Gradient clipping norm
            log_interval: Logging interval
            save_interval: Checkpoint save interval
            
        Returns:
            Training history dictionary
        """
        self.model.to(self.device)
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(
                train_loader, optimizer, criterion, clip_norm, log_interval
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Update learning rate
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            self.history['learning_rate'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, optimizer, scheduler, val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch + 1, optimizer, scheduler, val_metrics['accuracy'], 
                                   is_best=True)
        
        self.writer.close()
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        clip_norm: float,
        log_interval: int,
    ) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(points)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            acc = total_correct / total_samples
            pbar.set_postfix({'loss': loss.item():.4f}, {'acc': acc:.4f})
            
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(points)
                loss = criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item() * labels.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                acc = total_correct / total_samples
                pbar.set_postfix({'loss': loss.item():.4f}, {'acc': acc:.4f})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        val_acc: float = 0,
        is_best: bool = False,
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history,
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch:03d}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "model_best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0)
