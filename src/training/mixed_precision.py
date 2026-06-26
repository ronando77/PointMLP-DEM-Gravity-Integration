"""Mixed Precision Training (AMP)"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional
from tqdm import tqdm


class MixedPrecisionTrainer:
    """Trainer with Automatic Mixed Precision (AMP)"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_amp: bool = True,
    ):
        """Initialize mixed precision trainer
        
        Args:
            model: Model to train
            device: Device to train on
            use_amp: Whether to use mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        clip_norm: float = 1.0,
    ) -> dict:
        """Train one epoch with mixed precision
        
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
        
        pbar = tqdm(train_loader, desc="Training")
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                # Mixed precision training
                with autocast():
                    logits = self.model(points)
                    loss = criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training
                logits = self.model(points)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                optimizer.step()
            
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
    
    def validate(
        self,
        val_loader,
        criterion,
    ) -> dict:
        """Validate with mixed precision
        
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
        
        pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(points)
                        loss = criterion(logits, labels)
                else:
                    logits = self.model(points)
                    loss = criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                acc = total_correct / total_samples
                pbar.set_postfix({'loss': loss.item():.4f}, {'acc': acc:.4f})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
