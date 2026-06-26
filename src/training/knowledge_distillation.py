"""Knowledge Distillation (KD) training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class KDLoss(nn.Module):
    """Knowledge Distillation Loss"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        """Initialize KD loss
        
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight between distillation loss and cross-entropy loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KD loss
        
        Args:
            student_logits: Student model output
            teacher_logits: Teacher model output
            targets: Ground truth labels
            
        Returns:
            KD loss
        """
        # Soft target loss (KD loss)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Hard target loss (CE loss)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Combine losses
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return loss


class DistillationTrainer:
    """Trainer with Knowledge Distillation"""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        device: str = 'cuda',
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        """Initialize distillation trainer
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model (frozen)
            device: Device to train on
            temperature: KD temperature
            alpha: Loss weight
        """
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # KD loss
        self.kd_loss = KDLoss(temperature=temperature, alpha=alpha)
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        epoch: int,
    ) -> dict:
        """Train one epoch with KD
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch
            
        Returns:
            Metrics dictionary
        """
        self.student.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            student_logits = self.student(points)
            
            with torch.no_grad():
                teacher_logits = self.teacher(points)
            
            loss = self.kd_loss(student_logits, teacher_logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(student_logits, dim=1)
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
    ) -> dict:
        """Validate student model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Metrics dictionary
        """
        self.student.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        from tqdm import tqdm
        pbar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                student_logits = self.student(points)
                teacher_logits = self.teacher(points)
                loss = self.kd_loss(student_logits, teacher_logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                predictions = torch.argmax(student_logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                acc = total_correct / total_samples
                pbar.set_postfix({'loss': loss.item():.4f}, {'acc': acc:.4f})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
