"""Learning rate schedulers"""

import torch.optim as optim
from typing import Optional


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'step',
    num_epochs: int = 100,
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('step', 'cosine', 'exponential', 'linear', 'plateau')
        num_epochs: Number of training epochs
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Learning rate scheduler or None
    """
    
    if scheduler_type == 'step':
        # Step decay
        step_size = kwargs.get('step_size', num_epochs // 5)
        gamma = kwargs.get('gamma', 0.5)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        # Cosine annealing
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    
    elif scheduler_type == 'cosine_warmup':
        # Cosine annealing with warm restarts
        T_0 = kwargs.get('T_0', num_epochs // 10)
        T_mult = kwargs.get('T_mult', 1)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult
        )
    
    elif scheduler_type == 'exponential':
        # Exponential decay
        gamma = kwargs.get('gamma', 0.99)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'linear':
        # Linear decay
        return optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs
        )
    
    elif scheduler_type == 'plateau':
        # Reduce on plateau
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 10)
        min_lr = kwargs.get('min_lr', 1e-6)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, patience=patience, min_lr=min_lr
        )
    
    else:
        print(f"Unknown scheduler type: {scheduler_type}, using None")
        return None


class WarmupLR(optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup"""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int = 5,
                 num_epochs: int = 100, base_lr: float = 1e-3):
        """Initialize warmup scheduler
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            num_epochs: Total number of epochs
            base_lr: Base learning rate
        """
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for _ in self.optimizer.param_groups]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / \
                      (self.num_epochs - self.warmup_epochs)
            import math
            return [0.5 * self.base_lr * (1 + math.cos(math.pi * progress))
                   for _ in self.optimizer.param_groups]
