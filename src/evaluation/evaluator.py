"""Model evaluation utilities"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """Evaluator for classification models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """Initialize evaluator
        
        Args:
            model: PyTorch model
            device: Device to evaluate on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module = None,
    ) -> Dict:
        """Evaluate model
        
        Args:
            data_loader: Data loader
            criterion: Loss function (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating")
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(points)
                
                # Loss
                if criterion:
                    loss = criterion(logits, labels)
                    total_loss += loss.item() * labels.size(0)
                
                # Predictions
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
        }
        
        if criterion:
            metrics['loss'] = total_loss / len(all_labels)
        
        return metrics
    
    def get_classification_report(
        self,
        metrics: Dict,
        class_names: list = None,
    ) -> str:
        """Get classification report
        
        Args:
            metrics: Metrics dictionary from evaluate()
            class_names: Optional list of class names
            
        Returns:
            Classification report string
        """
        predictions = metrics['predictions']
        labels = metrics['labels']
        
        report = classification_report(
            labels, predictions,
            target_names=class_names,
            digits=4
        )
        return report
    
    def plot_confusion_matrix(
        self,
        metrics: Dict,
        class_names: list = None,
        save_path: str = None,
    ):
        """Plot confusion matrix
        
        Args:
            metrics: Metrics dictionary
            class_names: Optional class names
            save_path: Path to save figure
        """
        predictions = metrics['predictions']
        labels = metrics['labels']
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(
        self,
        metrics: Dict,
        save_path: str = None,
    ):
        """Plot ROC curves
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save figure
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        labels = metrics['labels']
        probs = metrics['probabilities']
        
        n_classes = probs.shape[1]
        
        # Binarize labels
        labels_bin = label_binarize(labels, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: str = None,
    ):
        """Plot training history
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
