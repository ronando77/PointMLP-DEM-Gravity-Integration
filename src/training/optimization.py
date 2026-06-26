"""Model optimization techniques (quantization, pruning)"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import warnings


def quantize_model(
    model: nn.Module,
    quantize_type: str = 'static',
    calibration_data: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Quantize model to INT8
    
    Args:
        model: PyTorch model
        quantize_type: 'static' or 'dynamic'
        calibration_data: Calibration data for static quantization
        
    Returns:
        Quantized model
    """
    # Set model to eval mode
    model.eval()
    
    if quantize_type == 'dynamic':
        # Dynamic quantization (no calibration needed)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear},
            dtype=torch.qint8
        )
    elif quantize_type == 'static':
        # Static quantization (requires calibration)
        if calibration_data is None:
            warnings.warn("No calibration data provided for static quantization, using dynamic instead")
            return quantize_model(model, 'dynamic')
        
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            if isinstance(calibration_data, list):
                for batch in calibration_data:
                    model(batch)
            else:
                model(calibration_data)
        
        # Convert
        torch.quantization.convert(model, inplace=True)
        quantized_model = model
    else:
        raise ValueError(f"Unknown quantization type: {quantize_type}")
    
    return quantized_model


def prune_model(
    model: nn.Module,
    pruning_ratio: float = 0.3,
    method: str = 'structured',
) -> Tuple[nn.Module, float]:
    """Prune model weights
    
    Args:
        model: PyTorch model
        pruning_ratio: Percentage of weights to prune (0-1)
        method: 'structured' or 'unstructured'
        
    Returns:
        Tuple of (pruned_model, sparsity_achieved)
    """
    if method == 'structured':
        # Structured pruning (prune entire channels)
        pruned_model = _structured_pruning(model, pruning_ratio)
    elif method == 'unstructured':
        # Unstructured pruning (prune individual weights)
        pruned_model = _unstructured_pruning(model, pruning_ratio)
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    # Calculate sparsity
    sparsity = _calculate_sparsity(pruned_model)
    
    return pruned_model, sparsity


def _structured_pruning(model: nn.Module, pruning_ratio: float) -> nn.Module:
    """Structured pruning"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.utils.prune.ln_structured(
                module, name='weight', amount=pruning_ratio, dim=0
            )
        elif isinstance(module, nn.Linear):
            nn.utils.prune.ln_structured(
                module, name='weight', amount=pruning_ratio, dim=0
            )
    
    # Remove pruning masks
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            nn.utils.prune.remove(module, 'weight')
    
    return model


def _unstructured_pruning(model: nn.Module, pruning_ratio: float) -> nn.Module:
    """Unstructured pruning"""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.utils.prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
    
    # Remove pruning masks
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            nn.utils.prune.remove(module, 'weight')
    
    return model


def _calculate_sparsity(model: nn.Module) -> float:
    """Calculate model sparsity (percentage of zeros)"""
    total_params = 0
    total_zeros = 0
    
    for param in model.parameters():
        total_params += param.numel()
        total_zeros += (param == 0).sum().item()
    
    if total_params == 0:
        return 0.0
    
    sparsity = total_zeros / total_params
    return sparsity


def get_model_size(model: nn.Module) -> Tuple[float, float]:
    """Get model size in MB
    
    Returns:
        Tuple of (model_size_mb, num_parameters)
    """
    num_params = sum(p.numel() for p in model.parameters())
    model_size = num_params * 4 / (1024 * 1024)  # Assuming 4 bytes per float32 parameter
    
    return model_size, num_params
