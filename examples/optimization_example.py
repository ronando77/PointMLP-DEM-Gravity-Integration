"""Example model optimization (quantization and pruning)"""

import torch
import torch.nn as nn
from src.pointmlp import PointMLP
from src.training.optimization import (
    quantize_model,
    prune_model,
    get_model_size,
)


def main():
    """Example optimization"""
    
    # Load model
    model = PointMLP(num_classes=40, num_points=1024)
    
    print("Original Model:")
    print("-" * 50)
    size_mb, num_params = get_model_size(model)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Parameters: {num_params:,}")
    
    # Test inference
    test_input = torch.randn(1, 1024, 3)
    with torch.no_grad():
        output = model(test_input)
    print(f"Output shape: {output.shape}")
    
    # Quantization
    print("\nQuantized Model (Dynamic):")
    print("-" * 50)
    quantized_model = quantize_model(model, quantize_type='dynamic')
    size_mb, num_params = get_model_size(quantized_model)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Parameters: {num_params:,}")
    
    # Test quantized inference
    with torch.no_grad():
        output_q = quantized_model(test_input)
    print(f"Output shape: {output_q.shape}")
    
    # Pruning
    print("\nPruned Model (30% unstructured):")
    print("-" * 50)
    pruned_model, sparsity = prune_model(
        model,
        pruning_ratio=0.3,
        method='unstructured'
    )
    size_mb, num_params = get_model_size(pruned_model)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Parameters: {num_params:,}")
    print(f"Sparsity: {sparsity:.2%}")
    
    # Test pruned inference
    with torch.no_grad():
        output_p = pruned_model(test_input)
    print(f"Output shape: {output_p.shape}")
    
    # Combined: Prune then Quantize
    print("\nPruned + Quantized Model:")
    print("-" * 50)
    pruned_model, sparsity = prune_model(model, pruning_ratio=0.3)
    quantized_pruned = quantize_model(pruned_model, quantize_type='dynamic')
    size_mb, num_params = get_model_size(quantized_pruned)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Parameters: {num_params:,}")
    print(f"Sparsity: {sparsity:.2%}")
    
    # Save optimized model
    print("\nSaving optimized models...")
    torch.save(quantized_model.state_dict(), 'model_quantized.pt')
    torch.save(pruned_model.state_dict(), 'model_pruned.pt')
    torch.save(quantized_pruned.state_dict(), 'model_optimized.pt')
    print("Done!")


if __name__ == '__main__':
    main()
