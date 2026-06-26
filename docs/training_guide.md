# Training Guide

## Quick Start

### 1. Prepare Your Data

Organize your LAS files in a directory:

```
data/
├── point_clouds/
│   ├── cloud1.las
│   ├── cloud2.las
│   ├── cloud3.las
│   └── ...
```

### 2. Configure Training

Edit `config_train.yaml`:

```yaml
data_dir: './data/point_clouds'
num_points: 1024
batch_size: 16
num_epochs: 100
learning_rate: 0.001
scheduler: 'cosine'
augment: true
device: 'cuda'
```

### 3. Run Training

```bash
python train.py --config config_train.yaml
```

Or with command line arguments:

```bash
python train.py \
  --data-dir ./data/point_clouds \
  --num-epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --scheduler cosine \
  --augment \
  --rotation \
  --scaling \
  --jitter
```

## Advanced Training Options

### Optimizers

```bash
python train.py \
  --data-dir ./data \
  --optimizer adam          # adam, adamw, or sgd
  --weight-decay 1e-4
```

### Learning Rate Schedulers

```bash
# Cosine annealing (recommended)
python train.py --data-dir ./data --scheduler cosine

# Step decay
python train.py --data-dir ./data --scheduler step

# Exponential decay
python train.py --data-dir ./data --scheduler exponential

# Reduce on plateau
python train.py --data-dir ./data --scheduler plateau
```

### Data Augmentation

```bash
python train.py \
  --data-dir ./data \
  --augment \
  --rotation      # Random rotation
  --scaling       # Random scaling
  --jitter        # Add Gaussian noise
  --dropout       # Random point dropout
```

### Multi-GPU Training

```bash
python train.py \
  --data-dir ./data \
  --multi-gpu
```

### Resume Training

```bash
python train.py \
  --data-dir ./data \
  --model-path ./checkpoints/model_best.pt
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

### Training Output

```
Epoch 1/100 | Train Loss: 2.3456, Train Acc: 0.1234 | Val Loss: 2.1234, Val Acc: 0.1567 | LR: 0.001000
Epoch 2/100 | Train Loss: 2.1234, Train Acc: 0.2134 | Val Loss: 1.9234, Val Acc: 0.2567 | LR: 0.000999
...
```

## Evaluation

### Evaluate Trained Model

```bash
python evaluate.py \
  --model-path ./checkpoints/model_best.pt \
  --data-dir ./data/point_clouds \
  --output-dir ./results
```

### Output

- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curves.png` - ROC curves for each class
- Classification report with precision, recall, F1-score

## Model Optimization

### Quantization

```python
from src.training.optimization import quantize_model

# Dynamic quantization (no calibration needed)
quantized_model = quantize_model(model, quantize_type='dynamic')

# Static quantization (requires calibration data)
quantized_model = quantize_model(model, quantize_type='static', 
                                calibration_data=calibration_loader)
```

### Pruning

```python
from src.training.optimization import prune_model

# Structured pruning (prune entire channels)
pruned_model, sparsity = prune_model(model, pruning_ratio=0.3, method='structured')

# Unstructured pruning (prune individual weights)
pruned_model, sparsity = prune_model(model, pruning_ratio=0.3, method='unstructured')

print(f"Model sparsity: {sparsity:.2%}")
```

### Model Size

```python
from src.training.optimization import get_model_size

size_mb, num_params = get_model_size(model)
print(f"Model size: {size_mb:.2f} MB")
print(f"Parameters: {num_params:,}")
```

## Training Tips

1. **Data Augmentation**: Use rotation, scaling, and jitter for better generalization
2. **Learning Rate**: Start with 0.001, adjust based on loss curves
3. **Batch Size**: Larger batches (32-64) often work better with sufficient GPU memory
4. **Scheduler**: Cosine annealing typically works best for this type of model
5. **Checkpoints**: Save best model and latest checkpoint for comparison
6. **Early Stopping**: Monitor validation loss and stop if not improving

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size: `--batch-size 4`
- Reduce num_points: `--num-points 512`
- Use gradient accumulation

### Training Diverges

- Reduce learning rate: `--learning-rate 0.0001`
- Increase clip norm: `--clip-norm 2.0`
- Add weight decay: `--weight-decay 1e-3`

### Validation Accuracy Not Improving

- Increase training time: `--num-epochs 200`
- Improve data augmentation
- Check data quality and labels
- Increase model capacity

## Configuration Examples

### Fast Training (for testing)

```yaml
data_dir: './data'
num_epochs: 10
batch_size: 32
learning_rate: 0.01
num_workers: 0
```

### Production Training

```yaml
data_dir: './data'
num_epochs: 200
batch_size: 32
learning_rate: 0.001
weight_decay: 1e-4
scheduler: 'cosine'
augment: true
multi_gpu: true
```
