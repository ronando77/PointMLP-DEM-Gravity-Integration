# Usage Guide

## Quick Start

### 1. Point Cloud Classification

```python
import torch
from src.pointmlp import PointMLP
from src.pointmlp import utils

# Initialize model
model = PointMLP(num_classes=40)
model.eval()

# Load point cloud (shape: [batch_size, num_points, 3])
point_cloud = torch.randn(1, 1024, 3)

# Normalize
point_cloud = utils.normalize_point_cloud(point_cloud)

# Classify
with torch.no_grad():
    logits = model(point_cloud)
    predictions = torch.argmax(logits, dim=1)

print(f"Predictions: {predictions}")
print(f"Confidence: {torch.softmax(logits, dim=1).max(dim=1)[0]}")
```

### 2. DEM-SAR Fusion

```python
import torch
from src.dem_fusion import PIMSRModel

# Initialize model
model = PIMSRModel(upscale_factor=4)
model.eval()

# Prepare data
dem_lr = torch.randn(1, 1, 64, 64)  # Low-res DEM
sar_hr = torch.randn(1, 1, 256, 256)  # High-res SAR

# Fuse
with torch.no_grad():
    dem_sr = model(dem_lr, sar_hr)

print(f"Super-resolved DEM shape: {dem_sr.shape}")
```

### 3. Gravity Terrain Correction

```python
import torch
from src.gravity_correction import AdaptiveTerrainCorrection

# Initialize model
model = AdaptiveTerrainCorrection()
model.eval()

# Prepare data
dem = torch.randn(1, 1, 256, 256)
gravity = torch.randn(1, 1, 256, 256)

# Apply correction
with torch.no_grad():
    corrected_gravity, corrections, complexity = model(dem, gravity)

print(f"Terrain complexity: {complexity.item():.4f}")
```

### 4. Integrated Pipeline

```python
from src.integration import IntegrationPipeline
import torch

# Initialize pipeline
pipeline = IntegrationPipeline(device='cuda')

# Prepare data
point_cloud = torch.randn(1, 1024, 3)
dem_lr = torch.randn(1, 1, 64, 64)
sar_hr = torch.randn(1, 1, 256, 256)
gravity = torch.randn(1, 1, 256, 256)

# Run full pipeline
results = pipeline.full_pipeline(point_cloud, dem_lr, sar_hr, gravity)

print("Results:")
for key, value in results.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape {value.shape}")
    else:
        print(f"  {key}: {value}")
```

## Configuration

Create a `config.yaml` file:

```yaml
pointcloud_num_points: 1024
pointcloud_num_classes: 40

dem_upscale_factor: 4
dem_patch_size: 64

gravity_dem_resolution: 30
gravity_depth_layers: 5

batch_size: 8
learning_rate: 0.001
num_epochs: 100
device: cuda

data_dir: ./data
output_dir: ./output
checkpoint_dir: ./checkpoints
```

Load configuration:

```python
from src.integration import Config

config = Config.from_yaml('config.yaml')
```

## Training

Example training loop:

```python
import torch
import torch.optim as optim
from src.pointmlp import PointMLP

model = PointMLP(num_classes=40)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    # Training loop
    model.train()
    for point_cloud, label in dataloader:
        optimizer.zero_grad()
        logits = model(point_cloud)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Model Checkpointing

```python
from src.integration import IntegrationPipeline

# Initialize pipeline
pipeline = IntegrationPipeline()

# Save checkpoint
pipeline.save_checkpoint('checkpoints/model.pt')

# Load checkpoint
pipeline.load_checkpoint('checkpoints/model.pt')
```

## Advanced: Custom Data Loading

```python
from src.dem_fusion import DEMSARDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = DEMSARDataset(
    dem_dir='./data/dem',
    sar_dir='./data/sar',
    upscale_factor=4,
    patch_size=64
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```
