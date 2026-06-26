# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support, optional)
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/ronando77/PointMLP-DEM-Gravity-Integration.git
cd PointMLP-DEM-Gravity-Integration
```

## Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n pointmlp-dem-gravity python=3.10
conda activate pointmlp-dem-gravity

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### For GPU Support

If you have a CUDA-capable GPU, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### For CPU Only

Default installation uses CPU-only PyTorch.

## Step 4: Install Package

```bash
pip install -e .
```

## Step 5: Verify Installation

```bash
python -c "import src; print('Installation successful!')"
```

## Optional: Development Setup

For development, install additional tools:

```bash
pip install black flake8 pytest pytest-cov
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues installing PyTorch, visit: https://pytorch.org/get-started/locally/

### Harmonica Installation

Harmonica may require additional dependencies:

```bash
conda install -c conda-forge harmonica
```

### GDAL/Rasterio Issues

For rasterio installation:

```bash
conda install -c conda-forge rasterio
```

## System Requirements

- Minimum RAM: 8GB
- Recommended RAM: 16GB+ (for large datasets)
- GPU: NVIDIA GPU with 4GB+ VRAM recommended
