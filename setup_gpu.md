# GPU Setup Guide for TensorFlow

## Current Status
- ✅ GPU Detected: NVIDIA GeForce RTX 5070
- ✅ NVIDIA Driver: 581.57
- ✅ CUDA Version: 13.0
- ❌ TensorFlow: CPU-only version (2.20.0)

## Fix: Install TensorFlow with GPU Support

### Option 1: Using pip (Recommended)

```bash
# Uninstall current TensorFlow
pip uninstall tensorflow

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

### Option 2: If Option 1 doesn't work

```bash
# Uninstall current TensorFlow
pip uninstall tensorflow

# Install specific version with GPU support
pip install tensorflow==2.17.0
```

### Verify Installation

After installation, run:
```bash
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

You should see: `GPUs: 1`

## Notes
- TensorFlow 2.10+ includes CUDA and cuDNN libraries automatically
- No need to manually install CUDA Toolkit or cuDNN
- Your RTX 5070 with CUDA 13.0 is fully supported
