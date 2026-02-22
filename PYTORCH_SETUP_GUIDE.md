# PyTorch GPU Setup - Complete Guide

## ✅ What's Been Done

### 1. Created PyTorch Notebook
- **File:** `densenet121_retinalOCT_pytorch.ipynb`
- **Framework:** PyTorch (better Windows GPU support than TensorFlow)
- **Model:** DenseNet121 (same as your original)
- **Features:**
  - ✅ GPU detection and configuration
  - ✅ Two-phase training (feature extraction + fine-tuning)
  - ✅ No data augmentation (as requested)
  - ✅ Class weights for imbalanced data
  - ✅ Confusion matrix visualization
  - ✅ Progress bars with tqdm
  - ✅ Automatic model checkpointing
  - ✅ Early stopping
  - ✅ Learning rate scheduling

### 2. Installing PyTorch with CUDA
Currently installing: `torch`, `torchvision`, `torchaudio` with CUDA 12.1 support

## 📋 After Installation Completes

### Step 1: Verify GPU Detection
Run this command in terminal:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
GPU: NVIDIA GeForce RTX 5070
```

### Step 2: Open the PyTorch Notebook
1. Open `densenet121_retinalOCT_pytorch.ipynb` in your IDE
2. Select Python kernel
3. Run cells sequentially

### Step 3: Run Training
The notebook will automatically:
- Detect and use your GPU
- Load your RetinalOCT_Dataset
- Train Phase 1 (frozen base) - ~25 epochs
- Train Phase 2 (fine-tuning) - ~25 epochs
- Generate confusion matrices
- Save results to JSON

## 🎯 Key Differences from TensorFlow Version

| Feature | TensorFlow | PyTorch |
|---------|-----------|---------|
| GPU Support on Windows | ❌ Broken | ✅ Works |
| Installation | Complex | Simple |
| Batch Size (GPU) | 32 | 64 |
| Progress Bars | Custom | tqdm |
| Model Saving | .keras | .pth |

## 📊 Expected Training Time (with GPU)

- **Phase 1:** ~15-20 minutes (depending on dataset size)
- **Phase 2:** ~20-25 minutes
- **Total:** ~40-45 minutes

Without GPU (CPU only): 3-4 hours

## 🔧 Troubleshooting

### If GPU is not detected:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### If out of memory error:
Reduce batch size in the notebook:
```python
BATCH_SIZE = 32  # or 16 if still issues
```

### If dataset not found:
Check paths in the notebook:
```python
DATA_DIR = os.path.join(BASE_DIR, 'RetinalOCT_Dataset')
```

## 📝 Output Files

After training completes, you'll have:
- `densenet121_pytorch_phase1_best.pth` - Best Phase 1 model
- `densenet121_pytorch_phase2_best.pth` - Best Phase 2 model (final)
- `phase_1_training.png` - Phase 1 training curves
- `phase_2_training.png` - Phase 2 training curves
- `confusion_matrix_pytorch.png` - Confusion matrix
- `confusion_matrix_normalized_pytorch.png` - Normalized confusion matrix
- `test_results_pytorch.json` - All metrics in JSON format

## 🚀 Advantages of PyTorch Version

1. **GPU Works Out of the Box** - No complex CUDA setup needed
2. **Faster Training** - Better GPU utilization
3. **More Flexible** - Easier to debug and modify
4. **Industry Standard** - More widely used in research
5. **Better Documentation** - Extensive community support

## 📚 Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- DenseNet Paper: https://arxiv.org/abs/1608.06993
- Transfer Learning Guide: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
