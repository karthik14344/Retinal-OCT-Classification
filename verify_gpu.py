import torch

print('=' * 60)
print('GPU VERIFICATION - PyTorch')
print('=' * 60)
print(f'\nPyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print('\n✅ GPU IS READY FOR TRAINING!')
else:
    print('\n⚠️ GPU NOT DETECTED - Will use CPU')

print('=' * 60)
