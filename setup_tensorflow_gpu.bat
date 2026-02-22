@echo off
echo ========================================
echo TensorFlow GPU Setup for Windows
echo ========================================
echo.

echo Step 1: Uninstalling current TensorFlow...
pip uninstall tensorflow tensorflow-intel -y

echo.
echo Step 2: Installing TensorFlow 2.17.0...
pip install tensorflow==2.17.0

echo.
echo Step 3: Installing CUDA libraries...
pip install nvidia-cudnn-cu12==8.9.7.29
pip install nvidia-cuda-runtime-cu12==12.3.101

echo.
echo Step 4: Verifying GPU detection...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs available:', len(tf.config.list_physical_devices('GPU'))); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

echo.
echo ========================================
echo Setup complete!
echo ========================================
pause
