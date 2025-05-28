#!/bin/bash
set -e

echo "Installing PyTorch with CUDA support for RunPod..."
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "Installing other ML dependencies..."
pip install \
    runpod>=1.5.0 \
    diffusers>=0.30.0 \
    transformers>=4.40.0 \
    accelerate>=0.24.0 \
    safetensors>=0.4.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    boto3>=1.34.0 \
    peft>=0.8.0 \
    protobuf>=3.20.0 \
    sentencepiece>=0.1.99 \
    requests>=2.31.0 \
    controlnet-aux>=0.0.6 \
    opencv-python-headless>=4.8.0 \
    urllib3>=1.26.0 \
    psutil>=5.9.0 \
    typing-extensions>=4.8.0 \
    packaging>=23.0

echo "Clearing pip cache..."
pip cache purge

echo "Installation completed successfully!" 