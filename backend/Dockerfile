# Use NVIDIA official CUDA base image - 使用确实存在的NVIDIA官方镜像
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/tmp/.cache/huggingface
ENV TRANSFORMERS_CACHE=/tmp/.cache/transformers

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy and run installation script
COPY backend/install_requirements.sh .
RUN chmod +x install_requirements.sh && ./install_requirements.sh

# Copy application code
COPY backend/handler.py .

# Copy environment variables (if .env file exists)
COPY backend/.env* ./

# Create model directory
RUN mkdir -p /runpod-volume

# Set Cloudflare R2 environment variables
ENV CLOUDFLARE_R2_ACCESS_KEY=5885b29961ce9fc2b593139d9de52f81
ENV CLOUDFLARE_R2_SECRET_KEY=a4415c670e669229db451ea7b38544c0a2e44dbe630f1f35f99f28a27593d181
ENV CLOUDFLARE_R2_BUCKET=image-generation
ENV CLOUDFLARE_R2_ENDPOINT=https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com

# Run the application
CMD ["python", "-u", "handler.py"] 