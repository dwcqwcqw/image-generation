# Use NVIDIA CUDA base image with PyTorch
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopencv-dev \
    python3-opencv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .

# Install dependencies with error handling
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt || \
    (echo "Initial pip install failed, trying with --force-reinstall" && \
     pip install --no-cache-dir --force-reinstall -r requirements.txt)

# Copy application code
COPY backend/handler.py .

# Create volume mount point for models
RUN mkdir -p /runpod-volume

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV CUDA_VISIBLE_DEVICES=0

# Health check to verify the container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Expose port (if needed for local testing)
EXPOSE 8000

# Default command with error handling
CMD ["python", "-u", "handler.py"] 