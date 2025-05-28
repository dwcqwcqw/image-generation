# Use NVIDIA CUDA base image with PyTorch
# Build: $(date +%s) - Force cache refresh
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
    && rm -rf /var/lib/apt/lists/*

# Fix NumPy version compatibility issue FIRST
RUN pip install --force-reinstall "numpy>=1.24.0,<2.0"

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/handler.py .

# Create volume mount point for models
RUN mkdir -p /runpod-volume

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Expose port (if needed for local testing)
EXPOSE 8000

# Default command
CMD ["python", "handler.py"] 