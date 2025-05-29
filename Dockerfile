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

# Set Cloudflare R2 environment variables
ENV CLOUDFLARE_R2_ACCESS_KEY=5885b29961ce9fc2b593139d9de52f81
ENV CLOUDFLARE_R2_SECRET_KEY=a4415c670e669229db451ea7b38544c0a2e44dbe630f1f35f99f28a27593d181
ENV CLOUDFLARE_R2_BUCKET=image-generation
ENV CLOUDFLARE_R2_ENDPOINT=https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com

# Expose port (if needed for local testing)
EXPOSE 8000

# Default command
CMD ["python", "handler.py"] 