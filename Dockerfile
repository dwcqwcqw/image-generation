# Use NVIDIA CUDA base image with PyTorch
# Build: $(date +%s) - Force cache refresh
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Install system dependencies with better error handling
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
    libgcc-s1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Update pip and install core packages first
RUN python -m pip install --upgrade pip setuptools wheel

# Fix NumPy version compatibility issue FIRST
RUN pip install --force-reinstall "numpy>=1.24.0,<2.0" --no-cache-dir

# Copy requirements and install Python dependencies in stages
COPY backend/requirements.txt .

# Install dependencies with better error handling
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt || \
    (echo "First install attempt failed, retrying with --force-reinstall" && \
     pip install --no-cache-dir --force-reinstall --timeout 300 -r requirements.txt)

# Copy application code
COPY backend/handler.py .
COPY backend/start_debug.py .

# Create volume mount points for models (including new anime models)
RUN mkdir -p /runpod-volume \
    && mkdir -p /runpod-volume/flux_base \
    && mkdir -p /runpod-volume/lora \
    && mkdir -p /runpod-volume/cartoon \
    && mkdir -p /runpod-volume/cartoon/lora

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set Cloudflare R2 environment variables
ENV CLOUDFLARE_R2_ACCESS_KEY=5885b29961ce9fc2b593139d9de52f81
ENV CLOUDFLARE_R2_SECRET_KEY=a4415c670e669229db451ea7b38544c0a2e44dbe630f1f35f99f28a27593d181
ENV CLOUDFLARE_R2_BUCKET=image-generation
ENV CLOUDFLARE_R2_ENDPOINT=https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com
# ENV CLOUDFLARE_R2_PUBLIC_DOMAIN=https://images.yourdomain.com  # Optional: Custom public domain

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Expose port (if needed for local testing)
EXPOSE 8000

# Default command with debug support
CMD ["python", "start_debug.py"] 