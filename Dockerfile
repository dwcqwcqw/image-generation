# Use NVIDIA CUDA base image with PyTorch
# Build: $(date +%s) - Force cache refresh: 1735571550
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Set environment variables early to prevent issues
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/transformers

# Install system dependencies with better error handling and reduced resource usage
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Update pip and install core packages first with smaller memory footprint
RUN python -m pip install --upgrade pip setuptools wheel --no-cache-dir

# Fix NumPy version compatibility issue FIRST
RUN pip install --force-reinstall "numpy>=1.24.0,<2.0" --no-cache-dir

# Copy requirements and install Python dependencies in smaller chunks to reduce memory usage
COPY backend/requirements.txt .

# Install dependencies with reduced memory usage and better error handling
RUN pip install --no-cache-dir --timeout 600 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r requirements.txt || \
    (echo "First install attempt failed, retrying with smaller batch size" && \
     pip install --no-cache-dir --force-reinstall --timeout 600 --no-deps \
     runpod boto3 requests Pillow numpy urllib3 psutil typing-extensions packaging && \
     pip install --no-cache-dir --timeout 600 \
     diffusers transformers accelerate safetensors peft protobuf sentencepiece compel \
     controlnet-aux opencv-python-headless huggingface-hub tokenizers regex)

# Copy application code
COPY backend/handler.py .
COPY backend/start_debug.py .

# Make scripts executable and set proper permissions
RUN chmod +x start_debug.py && \
    chmod 644 handler.py

# Create volume mount points for models with proper permissions
RUN mkdir -p /runpod-volume \
    && mkdir -p /runpod-volume/flux_base \
    && mkdir -p /runpod-volume/lora \
    && mkdir -p /runpod-volume/cartoon \
    && mkdir -p /runpod-volume/cartoon/lora \
    && chmod -R 755 /runpod-volume

# Create temporary directories with proper permissions
RUN mkdir -p /tmp/huggingface /tmp/transformers \
    && chmod -R 777 /tmp/huggingface /tmp/transformers

# Set Cloudflare R2 environment variables
ENV CLOUDFLARE_R2_ACCESS_KEY=5885b29961ce9fc2b593139d9de52f81
ENV CLOUDFLARE_R2_SECRET_KEY=a4415c670e669229db451ea7b38544c0a2e44dbe630f1f35f99f28a27593d181
ENV CLOUDFLARE_R2_BUCKET=image-generation
ENV CLOUDFLARE_R2_ENDPOINT=https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com

# Add health check with longer timeout for model loading
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD python -c "import torch; import runpod; print('Health check passed')" || exit 1

# Expose port (if needed for local testing)
EXPOSE 8000

# Use a more robust startup command with proper error handling
CMD ["python", "-u", "start_debug.py"] 