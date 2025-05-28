# Use NVIDIA CUDA base image with PyTorch - 更新到更新的版本
FROM runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY backend/handler.py .

# Copy environment variables (if .env file exists)
COPY backend/.env* ./

# Create model directory
RUN mkdir -p /runpod-volume

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/tmp/.cache/huggingface
ENV TRANSFORMERS_CACHE=/tmp/.cache/transformers

# Set Cloudflare R2 environment variables
ENV CLOUDFLARE_R2_ACCESS_KEY=5885b29961ce9fc2b593139d9de52f81
ENV CLOUDFLARE_R2_SECRET_KEY=a4415c670e669229db451ea7b38544c0a2e44dbe630f1f35f99f28a27593d181
ENV CLOUDFLARE_R2_BUCKET=image-generation
ENV CLOUDFLARE_R2_ENDPOINT=https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com

# Run the application
CMD ["python", "-u", "handler.py"] 