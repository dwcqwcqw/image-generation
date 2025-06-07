# Use NVIDIA CUDA base image with Python 3.10 and CUDA 12.1 for RTX 5090 support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with force NumPy downgrade for compatibility
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --force-reinstall "numpy>=1.24.0,<2.0"

# Copy application code explicitly to ensure all files are included
COPY handler.py .
COPY face_swap_integration.py .
COPY start_debug.py .
COPY utils/ ./utils/
COPY models/ ./models/
COPY pulid/ ./pulid/

# Copy any remaining files (but explicitly ensure critical ones above)
COPY . .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create volume directories
RUN mkdir -p /runpod-volume

# Expose port (if needed for local testing)
EXPOSE 8000

# Start the handler using the debug/check script
CMD ["python", "-u", "start_debug.py"] 