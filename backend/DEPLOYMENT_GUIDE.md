# 换脸优化系统部署指南
## Face Swap Optimization System Deployment Guide

### 🎯 概述

本指南详细说明如何在生产环境中部署优化后的换脸系统，确保获得最佳性能和质量。

### 📋 系统要求

#### 硬件要求
- **GPU**: NVIDIA GPU with CUDA 12.x support (推荐 RTX 4090/A100)
- **内存**: 至少 16GB RAM (推荐 32GB+)
- **存储**: 至少 50GB 可用空间
- **CPU**: 8核心以上 (推荐 16核心+)

#### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / Docker
- **Python**: 3.9-3.11 (推荐 3.10)
- **CUDA**: 12.0+ (推荐 12.1)
- **Docker**: 20.10+ (如使用容器部署)

### 🚀 快速部署

#### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv face_swap_env
source face_swap_env/bin/activate  # Linux/Mac
# face_swap_env\Scripts\activate  # Windows

# 升级pip
pip install --upgrade pip
```

#### 2. 安装CUDA依赖

```bash
# 安装NVIDIA Python包索引
pip install nvidia-pyindex

# 安装CUDA库 (生产环境必需)
pip install --extra-index-url https://pypi.nvidia.com nvidia-cublas-cu12
pip install --extra-index-url https://pypi.nvidia.com nvidia-cudnn-cu12
pip install --extra-index-url https://pypi.nvidia.com nvidia-cufft-cu12
pip install --extra-index-url https://pypi.nvidia.com nvidia-curand-cu12
pip install --extra-index-url https://pypi.nvidia.com nvidia-cusolver-cu12
pip install --extra-index-url https://pypi.nvidia.com nvidia-cusparse-cu12
pip install --extra-index-url https://pypi.nvidia.com nvidia-nvjitlink-cu12
```

#### 3. 安装核心依赖

```bash
# 安装ONNX Runtime GPU版本
pip uninstall onnxruntime -y
pip install onnxruntime-gpu

# 安装其他依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install insightface
pip install opencv-python
pip install gfpgan
pip install Pillow
pip install numpy
pip install boto3
```

#### 4. 模型文件部署

```bash
# 创建模型目录
mkdir -p /runpod-volume/models/insightface/models
mkdir -p /runpod-volume/models/gfpgan

# 下载模型文件 (需要根据实际情况调整路径)
# InsightFace模型
wget -O /runpod-volume/models/insightface/models/inswapper_128.onnx \
  "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"

# Buffalo模型 (高精度)
wget -O /runpod-volume/models/insightface/models/buffalo_sc.zip \
  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
unzip /runpod-volume/models/insightface/models/buffalo_sc.zip -d /runpod-volume/models/insightface/models/

# GFPGAN模型
wget -O /runpod-volume/models/gfpgan/GFPGANv1.4.pth \
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
```

#### 5. 环境变量配置

```bash
# 创建环境配置文件
cat > .env << EOF
# Cloudflare R2 配置
CLOUDFLARE_R2_ACCESS_KEY=your_access_key
CLOUDFLARE_R2_SECRET_KEY=your_secret_key
CLOUDFLARE_R2_BUCKET=your_bucket_name
CLOUDFLARE_R2_ENDPOINT=https://your_account_id.r2.cloudflarestorage.com
CLOUDFLARE_R2_PUBLIC_DOMAIN=https://your_custom_domain.com

# CUDA配置
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# 模型路径配置
FACE_SWAP_MODEL_PATH=/runpod-volume/models/insightface/models/inswapper_128.onnx
FACE_ANALYSIS_MODEL_PATH=/runpod-volume/models/insightface/models/buffalo_sc
GFPGAN_MODEL_PATH=/runpod-volume/models/gfpgan/GFPGANv1.4.pth
EOF
```

### 🐳 Docker部署

#### 1. Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    python3.10-dev \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3.10 install --no-cache-dir -r requirements.txt

# 安装CUDA依赖
RUN pip3.10 install nvidia-pyindex && \
    pip3.10 install --extra-index-url https://pypi.nvidia.com nvidia-cublas-cu12 && \
    pip3.10 install --extra-index-url https://pypi.nvidia.com nvidia-cudnn-cu12 && \
    pip3.10 install --extra-index-url https://pypi.nvidia.com nvidia-cufft-cu12 && \
    pip3.10 install --extra-index-url https://pypi.nvidia.com nvidia-curand-cu12 && \
    pip3.10 install --extra-index-url https://pypi.nvidia.com nvidia-cusolver-cu12 && \
    pip3.10 install --extra-index-url https://pypi.nvidia.com nvidia-cusparse-cu12 && \
    pip3.10 install --extra-index-url https://pypi.nvidia.com nvidia-nvjitlink-cu12

# 安装ONNX Runtime GPU
RUN pip3.10 uninstall onnxruntime -y && \
    pip3.10 install onnxruntime-gpu

# 复制应用代码
COPY . .

# 创建模型目录
RUN mkdir -p /runpod-volume/models

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3.10", "handler.py"]
```

#### 2. docker-compose.yml

```yaml
version: '3.8'

services:
  face-swap-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./models:/runpod-volume/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### 3. 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f face-swap-api
```

### ⚙️ 配置优化

#### 1. 性能配置

```python
# handler.py 中的优化配置
OPTIMIZATION_CONFIG = {
    # 检测配置
    "detection": {
        "high_quality": {
            "det_size": (1024, 1024),
            "det_thresh": 0.4,
            "multi_scale": True
        },
        "balanced": {
            "det_size": (640, 640),
            "det_thresh": 0.45,
            "multi_scale": False
        },
        "fast": {
            "det_size": (512, 512),
            "det_thresh": 0.5,
            "multi_scale": False
        }
    },
    
    # 混合配置
    "blending": {
        "dynamic_ratio": True,
        "region_aware": True,
        "lighting_match": True,
        "base_ratio": 0.88
    },
    
    # 增强配置
    "enhancement": {
        "adaptive_strength": True,
        "smart_blending": True,
        "preserve_original": 0.15
    }
}
```

#### 2. 内存优化

```python
# 内存管理配置
MEMORY_CONFIG = {
    "max_image_size": (2048, 2048),
    "batch_size": 1,
    "cache_models": True,
    "clear_cache_interval": 100
}
```

### 📊 监控和日志

#### 1. 性能监控

```python
# 添加到handler.py
import psutil
import GPUtil

def monitor_system_resources():
    """监控系统资源使用情况"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存使用率
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # GPU使用率
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100 if gpus else 0
    gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "gpu_usage": gpu_usage,
        "gpu_memory": gpu_memory
    }
```

#### 2. 日志配置

```python
import logging
from logging.handlers import RotatingFileHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/face_swap.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
```

### 🔧 故障排除

#### 1. CUDA问题

```bash
# 检查CUDA安装
nvidia-smi
nvcc --version

# 检查ONNX Runtime CUDA支持
python -c "import onnxruntime as ort; print('CUDA available:', 'CUDAExecutionProvider' in ort.get_available_providers())"

# 重新安装CUDA库
pip uninstall nvidia-* -y
pip install nvidia-pyindex
pip install --extra-index-url https://pypi.nvidia.com nvidia-cublas-cu12
```

#### 2. 内存问题

```bash
# 监控GPU内存
nvidia-smi -l 1

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

#### 3. 模型加载问题

```bash
# 检查模型文件
ls -la /runpod-volume/models/insightface/models/
ls -la /runpod-volume/models/gfpgan/

# 重新下载模型
rm -rf /runpod-volume/models/*
# 重新执行模型下载步骤
```

### 🧪 验证部署

#### 1. 运行测试脚本

```bash
python test_face_swap_optimization.py
```

#### 2. API健康检查

```bash
curl -X GET http://localhost:8000/health
```

#### 3. 功能测试

```bash
# 测试换脸API
curl -X POST http://localhost:8000/face-swap \
  -H "Content-Type: application/json" \
  -d '{
    "source_image": "base64_encoded_image",
    "target_image": "base64_encoded_image"
  }'
```

### 📈 性能基准

#### 预期性能指标

| 指标 | 目标值 | 备注 |
|------|--------|------|
| 换脸处理时间 | < 3秒 | 1024x1024图像 |
| GPU利用率 | > 80% | 处理期间 |
| 内存使用 | < 8GB | 峰值使用量 |
| 检测置信度 | > 0.85 | 平均值 |
| 成功率 | > 95% | 有效人脸图像 |

#### 性能优化建议

1. **批处理**: 对于多图像处理，使用批处理提升效率
2. **模型缓存**: 保持模型在内存中，避免重复加载
3. **异步处理**: 使用异步队列处理大量请求
4. **负载均衡**: 多GPU环境下分配负载

### 🔒 安全考虑

#### 1. 输入验证

```python
def validate_image_input(image_data):
    """验证输入图像"""
    # 检查文件大小
    if len(image_data) > 10 * 1024 * 1024:  # 10MB
        raise ValueError("Image too large")
    
    # 检查图像格式
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.format not in ['JPEG', 'PNG', 'WEBP']:
            raise ValueError("Unsupported image format")
    except Exception:
        raise ValueError("Invalid image data")
```

#### 2. 资源限制

```python
# 设置处理超时
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30秒超时
```

### 📝 维护指南

#### 1. 定期维护任务

- **每日**: 检查日志文件，监控系统资源
- **每周**: 清理临时文件，更新依赖包
- **每月**: 备份模型文件，性能评估

#### 2. 更新流程

```bash
# 备份当前版本
cp -r /app /app_backup_$(date +%Y%m%d)

# 更新代码
git pull origin main

# 重启服务
docker-compose restart face-swap-api
```

### 🎯 总结

通过本部署指南，您可以：

1. ✅ 正确配置CUDA环境，启用GPU加速
2. ✅ 部署所有优化功能，提升换脸质量
3. ✅ 监控系统性能，确保稳定运行
4. ✅ 处理常见问题，快速故障排除

部署完成后，换脸系统将具备：
- **高性能**: GPU加速，处理速度提升10-100倍
- **高质量**: 动态混合、区域感知、光照匹配
- **高稳定性**: 错误处理、资源监控、自动恢复
- **高可扩展性**: 容器化部署，易于扩展

如有问题，请参考故障排除章节或联系技术支持。 