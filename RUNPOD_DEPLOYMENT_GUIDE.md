# RunPod Serverless 部署配置指南

## 🚨 重要：正确的RunPod配置

### 1. 基本设置
- **名称**: `flux-image-generation`
- **模板**: `Custom Docker Image`

### 2. 容器配置 ⚠️ 关键设置
```
Registry: Docker Hub
Repository: dwcqwcqw/image-generation
Branch: master
Dockerfile Path: Dockerfile          # ← 必须是这个，不是 backend/Dockerfile
Build Context: /                     # ← 必须是根目录斜杠，不是 backend
```

### 3. 硬件配置
```
GPU: RTX 3090 (24GB) 或 A40 (48GB)
CPU: 8 vCPUs
RAM: 24GB
Container Disk: 20GB
Volume Disk: 50GB
```

### 4. 环境变量
```bash
CLOUDFLARE_R2_ACCESS_KEY=5885b29961ce9fc2b593139d9de52f81
CLOUDFLARE_R2_SECRET_KEY=a4415c670e669229db451ea7b38544c0a2e44dbe630f1f35f99f28a27593d181
CLOUDFLARE_R2_BUCKET=image-generation
CLOUDFLARE_R2_ENDPOINT=https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com
```

### 5. 网络配置
```
Min Workers: 0
Max Workers: 3
Idle Timeout: 5 seconds
Max Execution Time: 600 seconds
```

## 🔧 常见配置错误

### ❌ 错误配置
```
Dockerfile Path: backend/Dockerfile   # 这是错误的！
Build Context: backend               # 这是错误的！
```

### ✅ 正确配置
```
Dockerfile Path: Dockerfile          # 正确！
Build Context: /                     # 正确！
```

## 🚀 部署步骤

1. **进入RunPod控制台**
2. **找到你的Serverless端点**
3. **点击"Settings"或"Edit"**
4. **确认"Container Configuration"部分**:
   - Repository: `dwcqwcqw/image-generation`
   - Branch: `master`
   - **Dockerfile Path**: `Dockerfile` (不是 `backend/Dockerfile`)
   - **Build Context**: `/` (不是 `backend`)
5. **保存设置**
6. **点击"Deploy"或"Build"按钮**

## 🔍 验证构建

构建成功的日志应该显示：
```
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel
COPY backend/requirements.txt .
Successfully installed numpy-1.26.4
COPY backend/handler.py .
=== Starting AI Image Generation Backend ===
```

如果仍然看到错误，请检查你的RunPod UI配置是否与上面完全一致。 