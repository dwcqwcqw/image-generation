# 🔧 RunPod Serverless 部署问题修复

## 🚨 问题分析

根据 RunPod 部署日志，发现以下错误：

```
#9 ERROR: failed to calculate checksum of ref: "/requirements.txt": not found
#11 ERROR: failed to calculate checksum of ref: "/handler.py": not found
```

### 问题原因

1. **文件路径问题**: 后端文件位于 `backend/` 子目录中
2. **构建上下文**: Dockerfile 中的 COPY 命令找不到相对路径的文件
3. **RunPod 配置**: Dockerfile 路径与实际文件结构不匹配

## ✅ 修复方案

### 方案1: 修改 RunPod 配置（推荐）

在 RunPod Serverless 端点设置中更新：

```
Repository: dwcqwcqw/image-generation
Branch: master
Dockerfile Path: Dockerfile (使用根目录的Dockerfile)
Build Context: / (根目录)
```

### 方案2: 使用 backend 目录作为构建上下文

```
Repository: dwcqwcqw/image-generation  
Branch: master
Dockerfile Path: backend/Dockerfile
Build Context: backend (设置为backend目录)
```

## 🔄 修复内容

### 1. 创建根目录 Dockerfile

在项目根目录创建了新的 `Dockerfile`，正确引用 backend 目录中的文件：

```dockerfile
# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code  
COPY backend/handler.py .
```

### 2. 更新后端 Dockerfile

同时更新了 `backend/Dockerfile`，使其也能正确工作：

```dockerfile
COPY backend/requirements.txt .
COPY backend/handler.py .
```

## 🚀 RunPod 部署步骤

### 1. 创建新的 Serverless 端点

1. 登录 RunPod 控制台
2. 进入 Serverless 部分
3. 点击 "New Endpoint"

### 2. 基本配置

```
Name: ai-image-generation
Template: Custom Docker Image
Min Workers: 0
Max Workers: 3
Idle Timeout: 5 seconds
```

### 3. 容器配置

```
Registry: Docker Hub
Repository: dwcqwcqw/image-generation
Branch: master
Dockerfile Path: Dockerfile
Build Context: /
```

### 4. 硬件配置

```
GPU: A40 (24GB) 或 RTX 3090 (24GB)
CPU: 8 vCPUs
Memory: 24GB
Container Disk: 20GB
Volume Disk: 50GB (用于模型存储)
```

### 5. 环境变量设置

添加以下环境变量：

```bash
# Cloudflare R2 配置
CLOUDFLARE_R2_ACCESS_KEY_ID=5885b29961ce9fc2b593139d9de52f81
CLOUDFLARE_R2_SECRET_ACCESS_KEY=a4415c670e669229db451ea7b38544c0a2e44dbe630f1f35f99f28a27593d181
CLOUDFLARE_R2_BUCKET_NAME=image-generation
CLOUDFLARE_R2_ENDPOINT_URL=https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com

# 模型路径
FLUX_BASE_MODEL_PATH=/runpod-volume/flux_base
FLUX_LORA_MODEL_PATH=/runpod-volume/Flux-Uncensored-V2
```

### 6. 存储卷配置

确保模型文件在以下路径：
- `/runpod-volume/flux_base/` - FLUX 基础模型
- `/runpod-volume/Flux-Uncensored-V2/` - LoRA 模型

## 🔍 验证部署

### 1. 检查构建状态

在 RunPod 控制台查看：
- 构建日志无错误
- 端点状态显示 "Active"
- 有可用的 workers

### 2. 测试 API 端点

使用 curl 测试：

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task_type": "text-to-image",
      "params": {
        "prompt": "a beautiful sunset over mountains",
        "width": 1024,
        "height": 1024,
        "steps": 20
      }
    }
  }'
```

### 3. 检查响应

正常响应应该包含：
```json
{
  "status": "COMPLETED",
  "output": {
    "success": true,
    "data": {
      "images": ["base64_encoded_image"]
    }
  }
}
```

## 🛠️ 故障排除

### 构建失败

**问题**: 文件找不到
- **解决**: 使用根目录的 Dockerfile
- **路径**: 设置 Dockerfile Path 为 `Dockerfile`

**问题**: 依赖安装失败  
- **解决**: 检查 requirements.txt 格式
- **网络**: 确保 RunPod 能访问外部包管理器

### 运行时错误

**问题**: 模型加载失败
- **检查**: 模型文件是否在正确路径
- **权限**: 确保容器有读取权限

**问题**: R2 连接失败
- **验证**: 环境变量是否正确设置
- **网络**: 检查 R2 端点 URL

### 性能问题

**GPU 内存不足**:
- 增加 GPU 显存 (使用 A40 或 A100)
- 优化模型加载策略
- 调整批处理大小

**启动时间长**:
- 使用预热策略
- 优化模型加载
- 考虑持久化容器

## 📞 获取支持

- **RunPod 文档**: [docs.runpod.io](https://docs.runpod.io)
- **项目 Issues**: [GitHub Issues](https://github.com/dwcqwcqw/image-generation/issues)
- **Discord**: RunPod 社区

---

修复完成后，RunPod Serverless 部署应该能够成功！ 