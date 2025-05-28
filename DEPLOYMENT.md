# 🚀 部署指南

## 概述

这个AI图片生成网站需要以下服务配置：
- **RunPod Serverless**: 后端API服务
- **Cloudflare Pages**: 前端部署
- **Cloudflare R2**: 图片存储

## 📋 前置准备

### 1. 账户注册
- [RunPod](https://runpod.io) 账户
- [Cloudflare](https://cloudflare.com) 账户

### 2. 必需的配置信息
你需要获取以下信息（请将它们保存在安全的地方）：

**RunPod 配置:**
- API Key
- Endpoint ID (部署后获得)

**Cloudflare R2 配置:**
- Access Key ID
- Secret Access Key  
- Bucket Name: `image-generation`
- Account ID (用于构建Endpoint URL)

## 🔧 RunPod Serverless 部署

### 1. 创建 Serverless 端点

1. 登录 RunPod 控制台
2. 转到 "Serverless" 部分
3. 点击 "New Endpoint"
4. 填写基本信息：
   - **Name**: `ai-image-generation`
   - **Template**: 选择 "Custom Docker Image"

### 2. 配置仓库

- **Repository**: `dwcqwcqw/image-generation`
- **Branch**: `master`
- **Dockerfile Path**: `backend/Dockerfile`

### 3. 硬件配置

- **GPU Type**: A40 或 RTX 3090
- **Container Disk**: 20GB
- **Memory**: 24GB (推荐)

### 4. 环境变量设置

在 RunPod 环境变量部分添加：

```
CLOUDFLARE_R2_ACCESS_KEY_ID=<从Cloudflare R2获取>
CLOUDFLARE_R2_SECRET_ACCESS_KEY=<从Cloudflare R2获取>
CLOUDFLARE_R2_BUCKET_NAME=image-generation
CLOUDFLARE_R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
FLUX_BASE_MODEL_PATH=/runpod-volume/flux_base
FLUX_LORA_MODEL_PATH=/runpod-volume/Flux-Uncensored-V2
```

### 5. 存储卷配置

确保模型文件在正确路径：
- FLUX 基础模型: `/runpod-volume/flux_base`
- LoRA 模型: `/runpod-volume/Flux-Uncensored-V2`

## ☁️ Cloudflare R2 设置

### 1. 创建 R2 Bucket

1. 登录 Cloudflare 控制台
2. 转到 "R2 Object Storage"
3. 创建新 Bucket：
   - **Bucket Name**: `image-generation`
   - **Location**: 选择合适的区域

### 2. 生成 API Token

1. 在 R2 页面点击 "Manage R2 API tokens"
2. 创建新 Token：
   - **Token Name**: `image-generation-api`
   - **Permissions**: Read & Write
   - **Bucket**: `image-generation`

### 3. 记录配置信息

保存以下信息：
- Access Key ID
- Secret Access Key
- Bucket Name: `image-generation`
- Endpoint URL: `https://<account-id>.r2.cloudflarestorage.com`

## 🌐 Cloudflare Pages 部署

### 1. 连接 GitHub 仓库

1. 登录 Cloudflare 控制台
2. 转到 "Pages"
3. 点击 "Create a project"
4. 连接 GitHub 仓库：`dwcqwcqw/image-generation`

### 2. 构建配置

设置构建参数：
- **Framework preset**: Next.js
- **Build command**: `cd frontend && npm install && npm run build`
- **Build output directory**: `frontend/out`
- **Root directory**: `/`

### 3. 环境变量配置

在 Pages 项目设置中添加环境变量：

```
RUNPOD_API_KEY=<从RunPod获取>
RUNPOD_ENDPOINT_ID=<RunPod部署后获得>
NEXT_PUBLIC_API_URL=<Pages部署后的域名>
CLOUDFLARE_R2_ACCESS_KEY=<R2 Access Key>
CLOUDFLARE_R2_SECRET_KEY=<R2 Secret Key>
CLOUDFLARE_R2_BUCKET=image-generation
CLOUDFLARE_R2_ENDPOINT=<R2 Endpoint URL>
```

## 🔗 配置 API 路由

### 方法1: 使用 Pages Functions (推荐)

前端 API 路由会自动转发到 RunPod 端点。

### 方法2: 直接调用 RunPod

前端可以直接调用 RunPod API：
`https://api.runpod.ai/v2/<endpoint-id>/runsync`

## ✅ 部署验证

### 1. 检查 RunPod 端点

1. 在 RunPod 控制台查看端点状态
2. 检查是否显示 "Active"
3. 查看日志确认无错误

### 2. 测试 API

使用 curl 测试 RunPod 端点：

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a beautiful sunset",
      "steps": 20,
      "width": 1024,
      "height": 1024
    }
  }'
```

### 3. 检查前端部署

1. 访问 Cloudflare Pages 提供的域名
2. 测试图片生成功能
3. 检查生成的图片能否正常显示

## 🛠️ 故障排除

### RunPod 问题

**部署失败:**
- 检查 GitHub 仓库访问权限
- 验证 Dockerfile 路径正确
- 确认分支名称 (`master`)

**运行时错误:**
- 检查环境变量是否正确设置
- 查看 RunPod 日志
- 验证模型文件路径

### Cloudflare Pages 问题

**构建失败:**
- 检查构建命令和输出目录
- 验证 Node.js 版本兼容性
- 查看构建日志

**API 连接问题:**
- 验证 RunPod API Key 和 Endpoint ID
- 检查网络连接和 CORS 设置
- 确认环境变量正确配置

### R2 存储问题

**上传失败:**
- 检查 API Token 权限
- 验证 Bucket 名称和区域
- 确认 Endpoint URL 格式

## 📞 获取支持

- **RunPod**: [支持文档](https://docs.runpod.io)
- **Cloudflare**: [开发者文档](https://developers.cloudflare.com)
- **项目**: [GitHub Issues](https://github.com/dwcqwcqw/image-generation/issues)

## 🔐 安全注意事项

- 不要在代码中硬编码 API 密钥
- 定期轮换 API 密钥
- 监控 API 使用量和费用
- 设置适当的访问权限

---

完成部署后，你将拥有一个功能齐全的 AI 图片生成网站！ 