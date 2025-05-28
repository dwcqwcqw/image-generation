# 部署指南

本指南将帮助你部署AI图片生成网站的前端和后端。

## 1. 准备工作

### 1.1 RunPod 账户设置
1. 注册 [RunPod](https://www.runpod.io/) 账户
2. 获取API密钥
3. 创建一个新的Serverless端点

### 1.2 Cloudflare 账户设置
1. 注册 [Cloudflare](https://cloudflare.com/) 账户
2. 设置 R2 存储桶
3. 获取 R2 API 凭据

## 2. 后端部署 (RunPod Serverless)

### 2.1 准备模型文件
确保你的 RunPod Volume 中有以下模型：
- `/runpod-volume/flux_base` - FLUX 基础模型
- `/runpod-volume/Flux-Uncensored-V2` - FLUX LoRA 模型

### 2.2 构建 Docker 镜像
```bash
cd backend
docker build -t your-registry/ai-image-generator:latest .
docker push your-registry/ai-image-generator:latest
```

### 2.3 在 RunPod 中创建 Serverless 端点
1. 登录 RunPod 控制台
2. 创建新的 Serverless 端点
3. 配置以下设置：
   - Docker 镜像: `your-registry/ai-image-generator:latest`
   - GPU: A40/A100 或更高
   - 内存: 24GB+
   - Volume: 挂载包含模型的 Volume

### 2.4 环境变量配置
在 RunPod 端点中设置以下环境变量：
```
CLOUDFLARE_R2_ACCESS_KEY=your_r2_access_key
CLOUDFLARE_R2_SECRET_KEY=your_r2_secret_key
CLOUDFLARE_R2_BUCKET=your_bucket_name
CLOUDFLARE_R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
```

## 3. 前端部署 (Cloudflare Pages)

### 3.1 准备代码
```bash
cd frontend
npm install
npm run build
```

### 3.2 连接到 GitHub
1. 将代码推送到 GitHub 仓库
2. 在 Cloudflare Pages 中连接 GitHub 仓库

### 3.3 配置构建设置
- **构建命令**: `cd frontend && npm install && npm run build`
- **构建输出目录**: `frontend/.next`
- **Node.js 版本**: 18

### 3.4 环境变量配置
在 Cloudflare Pages 中设置以下环境变量：
```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
NEXT_PUBLIC_API_URL=https://your-domain.pages.dev/api
```

## 4. 域名配置

### 4.1 自定义域名 (可选)
1. 在 Cloudflare Pages 中添加自定义域名
2. 配置 DNS 设置
3. 启用 SSL/TLS

### 4.2 更新环境变量
如果使用自定义域名，更新：
```
NEXT_PUBLIC_API_URL=https://your-custom-domain.com/api
```

## 5. 测试部署

### 5.1 功能测试
1. 访问你的网站
2. 测试文生图功能
3. 测试图生图功能
4. 验证图片下载功能

### 5.2 性能监控
1. 检查 RunPod 端点日志
2. 监控 Cloudflare Pages 分析
3. 检查错误率和响应时间

## 6. 常见问题

### 6.1 RunPod 超时
- 增加超时设置
- 检查模型加载时间
- 优化推理参数

### 6.2 图片无法显示
- 检查 R2 存储桶权限
- 验证 CORS 设置
- 检查 URL 生成逻辑

### 6.3 API 错误
- 检查环境变量配置
- 验证 RunPod 端点状态
- 查看详细错误日志

## 7. 扩展和优化

### 7.1 性能优化
- 启用 CDN 缓存
- 优化图片压缩
- 实现请求排队

### 7.2 功能扩展
- 添加用户认证
- 实现图片库
- 添加更多模型选项

## 8. 成本优化

### 8.1 RunPod 成本
- 使用 Spot 实例
- 优化运行时间
- 监控 GPU 使用率

### 8.2 Cloudflare 成本
- 监控 R2 存储使用
- 设置存储策略
- 优化图片大小 