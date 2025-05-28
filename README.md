# AI Image Generation Website

一个基于FLUX模型的AI图片生成网站，支持文生图和图生图功能。

## 技术栈

- **前端**: Next.js + TypeScript + Tailwind CSS
- **后端**: RunPod Serverless  
- **存储**: Cloudflare R2
- **部署**: Cloudflare Pages
- **模型**: FLUX (基础模型 + Uncensored LoRA)

## 功能特性

### 文生图 (Text-to-Image)
- 正面提示词输入
- 负面提示词输入  
- 迭代步数调节
- 图片尺寸设置
- 生成数量选择 (1-4张)
- CFG Scale 调节
- 种子设置 (支持随机)

### 图生图 (Image-to-Image)  
- 本地图片上传
- 正面提示词输入
- 负面提示词输入
- 迭代步数调节
- 图片尺寸设置  
- 生成数量选择 (1-4张)
- CFG Scale 调节
- 种子设置 (支持随机)
- 重绘幅度 (Denoising) 调节

## 项目结构

```
├── frontend/           # Next.js 前端应用
├── backend/           # RunPod Serverless 后端代码
├── docs/             # 文档
└── deploy/           # 部署配置
```

## 安装和运行

### 前端开发
```bash
cd frontend
npm install
npm run dev
```

### 部署
- 前端: Cloudflare Pages
- 后端: RunPod Serverless
- 存储: Cloudflare R2

## 环境变量

需要配置以下环境变量:
- `RUNPOD_API_KEY`: RunPod API密钥
- `RUNPOD_ENDPOINT_ID`: RunPod端点ID
- `CLOUDFLARE_R2_ACCESS_KEY`: Cloudflare R2访问密钥
- `CLOUDFLARE_R2_SECRET_KEY`: Cloudflare R2秘密密钥
- `CLOUDFLARE_R2_BUCKET`: R2存储桶名称
- `CLOUDFLARE_R2_ENDPOINT`: R2端点URL 