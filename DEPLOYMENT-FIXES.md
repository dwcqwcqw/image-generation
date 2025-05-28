# 🔧 Cloudflare Pages 部署问题修复

## 🚨 问题分析

根据部署日志，发现以下问题：

### 1. 输出目录不匹配
```
Error: Output directory "frontend/out" not found.
```

**原因**: Next.js默认构建到`.next`目录，但配置期望`frontend/out`

### 2. 构建时环境变量检查错误
```
Missing RunPod configuration
```

**原因**: API路由在构建时就检查环境变量，但构建时这些变量不存在

## ✅ 修复方案

### 1. 配置 Next.js 静态导出

修改 `frontend/next.config.js`，添加静态导出配置：

```javascript
const nextConfig = {
  output: 'export',           // 启用静态导出
  trailingSlash: true,        // URL末尾添加斜杠
  distDir: 'out',            // 输出到out目录
  // ... 其他配置
}
```

### 2. 修复 Cloudflare Pages 配置

更新 `deploy/cloudflare-pages.yml`：

```yaml
build:
  command: cd frontend && npm install && npm run build
  publish: frontend/out      # 正确的输出目录

build_settings:
  root_dir: "/"
  build_command: cd frontend && npm install && npm run build
  publish_directory: frontend/out
```

### 3. 修复 API 路由环境变量检查

将环境变量检查从模块级别移动到函数内部：

**之前 (❌)**:
```typescript
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID

if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
  console.error('Missing RunPod configuration')  // 构建时就报错
}

export async function POST(request: NextRequest) {
  // ...
}
```

**之后 (✅)**:
```typescript
export async function POST(request: NextRequest) {
  const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY
  const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID

  if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
    return NextResponse.json(
      { success: false, error: 'Server configuration error' },
      { status: 500 }
    )
  }
  // ...
}
```

## 🚀 部署流程

### 1. Cloudflare Pages 设置

1. **连接仓库**: `dwcqwcqw/image-generation`
2. **框架预设**: Next.js (Static HTML Export)
3. **构建命令**: `cd frontend && npm install && npm run build`
4. **构建输出目录**: `frontend/out`
5. **根目录**: `/`

### 2. 环境变量配置

在 Cloudflare Pages 项目设置中添加：

```
RUNPOD_API_KEY=<你的RunPod API密钥>
RUNPOD_ENDPOINT_ID=<你的RunPod端点ID>
NEXT_PUBLIC_API_URL=<你的Pages域名>
```

### 3. 验证部署

1. 检查构建日志没有错误
2. 访问部署的网站
3. 测试图片生成功能

## ⚠️ 注意事项

### 静态导出限制

使用 `output: 'export'` 后，以下功能会受限：
- 不支持服务器端渲染 (SSR)
- API 路由需要特殊处理
- 某些 Next.js 功能可能不可用

### API 路由处理

静态导出模式下，API 路由会被构建为静态文件，实际API功能需要通过 Cloudflare Pages Functions 或外部服务实现。

### 推荐做法

对于生产环境，建议：
1. 使用 Cloudflare Pages Functions 处理API请求
2. 或者前端直接调用 RunPod API（需要处理CORS）
3. 考虑使用 Vercel 等支持 SSR 的平台

## 🔄 重新部署

修复提交后，Cloudflare Pages 会自动重新部署。如果仍有问题：

1. 检查 Cloudflare Pages 项目设置
2. 确认环境变量正确配置
3. 查看新的构建日志
4. 必要时手动重新部署

---

修复完成后，部署应该能够成功！ 