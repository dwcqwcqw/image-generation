# Cloudflare Pages 配置检查清单

## 🔧 需要检查的配置项

### 1. 环境变量设置
在 Cloudflare Pages Dashboard > Settings > Environment variables 中确认：

```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
NEXT_PUBLIC_RUNPOD_API_KEY=your_runpod_api_key  
NEXT_PUBLIC_RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
```

### 2. 构建设置
- **Build command**: `cd frontend && npm run build`
- **Build output directory**: `frontend/out` 或 `frontend/.next`
- **Node.js version**: 18.x 或更高

### 3. Functions (API Routes) 配置
确保在 Pages > Settings > Functions 中：
- ✅ Functions enabled
- ✅ Compatibility date: 2023-05-18 或更新

### 4. CORS 和安全设置
在 Security > Origin Rules 中可能需要添加规则

## 🚨 常见问题

1. **API 路由 404**: Cloudflare Pages 可能不支持 `/api/*` 路由
2. **CORS 错误**: 需要在 R2 或 Cloudflare 层面配置
3. **环境变量**: 生产环境变量可能未正确设置 