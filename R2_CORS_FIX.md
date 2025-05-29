# R2 CORS 问题修复指南

## 浏览器错误分析
根据截图显示的错误：
1. **CORS策略阻止**: 缺少 `Access-Control-Allow-Origin` 响应头
2. **资源加载失败**: `ERR_FAILED` 错误
3. **图片下载失败**: `TypeError: Failed to fetch`

## 修复步骤

### 步骤1: 检查R2存储桶公共访问
1. 登录 **Cloudflare Dashboard**
2. 进入 **R2 Object Storage** 
3. 选择存储桶 `image-generation`
4. 点击 **Settings** 标签
5. 找到 **Public Access** 部分
6. 确保设置为: **Allow**

### 步骤2: 更新CORS策略
在R2存储桶设置中，将CORS策略更新为：

```json
[
  {
    "AllowedOrigins": [
      "https://9cb921e.image-generation-dfn.pages.dev",
      "https://*.pages.dev",
      "http://localhost:3000",
      "*"
    ],
    "AllowedMethods": ["GET", "HEAD", "OPTIONS"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["*"],
    "MaxAgeSeconds": 3600
  }
]
```

### 步骤3: 验证CORS设置
等待5-10分钟让CORS策略生效，然后测试：

```bash
# 测试CORS预检请求
curl -X OPTIONS \
  -H "Origin: https://9cb921e.image-generation-dfn.pages.dev" \
  -H "Access-Control-Request-Method: GET" \
  "https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/test"
```

应该返回包含以下头的响应：
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
```

### 步骤4: 临时解决方案 - 使用代理
如果CORS问题持续存在，可以在前端添加图片代理：

在 `frontend/src/app/api/proxy-image/route.ts` 创建：

```typescript
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const imageUrl = searchParams.get('url');
  
  if (!imageUrl) {
    return NextResponse.json({ error: 'URL parameter required' }, { status: 400 });
  }

  try {
    const response = await fetch(imageUrl);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const imageBuffer = await response.arrayBuffer();
    
    return new NextResponse(imageBuffer, {
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'image/png',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json({ error: 'Failed to fetch image' }, { status: 500 });
  }
}
```

然后在前端组件中使用代理URL：
```typescript
// 将 R2 URL 转换为代理 URL
const proxyUrl = `/api/proxy-image?url=${encodeURIComponent(originalImageUrl)}`;
```

### 步骤5: 检查环境变量
确保后端环境变量正确设置：

```bash
CLOUDFLARE_R2_PUBLIC_DOMAIN=https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com
```

## 快速验证

### 测试1: 直接访问图片
复制错误中的图片URL，直接在浏览器地址栏访问：
`https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/c142bb6f-174d-41b-936a-2e85a1ba68c4.png`

- **如果加载成功**: CORS问题
- **如果404/403**: 存储桶权限问题

### 测试2: 检查响应头
使用浏览器开发工具Network标签，查看图片请求的响应头是否包含：
```
Access-Control-Allow-Origin: *
```

## 预期结果
修复后应该看到：
- ✅ 图片正常加载显示
- ✅ 下载功能正常工作
- ✅ 无CORS错误信息
- ✅ Network标签显示200状态码 