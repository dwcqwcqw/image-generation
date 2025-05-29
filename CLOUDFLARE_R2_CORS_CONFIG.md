# Cloudflare R2 CORS Configuration Guide

## Problem
前端无法显示和下载图片，出现CORS错误：
```
Access to fetch at 'https://...' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

## Solution
需要为Cloudflare R2存储桶配置CORS策略。

## Steps to Configure CORS in Cloudflare Dashboard

1. **登录Cloudflare Dashboard**
   - 访问 https://dash.cloudflare.com/
   - 选择你的账户

2. **进入R2存储**
   - 在左侧菜单中选择 "R2 Object Storage"
   - 点击 "Manage R2 API tokens" 或直接进入存储桶管理

3. **选择存储桶**
   - 找到并点击 `image-generation` 存储桶

4. **配置CORS**
   - 在存储桶页面中，找到 "Settings" 或 "CORS policy" 选项卡
   - 点击 "Add CORS policy" 或 "Edit CORS policy"

5. **添加CORS规则**
   添加以下CORS配置：
   ```json
   [
     {
       "AllowedOrigins": [
         "https://34237b51.image-generation-dfn.pages.dev",
         "https://*.pages.dev",
         "http://localhost:3000",
         "http://localhost:3001",
         "*"
       ],
       "AllowedMethods": [
         "GET",
         "HEAD",
         "PUT",
         "POST",
         "DELETE"
       ],
       "AllowedHeaders": [
         "*"
       ],
       "ExposeHeaders": [
         "ETag"
       ],
       "MaxAgeSeconds": 3600
     }
   ]
   ```

## Alternative: Using Cloudflare R2 Custom Domain (Recommended)

更好的解决方案是为R2存储桶配置自定义域名，这样可以避免CORS问题：

1. **在Cloudflare Dashboard中**
   - 进入 R2 存储桶设置
   - 选择 "Custom Domains"
   - 添加一个自定义域名，比如 `images.yourdomain.com`

2. **更新环境变量**
   修改 `Dockerfile` 中的环境变量：
   ```dockerfile
   ENV CLOUDFLARE_R2_PUBLIC_DOMAIN=https://images.yourdomain.com
   ```

3. **更新代码**
   在 `handler.py` 中使用自定义域名：
   ```python
   # 如果配置了自定义域名，使用自定义域名
   public_domain = os.getenv("CLOUDFLARE_R2_PUBLIC_DOMAIN")
   if public_domain:
       public_url = f"{public_domain}/{filename}"
   else:
       # 回退到标准R2 URL
       account_id = CLOUDFLARE_R2_ENDPOINT.split('//')[1].split('.')[0]
       public_url = f"https://{CLOUDFLARE_R2_BUCKET}.{account_id}.r2.cloudflarestorage.com/{filename}"
   ```

## Verification

配置完成后，测试访问一个图片URL：
```bash
curl -I "https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/test.png"
```

应该看到以下头部：
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, HEAD, PUT, POST, DELETE
Access-Control-Allow-Headers: *
```

## Current URLs Format

修复后的URL格式：
- 原错误格式: `https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/image-generation/generated/xxx.png`
- 正确格式: `https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/xxx.png`

## Next Steps

1. 配置上述CORS策略
2. 重新部署backend代码到RunPod
3. 测试图片生成和显示功能 