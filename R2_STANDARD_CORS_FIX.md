# 🔧 R2 标准格式 CORS 权限修复指南

## 问题分析

根据你的测试截图，R2 URL返回 HTTP 400 "InvalidArgument - Authorization" 错误：
```
https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/xxx.png
```

这是R2存储桶的**权限配置问题**，需要正确配置Public Access和CORS策略。

## 📋 修复步骤

### 步骤1: 启用R2存储桶公共访问

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com)
2. 进入 **R2 Object Storage**
3. 点击你的存储桶（`image-generation`）
4. 进入 **Settings** 标签
5. 找到 **Public Access** 部分
6. 将设置更改为 **"Allow"**

### 步骤2: 配置CORS策略

在同一个Settings页面，找到 **CORS policy** 部分，设置为：

```json
[
  {
    "AllowedOrigins": [
      "https://d024556d.image-generation-dfn.pages.dev",
      "https://*.image-generation-dfn.pages.dev", 
      "https://*.pages.dev",
      "http://localhost:3000",
      "http://localhost:3001",
      "*"
    ],
    "AllowedMethods": ["GET", "HEAD", "OPTIONS"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["Content-Length", "Content-Type", "ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

**重要注意事项：**
- 确保字段名是 `ExposeHeaders` 而不是 `ExposedHeaders`
- 确保字段名是 `AllowedOrigins` 而不是 `AllowOrigins`
- `MaxAgeSeconds` 不要设置过大（建议3600秒）

### 步骤3: 验证存储桶权限

确保R2存储桶有正确的权限设置：

1. 在R2存储桶设置中，确认 **Public Access** 为 **"Allow"**
2. 确认存储桶策略允许公共读取访问

### 步骤4: 测试访问

配置完成后等待5-10分钟让策略生效，然后：

1. 清除浏览器缓存
2. 重新生成一张图片
3. 检查图片URL是否正常访问

## 🔧 故障排除

### 如果仍然有CORS错误

1. **检查域名配置**：确保你的Cloudflare Pages域名包含在CORS策略中
2. **等待传播**：CORS策略可能需要几分钟才能生效
3. **清除缓存**：清除浏览器和CDN缓存

### 如果仍然有认证错误

1. **检查ACL设置**：确保上传时设置了 `ACL='public-read'`
2. **检查存储桶策略**：可能需要添加存储桶策略允许公共访问

### 验证命令

使用curl测试图片URL：
```bash
curl -I "https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/test.png"
```

成功的响应应该包含：
```
HTTP/2 200
Access-Control-Allow-Origin: *
Content-Type: image/png
```

## 🚨 重要提醒

1. **不要启用Public Development URL**：既然你已经禁用了，保持禁用状态
2. **使用标准R2格式**：继续使用 `bucket.account-id.r2.cloudflarestorage.com` 格式
3. **关注权限配置**：问题的根源是R2存储桶权限，不是URL格式

## ✅ 预期结果

修复后你应该看到：

1. **后端日志**：
   ```
   ✓ Successfully uploaded to (standard R2): https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/xxx.png
   ```

2. **前端**：
   - 图片正常显示
   - 下载功能正常工作
   - 无CORS错误

3. **HTTP状态**：
   - 图片请求返回200状态码
   - 包含正确的CORS头

## 📞 需要帮助？

如果按照上述步骤仍然有问题，请提供：

1. R2存储桶Settings页面的截图
2. 浏览器Network标签中图片请求的详细信息
3. 后端日志中的上传确认信息

我会进一步帮你调试配置！ 