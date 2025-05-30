# 🔧 R2 Public Domain 修复指南

## 问题分析

根据你的测试截图：

1. **老 R2 URL** 返回 HTTP 400 "InvalidArgument - Authorization":
   ```
   https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/xxx.png
   ```

2. **新 R2 URL** 完美工作:
   ```
   https://pub-5a18b069cd0644588901.r2.dev/generated/xxx.png
   ```

## 💡 解决方案

你的 R2 存储桶已经启用了 **Public Access**，但使用了新的 public domain 格式。我们需要更新后端以生成正确的 public URLs。

## 📋 修复步骤

### 步骤 1: 获取正确的 R2 Public Domain

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com)
2. 进入 **R2 Object Storage**
3. 点击你的存储桶（`image-generation`）
4. 进入 **Settings** 标签
5. 找到 **Public Access** 部分
6. 复制 **Public R2.dev Subdomain** URL

应该类似：`pub-xxxxxxxxx.r2.dev`

### 步骤 2: 更新 RunPod 环境变量

在 RunPod 容器的环境变量中添加：

```bash
CLOUDFLARE_R2_PUBLIC_BUCKET_DOMAIN=pub-xxxxxxxxx.r2.dev
```

将 `pub-xxxxxxxxx.r2.dev` 替换为你在步骤1中获取的实际域名。

### 步骤 3: 重新部署容器

1. 停止当前 RunPod 容器
2. 添加新的环境变量
3. 重新启动容器

### 步骤 4: 测试修复

1. 生成一张新图片
2. 检查后端日志，应该看到：
   ```
   ✓ Successfully uploaded to (R2 public domain): https://pub-xxxxxxxxx.r2.dev/generated/xxx.png
   ```
3. 验证图片在前端正常显示

## 🔧 临时解决方案（如果找不到 Public Domain）

如果你找不到 R2 public domain，可以：

### 选项A: 启用 R2 Custom Domain

1. 在 R2 存储桶设置中
2. 点击 **Custom Domains**
3. 添加自定义域名（如 `images.yourdomain.com`）
4. 设置环境变量：
   ```bash
   CLOUDFLARE_R2_PUBLIC_DOMAIN=https://images.yourdomain.com
   ```

### 选项B: 修复 R2 CORS 和权限

1. 确保 R2 存储桶的 **Public Access** 设置为 `Allowed`
2. 更新 CORS 策略为：
   ```json
   [
     {
       "AllowedOrigins": ["*"],
       "AllowedMethods": ["GET", "HEAD"],
       "AllowedHeaders": ["*"],
       "ExposeHeaders": ["*"],
       "MaxAgeSeconds": 3600
     }
   ]
   ```

## ✅ 验证修复

修复后，你应该看到：

1. **后端日志**：
   ```
   ✓ Successfully uploaded to (R2 public domain): https://pub-xxx.r2.dev/generated/xxx.png
   ```

2. **前端**：
   - 图片正常显示
   - 下载功能正常工作
   - 无 CORS 错误

3. **浏览器开发工具**：
   - 图片请求返回 200 状态码
   - 无网络错误

## 🚨 重要提醒

- 使用 **Public R2 domain** 是最简单的解决方案，因为它避免了 CORS 问题
- 确保在 Cloudflare Dashboard 中正确配置了 Public Access
- 新的 `.r2.dev` 域名比老的 `.r2.cloudflarestorage.com` 更稳定

## 📞 需要帮助？

如果你在 Cloudflare Dashboard 中找不到 Public Domain 或遇到其他问题，请：

1. 截图 R2 存储桶的 Settings 页面
2. 分享具体的错误信息
3. 告诉我存储桶的名称和配置

我会帮你进一步调试！ 