# Cloudflare R2 CORS 配置指南

## 🚨 重要：需要你手动配置R2存储的CORS设置

图片无法显示的根本原因是R2存储没有正确的CORS配置。你需要在Cloudflare Dashboard中进行以下操作：

## 📋 配置步骤

### 1. 进入R2存储管理
1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com)
2. 选择你的账户
3. 点击左侧菜单的 **"R2 Object Storage"**
4. 找到你的存储桶（应该是 `image-generation` 或类似名称）

### 2. 配置CORS规则
1. 点击你的存储桶名称进入详情页
2. 点击 **"Settings"** 标签页
3. 找到 **"CORS policy"** 部分
4. 点击 **"Add CORS policy"** 或 **"Edit"**

### 3. 添加CORS配置

复制以下JSON配置并粘贴：

```json
[
  {
    "AllowedOrigins": [
      "*"
    ],
    "AllowedMethods": [
      "GET",
      "HEAD",
      "POST",
      "PUT",
      "DELETE"
    ],
    "AllowedHeaders": [
      "*"
    ],
    "ExposedHeaders": [
      "ETag",
      "Content-Length",
      "Content-Type"
    ],
    "MaxAgeSeconds": 3600
  }
]
```

### 4. 更安全的CORS配置（推荐）

如果你想要更安全的配置，请替换为：

```json
[
  {
    "AllowedOrigins": [
      "https://your-site.pages.dev",
      "https://your-custom-domain.com",
      "http://localhost:3000",
      "http://localhost:3001",
      "http://localhost:3002"
    ],
    "AllowedMethods": [
      "GET",
      "HEAD"
    ],
    "AllowedHeaders": [
      "Accept",
      "Content-Type",
      "Origin",
      "User-Agent"
    ],
    "ExposedHeaders": [
      "Content-Length",
      "Content-Type"
    ],
    "MaxAgeSeconds": 86400
  }
]
```

**注意**：将 `your-site.pages.dev` 和 `your-custom-domain.com` 替换为你的实际域名。

## 🔧 其他可能的解决方案

### 方案A：R2自定义域名
1. 在R2存储桶设置中添加自定义域名
2. 这样可以避免CORS问题，因为图片会从同一域名提供

### 方案B：Cloudflare Transform Rules
1. 在 Cloudflare Dashboard > Rules > Transform Rules 中
2. 添加Response Header规则：
   - **When incoming requests match**: `hostname` contains `r2.cloudflarestorage.com`
   - **Then**: Add header `Access-Control-Allow-Origin` with value `*`

## 🧪 验证配置

配置完成后，请：

1. 等待5-10分钟让配置生效
2. 清除浏览器缓存
3. 重新访问你的Cloudflare Pages网站
4. 检查浏览器控制台是否还有CORS错误

## ⚡ 临时解决方案

在配置CORS期间，你可以：

1. 使用开发环境（localhost）进行测试
2. 或者在浏览器中禁用CORS检查（仅用于测试）：
   ```bash
   # Chrome (仅用于测试，不安全)
   chrome --disable-web-security --user-data-dir="/tmp/chrome_dev_session"
   ```

## 📞 如果还有问题

如果按照上述步骤配置后仍然有问题，请：

1. 检查网络面板中的具体错误信息
2. 确认R2存储桶的访问权限设置
3. 检查Cloudflare Pages的Functions是否启用
4. 验证环境变量是否正确设置

配置完成后，图片应该能正常显示和下载了！ 