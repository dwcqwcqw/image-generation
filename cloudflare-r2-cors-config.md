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

### 3. 添加CORS配置（修正版本）

**使用这个修正的JSON配置：**

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["Content-Length", "Content-Type"],
    "MaxAgeSeconds": 3600
  }
]
```

### 4. 如果上面还有问题，试试最简配置：

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET"],
    "AllowedHeaders": ["*"]
  }
]
```

### 5. 更安全的CORS配置（推荐生产环境）

如果简单配置工作后，你可以改为更安全的配置：

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
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Accept", "Content-Type", "Origin"],
    "ExposeHeaders": ["Content-Length", "Content-Type"],
    "MaxAgeSeconds": 3600
  }
]
```

**注意**：将 `your-site.pages.dev` 和 `your-custom-domain.com` 替换为你的实际域名。

## 🔧 R2 CORS配置的常见问题

### 问题1: 字段名称错误
- ❌ `ExposedHeaders` → ✅ `ExposeHeaders`
- ❌ `AllowOrigins` → ✅ `AllowedOrigins`

### 问题2: MaxAgeSeconds限制
- R2可能对MaxAgeSeconds有限制，建议使用3600（1小时）而不是86400

### 问题3: 必需字段
- `AllowedOrigins` 和 `AllowedMethods` 是必需的
- 其他字段是可选的

## 🔧 其他可能的解决方案

### 方案A：R2自定义域名（推荐）
1. 在R2存储桶设置中添加自定义域名
2. 这样可以避免CORS问题，因为图片会从同一域名提供
3. 进入你的R2存储桶 → Settings → Custom Domains
4. 添加你的域名（如：`images.yourdomain.com`）

### 方案B：Cloudflare Transform Rules
1. 在 Cloudflare Dashboard > Rules > Transform Rules 中
2. 添加Response Header规则：
   - **When incoming requests match**: `hostname` contains `r2.cloudflarestorage.com`
   - **Then**: Add header `Access-Control-Allow-Origin` with value `*`

### 方案C：使用公开访问（最简单）
1. 在R2存储桶设置中
2. 确保存储桶的访问权限设置为Public
3. 这样可以直接访问图片而无需CORS

## 🧪 验证配置

配置完成后，请：

1. **等待5-10分钟**让配置生效
2. **清除浏览器缓存**（重要！）
3. **重新访问**你的Cloudflare Pages网站
4. **检查浏览器控制台**是否还有CORS错误
5. **测试图片显示**和下载功能

## ⚡ 立即测试方法

完成CORS配置后，可以立即测试：

1. 打开浏览器开发者工具
2. 在控制台中运行：
```javascript
fetch('你的R2图片URL')
  .then(response => console.log('Success:', response.status))
  .catch(error => console.log('Error:', error))
```

如果返回状态200且没有CORS错误，说明配置成功！

## 📞 如果还有问题

如果按照上述步骤配置后仍然有问题，请：

1. **截图具体的错误信息**（包括CORS policy页面的错误）
2. **告诉我你的R2存储桶名称**
3. **分享一个示例图片URL**
4. **检查存储桶的权限设置**是否为Public

我可以帮你进一步调试！ 