# 图片显示和下载问题测试指南

## 问题概述
- 前端无法显示生成的图片（CORS错误）
- 图片下载失败（400错误）
- R2 URL格式不正确

## 修复内容

### 1. 修复了R2 URL格式
**之前（错误）**:
```
https://c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/image-generation/generated/xxx.png
```

**现在（正确）**:
```
https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/xxx.png
```

### 2. 添加了自定义域名支持
可以通过设置环境变量 `CLOUDFLARE_R2_PUBLIC_DOMAIN` 来使用自定义域名，避免CORS问题。

## 测试步骤

### 第一步：重新部署Backend
1. 确保所有代码更改已保存
2. 重新构建并部署Docker镜像到RunPod
3. 等待部署完成

### 第二步：配置CORS（必须）
按照 `CLOUDFLARE_R2_CORS_CONFIG.md` 文件中的说明配置Cloudflare R2存储桶的CORS策略。

### 第三步：测试图片生成
1. 打开前端应用: https://34237b51.image-generation-dfn.pages.dev/
2. 生成一张图片
3. 观察生成的图片URL格式是否正确

### 第四步：测试图片显示
1. 检查生成的图片是否在前端正常显示
2. 查看浏览器开发者工具的Network标签
3. 确认图片请求返回200状态码，没有CORS错误

### 第五步：测试图片下载
1. 点击图片的下载按钮
2. 确认图片能正常下载到本地
3. 检验下载的图片文件完整性

## 调试信息

### 检查RunPod日志
在RunPod控制台中查看日志，应该看到类似信息：
```
✓ Successfully uploaded to (standard R2): https://image-generation.c7c141ce43d175e60601edc46d904553.r2.cloudflarestorage.com/generated/xxx.png
```

### 测试URL格式
手动访问一个生成的图片URL，应该：
1. 返回图片内容
2. 包含正确的CORS头部
3. 状态码为200

### 浏览器开发者工具检查
1. 打开浏览器开发者工具
2. 查看Console标签，确认没有CORS错误
3. 查看Network标签，确认图片请求成功

## 预期结果

修复成功后应该：
- ✅ 图片在前端正常显示
- ✅ 图片可以正常下载
- ✅ 没有CORS错误
- ✅ URL格式正确
- ✅ 浏览器开发者工具中没有网络错误

## 故障排除

### 如果仍有CORS错误
1. 确认已按照 `CLOUDFLARE_R2_CORS_CONFIG.md` 配置CORS策略
2. 等待CORS策略生效（可能需要几分钟）
3. 清除浏览器缓存并重试

### 如果URL格式仍然错误
1. 确认backend代码已更新
2. 重新部署RunPod容器
3. 检查RunPod日志中的上传成功信息

### 如果图片无法访问
1. 检查R2存储桶的公共访问设置
2. 确认图片确实已上传到R2
3. 在Cloudflare Dashboard中验证存储桶内容

## 联系支持
如果问题持续存在，请提供：
1. 生成的图片URL示例
2. 浏览器开发者工具的错误截图
3. RunPod部署日志 