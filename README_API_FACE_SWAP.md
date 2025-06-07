# 外部API换脸功能集成指南

## 概述

本项目已集成外部RunPod Serverless API进行换脸处理，避免了本地CUDA兼容性问题，提供了更稳定和高效的换脸功能。

## ✅ 最新更新

### 2025-06-07 修复

- ✅ **支持URL和Base64两种结果格式**: API现在可以返回图像URL或Base64编码数据
- ✅ **自动格式检测**: 系统自动检测API返回的是URL还是Base64格式
- ✅ **智能处理**: URL格式直接下载，Base64格式直接解码
- ✅ **增强调试**: 添加详细的结果类型和内容日志
- ✅ **错误处理**: 改进错误处理和回退机制

### 参数格式修复

- ✅ **process_type参数**: 从 `"single_image"` 修正为 `"single-image"`
- ✅ **API认证**: 修复401认证失败问题，支持自动环境变量配置

## 特性对比

| 特性 | 本地换脸 | 外部API换脸 |
|------|----------|-------------|
| 依赖复杂度 | 高 (CUDA、InsightFace、ONNX) | 低 (仅HTTP请求) |
| 部署难度 | 困难 (环境兼容性问题) | 简单 |
| 处理速度 | 慢 (CPU回退) | 快 (专用GPU) |
| 稳定性 | 中等 | 高 |
| 内存占用 | 高 | 低 |
| 结果格式 | PIL图像对象 | **URL/Base64 (自动检测)** |

## 配置指南

### 1. RunPod Serverless部署配置

如果您使用RunPod Serverless部署，有两种方法配置API密钥：

#### 方法1：通过runpod.toml配置文件（推荐）

项目根目录的`runpod.toml`文件已经预配置了环境变量：

```toml
[runtime.env]
# RunPod API配置 - 外部换脸API
RUNPOD_API_KEY = "${RUNPOD_API_KEY}"
FACE_SWAP_ENDPOINT = "${FACE_SWAP_ENDPOINT}"
```

#### 方法2：自动设置（启动时检查）

启动脚本`backend/start_debug.py`会自动检查并设置API密钥：

```python
def check_and_set_env_vars():
    """检查和设置关键环境变量"""
    runpod_api_key = os.getenv("RUNPOD_API_KEY", "")
    if not runpod_api_key:
        preset_api_key = "your_runpod_api_key_here"  # 请在部署时设置正确的API密钥
        os.environ["RUNPOD_API_KEY"] = preset_api_key
```

### 2. 本地开发环境配置

```bash
# 设置环境变量
export RUNPOD_API_KEY="your_runpod_api_key_here"
export FACE_SWAP_ENDPOINT="https://api.runpod.ai/v2/sbta9w9yx2cc1e"
```

## API调用格式

### 请求格式

```json
{
  "input": {
    "process_type": "single-image",
    "source_file": "https://example.com/source.jpg",
    "target_file": "https://example.com/target.jpg",
    "options": {
      "mouth_mask": true,
      "use_face_enhancer": true
    }
  }
}
```

### 响应格式

系统现在支持两种响应格式：

#### 格式1：URL格式
```json
{
  "status": "COMPLETED",
  "output": {
    "result": "https://example.com/result.jpg"
  }
}
```

#### 格式2：Base64格式
```json
{
  "status": "COMPLETED",
  "output": {
    "result": "iVBORw0KGgoAAAANSUhEUgAAAA..."
  }
}
```

## 系统架构

```
用户上传参考图像 + 输入提示词
         ↓
    文生图生成初始图像
         ↓
    上传图像到R2存储获取URL
         ↓
    调用外部RunPod换脸API
         ↓
    【新增】自动检测结果格式
    ├─ URL格式 → 直接下载图像
    └─ Base64格式 → 解码图像数据
         ↓
    上传最终结果到R2存储
         ↓
    返回公共访问URL
```

## 测试和验证

### 运行格式检测测试

```bash
python test_url_format.py
```

期望输出：
```
🧪 API换脸结果格式处理测试
==================================================
=== URL格式检测测试 ===
✅ HTTPS URL: True (期望: True)
✅ HTTP URL: True (期望: True)
✅ Base64数据: False (期望: False)

=== URL下载测试 ===
✅ 下载成功，状态码: 200
✅ 图像解析成功: (239, 178) 像素

=== Base64处理测试 ===
✅ Base64解码成功: (1, 1) 像素

🎯 总体测试结果: ✅ 全部通过
```

### 运行API认证测试

```bash
python fix_api_auth.py
```

期望输出：
```
=== API配置检查 ===
🔑 API密钥长度: 45 字符
✅ API认证测试通过
```

## 故障排除

### 常见问题

1. **401认证失败**
   - 检查 `RUNPOD_API_KEY` 环境变量是否正确设置
   - 运行 `python fix_api_auth.py` 进行诊断

2. **Base64解码失败**
   - ✅ **已修复**: 系统现在自动检测URL格式并直接下载
   - 支持两种格式：URL和Base64

3. **process_type错误**
   - ✅ **已修复**: 使用正确的 `"single-image"` 格式

4. **任务提交失败**
   - 检查API端点是否正确
   - 验证网络连接和防火墙设置

### 调试日志示例

成功的API调用会显示：
```
📤 提交换脸任务...
✅ 任务已提交，ID: abc123-def456-ghi789
🔄 查询任务状态 (1/60)...
📋 任务状态: IN_QUEUE
📋 任务状态: IN_PROGRESS  
📋 任务状态: COMPLETED
✅ 换脸API调用成功
🔍 结果类型: <class 'str'>
🔍 结果内容: https://example.com/result.jpg
📥 下载换脸结果图像: https://example.com/result.jpg
✅ API换脸成功完成 (URL)
```

## 安全注意事项

- 🚨 请勿将API密钥提交到公共代码仓库
- 🔒 使用环境变量或安全的配置管理方式
- 🔄 定期轮换API密钥以提高安全性
- 🌐 确保R2存储桶的访问权限配置正确

## 技术支持

如需技术支持，请提供：
1. 完整的错误日志
2. API配置信息（隐藏敏感信息）
3. 重现步骤
4. 环境信息（Python版本、依赖库版本等） 