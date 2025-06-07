# API密钥设置说明

为了使用外部换脸API功能，您需要正确设置RunPod API密钥。

## 获取API密钥

1. 登录RunPod控制台：https://www.runpod.io/
2. 进入"Settings" → "API Keys"
3. 生成或复制您的API密钥
4. 密钥格式类似：`rpa_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

## 设置方法

### 方法1：环境变量（推荐）

在RunPod部署时设置环境变量：

```bash
RUNPOD_API_KEY=您的实际API密钥
FACE_SWAP_ENDPOINT=https://api.runpod.ai/v2/sbta9w9yx2cc1e
```

### 方法2：修改代码

在`backend/start_debug.py`中找到以下行：

```python
preset_api_key = "your_runpod_api_key_here"
```

将其替换为您的实际API密钥。

## 验证设置

运行诊断工具验证配置：

```bash
python fix_api_auth.py
```

期望输出：
```
✅ API配置正常，认证成功
```

## 安全提醒

- 🚨 请勿将API密钥提交到公共代码仓库
- 🔒 使用环境变量或安全的配置管理方式
- 🔄 定期轮换API密钥以提高安全性 