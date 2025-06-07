# 外部API换脸功能集成说明

## 概述

本系统已升级为使用外部RunPod Serverless API进行换脸处理，替代之前的本地换脸功能。这种方式具有以下优势：

- ✅ **无需本地CUDA依赖**：避免复杂的ONNX Runtime GPU配置问题
- ✅ **高性能处理**：专用的换脸服务器提供更快速度和更高质量
- ✅ **简化部署**：减少本地依赖和模型文件需求
- ✅ **稳定性提升**：专门优化的换脸服务，避免版本兼容性问题

## 系统架构

```
用户上传参考图像 + 输入提示词
          ↓
    文生图生成初始图像
          ↓
    上传图像到R2存储（获得临时URL）
          ↓
    调用外部RunPod换脸API
          ↓
    接收Base64编码的换脸结果
          ↓
    上传最终结果到R2并返回给用户
```

## 环境变量配置

在RunPod部署时，需要设置以下环境变量：

```bash
# RunPod换脸API配置
RUNPOD_API_KEY=your_runpod_api_key_here
FACE_SWAP_ENDPOINT=https://api.runpod.ai/v2/sbta9w9yx2cc1e
```

### 获取API密钥

1. 登录RunPod控制台
2. 进入API设置页面
3. 生成或复制API密钥
4. 将密钥设置为环境变量`RUNPOD_API_KEY`

## 代码修改详情

### 主要文件修改

#### `backend/handler.py`

1. **新增API换脸函数**:
   - `upload_image_to_temp_url()` - 上传图像到临时URL
   - `call_face_swap_api()` - 调用外部换脸API
   - `process_face_swap_api_pipeline()` - API换脸处理管道

2. **修改换脸流程**:
   - `_process_realistic_with_face_swap()` 函数现在使用API而非本地处理
   - 移除对本地InsightFace和GFPGAN依赖的强制要求

3. **环境变量配置**:
   ```python
   RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
   FACE_SWAP_ENDPOINT = os.getenv("FACE_SWAP_ENDPOINT", "https://api.runpod.ai/v2/sbta9w9yx2cc1e")
   ```

## API调用流程

### 1. 任务提交

```python
submit_payload = {
    "input": {
        "process_type": "single_image",
        "source_file": source_image_url,  # 用户上传的参考图像
        "target_file": target_image_url,  # 生成的图像
        "options": {
            "mouth_mask": True,
            "use_face_enhancer": True
        }
    }
}
```

### 2. 状态轮询

系统会自动轮询任务状态，最多等待5分钟：

- `IN_QUEUE` - 任务排队中
- `IN_PROGRESS` - 任务处理中
- `COMPLETED` - 任务完成
- `FAILED` - 任务失败

### 3. 结果处理

API返回Base64编码的图像，系统会：
1. 解码Base64数据
2. 转换为PIL Image对象
3. 上传到R2存储
4. 返回最终URL给用户

## 测试验证

运行测试脚本验证API功能：

```bash
# 设置环境变量
export RUNPOD_API_KEY="your_api_key"
export FACE_SWAP_ENDPOINT="https://api.runpod.ai/v2/sbta9w9yx2cc1e"

# 运行测试
python3 test_api_face_swap.py
```

期望输出：
```
=== 外部API换脸功能测试 ===
🔑 API密钥: rpa_YT0BFB...N4ZQ1tdxlb
🌐 API端点: https://api.runpod.ai/v2/sbta9w9yx2cc1e
🔍 测试API端点可用性...
✅ API端点可达
✅ API换脸功能测试通过!
💡 系统已准备好使用外部API进行换脸处理
```

## 错误处理

系统包含完善的错误处理机制：

1. **API不可用时**：返回原始生成图像
2. **网络超时**：自动重试和回退
3. **任务失败**：记录详细错误信息并回退到原始图像
4. **Base64解码失败**：安全回退到原始图像

## 性能对比

| 指标 | 本地换脸 | API换脸 |
|------|----------|---------|
| 依赖复杂度 | 高（CUDA、ONNX、InsightFace） | 低（仅HTTP请求） |
| 部署难度 | 困难 | 简单 |
| 处理速度 | 慢（CPU回退） | 快（专用GPU） |
| 内存占用 | 高（模型加载） | 低（无本地模型） |
| 稳定性 | 中等（版本兼容） | 高（专门优化） |

## 兼容性说明

- 保留了原有的本地换脸代码作为备用
- 前端调用接口保持不变
- 结果格式完全兼容
- 添加了`faceSwapMethod: 'external_api'`字段用于标识

## 部署建议

1. **优先使用API换脸**：新部署建议直接使用API版本
2. **环境变量必须设置**：确保`RUNPOD_API_KEY`正确配置
3. **网络连接稳定**：确保RunPod服务器到API端点的网络连接良好
4. **监控API额度**：注意RunPod API的使用额度和计费

## 后续优化

1. **缓存机制**：对相同参考图像的重复请求进行缓存
2. **批处理支持**：一次性处理多张图像
3. **质量参数调优**：根据用户需求调整换脸质量参数
4. **备用端点**：配置多个API端点实现高可用 