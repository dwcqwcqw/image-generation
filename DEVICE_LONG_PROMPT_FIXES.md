# Device Error & Long Prompt Fix Documentation

## 修复的问题

### 1. 动漫切换到真人模型的Device错误

**问题描述:**
```
Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

**原因分析:**
- FLUX模型使用device mapping时会分散模型组件到不同device
- 切换到其他模型时，新模型的tensor位置与之前不一致
- 导致推理时tensor不在同一device上

**解决方案:**
1. 禁用FLUX的device mapping，直接加载到单一device
2. 彻底清理之前的模型状态
3. 跳过FLUX的CPU offload以避免device冲突

**修改文件:**
- `backend/handler.py` - load_flux_model函数
- `backend/handler.py` - load_specific_model函数

### 2. 长Prompt处理仍被截断

**问题描述:**
```
The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens
```

**原因分析:**
- 动漫模型的Compel配置错误，使用了SDXL参数但模型不是SDXL
- FLUX模型的long prompt处理缺少re模块import
- 需要针对不同模型类型使用不同的长prompt策略

**解决方案:**
1. 修复动漫模型Compel参数，移除SDXL特有的配置
2. 添加missing的re模块import
3. 优化长prompt检测和处理逻辑

**修改文件:**
- `backend/handler.py` - generate_diffusers_images函数
- `backend/handler.py` - 添加re import

### 3. 前端LoRA选项优化

**问题描述:**
- 动漫模型显示"不使用LoRA"选项，但应该总是使用LoRA
- 缺少新的anal_sex LoRA选项

**解决方案:**
1. 移除动漫模型的"不使用LoRA"选项
2. 确保动漫模型总是选择一个默认LoRA
3. 添加anal_sex LoRA到真人模型选项中

**修改文件:**
- `frontend/src/components/LoRASelector.tsx`
- `backend/handler.py` - LORA_FILE_PATTERNS

## 技术细节

### Device Management Strategy
```python
# 禁用device mapping
device_mapping_enabled = False

# 直接加载到单一device
txt2img_pipe = FluxPipeline.from_pretrained(
    base_path,
    **model_kwargs
).to(device)

# 彻底清理之前的模型
if txt2img_pipe is not None:
    del txt2img_pipe
    txt2img_pipe = None
torch.cuda.empty_cache()
```

### Compel Configuration Fix
```python
# 修复前 (错误的SDXL配置)
compel = Compel(
    tokenizer=txt2img_pipe.tokenizer,
    text_encoder=txt2img_pipe.text_encoder,
    tokenizer_2=txt2img_pipe.tokenizer_2,  # ❌ 动漫模型可能没有
    text_encoder_2=txt2img_pipe.text_encoder_2,  # ❌ 动漫模型可能没有
    returned_embeddings_type="clip_mean_pooled",  # ❌ SDXL特有
    requires_pooled=[False, True]  # ❌ SDXL特有
)

# 修复后 (通用配置)
compel = Compel(
    tokenizer=txt2img_pipe.tokenizer,
    text_encoder=txt2img_pipe.text_encoder,
)
```

### Frontend LoRA Logic Fix
```typescript
// 动漫模型不允许选择null
if (baseModel === 'anime' && lora === null) return;

// 移除UI中的"不使用LoRA"选项
// {baseModel === 'anime' && (不使用LoRA选项)} - 已删除
```

## 测试建议

1. **Device切换测试:**
   - 启动真人模型生成图像
   - 切换到动漫模型生成图像
   - 再切换回真人模型
   - 确认无device错误

2. **长Prompt测试:**
   - 使用超过77 tokens的prompt
   - 测试动漫和真人模型
   - 确认无truncation警告

3. **LoRA功能测试:**
   - 确认动漫模型总是有默认LoRA
   - 测试新的anal_sex LoRA选项
   - 确认LoRA正确加载和生效

## 预期效果

- ✅ 模型切换无device错误
- ✅ 长prompt正确处理，无截断
- ✅ 动漫模型总是使用LoRA
- ✅ 新增anal_sex LoRA选项可用
- ✅ 系统更稳定，用户体验更好 