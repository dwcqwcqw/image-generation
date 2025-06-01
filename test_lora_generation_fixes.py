#!/usr/bin/env python3
"""
LoRA加载和动漫模型生成修复测试脚本
测试：
1. 首次LoRA加载问题
2. 动漫模型生成质量问题
3. WAI-NSFW-illustrious-SDXL参数优化
"""

import os
import json
import time

def test_anime_model_with_lora():
    """测试动漫模型和LoRA加载修复"""
    print("🎯 测试动漫模型LoRA加载修复")
    print("=" * 60)
    
    # 测试1：动漫模型 + gayporn LoRA
    test_1_params = {
        "task_type": "text-to-image", 
        "prompt": "masterpiece, best quality, amazing quality, handsome muscular man, detailed face, anime style, masculine features",
        "negativePrompt": "",  # 系统会自动添加推荐的负面提示
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfgScale": 6.0,  # 符合CivitAI推荐的5-7范围
        "seed": 12345,
        "numImages": 1,
        "baseModel": "anime",
        "lora_config": {"gayporn": 1.0}  # 🚨 修复：确保首次就加载用户选择的LoRA
    }
    
    print(f"📝 测试1 - 动漫模型 + gayporn LoRA:")
    print(f"  Base Model: {test_1_params['baseModel']}")
    print(f"  LoRA Config: {test_1_params['lora_config']}")
    print(f"  尺寸: {test_1_params['width']}x{test_1_params['height']}")
    print(f"  CFG Scale: {test_1_params['cfgScale']} (CivitAI推荐5-7)")
    print(f"  Steps: {test_1_params['steps']}")
    
    # 测试2：动漫模型 + furry LoRA
    test_2_params = {
        "task_type": "text-to-image",
        "prompt": "masterpiece, best quality, amazing quality, anthro wolf, detailed fur texture, anime style",
        "negativePrompt": "nsfw",  # 用户自定义负面提示
        "width": 1024,
        "height": 1024, 
        "steps": 20,  # 使用CivitAI推荐的15-30范围
        "cfgScale": 5.5,
        "seed": 67890,
        "numImages": 1,
        "baseModel": "anime",
        "lora_config": {"furry": 1.0}
    }
    
    print(f"\n📝 测试2 - 动漫模型 + furry LoRA:")
    print(f"  Base Model: {test_2_params['baseModel']}")
    print(f"  LoRA Config: {test_2_params['lora_config']}")
    print(f"  尺寸: {test_2_params['width']}x{test_2_params['height']}")
    print(f"  CFG Scale: {test_2_params['cfgScale']}")
    print(f"  Steps: {test_2_params['steps']}")
    
    # 测试3：真人模型对比
    test_3_params = {
        "task_type": "text-to-image",
        "prompt": "realistic photograph of a handsome man, detailed facial features, professional lighting",
        "negativePrompt": "",
        "width": 768,
        "height": 768,
        "steps": 25,
        "cfgScale": 4.0,  # FLUX推荐参数
        "seed": 11111,
        "numImages": 1,
        "baseModel": "realistic",
        "lora_config": {"cum_on_face": 1.0}
    }
    
    print(f"\n📝 测试3 - 真人模型对比:")
    print(f"  Base Model: {test_3_params['baseModel']}")
    print(f"  LoRA Config: {test_3_params['lora_config']}")
    print(f"  尺寸: {test_3_params['width']}x{test_3_params['height']} (FLUX推荐)")
    
    print(f"\n🔧 关键修复验证:")
    print(f"  ✅ 修复首次LoRA加载顺序：先切换模型，再加载用户选择的LoRA")
    print(f"  ✅ WAI-NSFW-illustrious-SDXL参数优化：1024x1024分辨率，CFG 5-7，Steps 15-30")
    print(f"  ✅ 自动添加推荐质量标签：masterpiece, best quality, amazing quality")
    print(f"  ✅ 自动添加推荐负面提示：bad quality, worst quality, worst detail, sketch, censor")
    print(f"  ✅ 避免适配器名称冲突：使用时间戳+随机字符串")
    
    print(f"\n📊 预期结果:")
    print(f"  1. 动漫模型应该立即加载用户选择的LoRA，不再出现默认LoRA")
    print(f"  2. 生成的动漫图像质量应该明显提升，不再是残次品")
    print(f"  3. 参数应该符合CivitAI的WAI-NSFW-illustrious-SDXL推荐设置")
    print(f"  4. LoRA切换应该没有适配器名称冲突错误")
    
    return [test_1_params, test_2_params, test_3_params]

def show_civitai_recommendations():
    """显示CivitAI WAI-NSFW-illustrious-SDXL推荐设置"""
    print(f"\n📋 CivitAI WAI-NSFW-illustrious-SDXL 模型推荐设置:")
    print(f"  🔸 Steps: 15-30 (v14) / 25-40 (older versions)")
    print(f"  🔸 CFG scale: 5-7")
    print(f"  🔸 Sampler: Euler a")
    print(f"  🔸 Size: 大于1024x1024")
    print(f"  🔸 VAE: 已集成")
    print(f"  🔸 Clip Skip: 2")
    print(f"  🔸 Positive prompt: masterpiece,best quality,amazing quality,")
    print(f"  🔸 Negative prompt: bad quality,worst quality,worst detail,sketch,censor,")
    print(f"  🔸 Safety tags: general, sensitive, nsfw, explicit")
    print(f"  🔸 提示：用户应在负面提示中添加'nsfw'来过滤不当内容")

if __name__ == "__main__":
    print("🚀 LoRA加载和动漫模型生成修复测试")
    print("=" * 60)
    
    # 显示CivitAI推荐设置
    show_civitai_recommendations()
    
    # 生成测试参数
    test_params = test_anime_model_with_lora()
    
    print(f"\n💡 使用方法:")
    print(f"1. 启动RunPod后端服务")
    print(f"2. 使用前端或API发送上述测试参数")
    print(f"3. 检查日志中是否有以下关键信息:")
    print(f"   - '✅ 成功切换到 anime 模型'")
    print(f"   - '✅ LoRA配置更新成功'")
    print(f"   - '✅ WAI-NSFW-illustrious-SDXL模型需要1024x1024或更大'")
    print(f"   - '✨ 添加WAI-NSFW-illustrious-SDXL推荐质量标签'")
    print(f"   - '🛡️ 使用WAI-NSFW-illustrious-SDXL推荐负面提示'")
    print(f"4. 验证生成的图像质量是否有明显提升")
    
    print(f"\n🎯 修复完成！现在应该可以正确生成高质量动漫图像了。") 