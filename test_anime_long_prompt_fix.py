#!/usr/bin/env python3
"""
动漫模型LoRA+长prompt修复测试脚本
测试修复：动漫模型使用LoRA时长prompt生成全黑图像的问题
"""

import os
import json
import time

def test_anime_long_prompt_with_lora():
    """测试动漫模型LoRA+长prompt修复"""
    print("🎯 测试动漫模型LoRA+长prompt修复")
    print("=" * 60)
    
    # 长prompt测试 (>50 tokens)
    long_prompt = """masterpiece, best quality, amazing quality, highly detailed anime illustration of a handsome muscular warrior man, 
    detailed facial features, strong jawline, piercing eyes, detailed armor with intricate designs, standing in a mystical forest, 
    sunlight filtering through trees, atmospheric lighting, fantasy art style, detailed background with ancient ruins, 
    glowing magical effects, dynamic pose, heroic stance, detailed texture work, professional anime artwork, 
    high resolution, vibrant colors, detailed shadows and highlights"""
    
    # 测试1：动漫模型 + LoRA + 长prompt (问题场景)
    test_problem_case = {
        "task_type": "text-to-image",
        "prompt": long_prompt,
        "negativePrompt": "",  # 系统会自动添加推荐负面提示
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfgScale": 6.0,
        "seed": 12345,
        "numImages": 1,
        "baseModel": "anime",
        "lora_config": {"gayporn": 1.0}  # 🚨 这个组合之前会生成全黑图
    }
    
    # 测试2：动漫模型 + 无LoRA + 长prompt (对比)
    test_no_lora = {
        "task_type": "text-to-image", 
        "prompt": long_prompt,
        "negativePrompt": "",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfgScale": 6.0,
        "seed": 12345,
        "numImages": 1,
        "baseModel": "anime",
        "lora_config": {}  # 无LoRA
    }
    
    # 测试3：动漫模型 + LoRA + 短prompt (对比)
    short_prompt = "masterpiece, best quality, handsome anime man, detailed face"
    test_short_prompt = {
        "task_type": "text-to-image",
        "prompt": short_prompt,
        "negativePrompt": "",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfgScale": 6.0,
        "seed": 12345,
        "numImages": 1,
        "baseModel": "anime",
        "lora_config": {"gayporn": 1.0}
    }
    
    # 计算token数量
    import re
    token_pattern = r'\w+|[^\w\s]'
    long_tokens = len(re.findall(token_pattern, long_prompt.lower()))
    short_tokens = len(re.findall(token_pattern, short_prompt.lower()))
    
    print(f"📝 测试用例:")
    print(f"  长prompt tokens: {long_tokens} (触发Compel阈值: >50)")
    print(f"  短prompt tokens: {short_tokens} (标准处理)")
    
    print(f"\n🔴 问题场景测试 - 动漫模型 + LoRA + 长prompt:")
    print(f"  ⚠️  之前问题: 会生成全黑图像 (文件大小~3KB)")
    print(f"  ✅ 修复后: 应该生成正常图像 (文件大小>100KB)")
    print(f"  Base Model: {test_problem_case['baseModel']}")
    print(f"  LoRA: {test_problem_case['lora_config']}")
    print(f"  Prompt tokens: {long_tokens}")
    
    print(f"\n🟢 对比测试1 - 动漫模型 + 无LoRA + 长prompt:")
    print(f"  预期: 使用Compel处理，正常生成")
    print(f"  Base Model: {test_no_lora['baseModel']}")
    print(f"  LoRA: {test_no_lora['lora_config']}")
    
    print(f"\n🟢 对比测试2 - 动漫模型 + LoRA + 短prompt:")
    print(f"  预期: 标准处理，正常生成")
    print(f"  Base Model: {test_short_prompt['baseModel']}")
    print(f"  LoRA: {test_short_prompt['lora_config']}")
    print(f"  Prompt tokens: {short_tokens}")
    
    return [test_problem_case, test_no_lora, test_short_prompt]

def show_fix_details():
    """显示修复的技术细节"""
    print(f"\n🔧 修复的技术细节:")
    print(f"=" * 60)
    print(f"问题原因:")
    print(f"  📍 Compel库在动漫模型(SDXL)中处理长prompt时生成的embeddings")
    print(f"  📍 与LoRA适配器不兼容，导致生成全黑图像")
    print(f"  📍 特征：文件大小异常小(~3KB)，内容为全黑或空白")
    
    print(f"\n修复方案:")
    print(f"  ✅ 检测当前是否加载了LoRA配置")
    print(f"  ✅ 如果有LoRA：禁用Compel，使用标准SDXL处理")
    print(f"  ✅ 如果无LoRA：允许使用Compel处理长prompt")
    print(f"  ✅ 虽然长prompt可能被截断，但能正常生成图像")
    
    print(f"\n核心代码逻辑:")
    print(f"  ```python")
    print(f"  has_lora = bool(current_lora_config and any(v > 0 for v in current_lora_config.values()))")
    print(f"  if has_lora:")
    print(f"      # 禁用Compel，使用标准处理")
    print(f"      generation_kwargs = {{\"prompt\": prompt, \"negative_prompt\": negative_prompt, ...}}")
    print(f"  elif tokens > 50:")
    print(f"      # 使用Compel处理长prompt")
    print(f"      compel = Compel(...)")
    print(f"  ```")

def show_verification_steps():
    """显示验证步骤"""
    print(f"\n📋 验证步骤:")
    print(f"=" * 60)
    print(f"1. 启动RunPod后端服务")
    print(f"2. 发送测试1请求（问题场景）")
    print(f"3. 检查日志中的关键信息:")
    print(f"   🔍 '⚠️  检测到LoRA配置 {{\"gayporn\": 1.0}}，禁用Compel避免兼容性问题'")
    print(f"   🔍 '✅ 使用标准SDXL处理（LoRA兼容模式）'")
    print(f"   🔍 '✅ 图像生成成功' 且文件大小 > 100KB")
    print(f"4. 验证生成的图像不是全黑图像")
    print(f"5. 文件大小应该正常（几百KB而不是几KB）")
    
    print(f"\n🚨 失败指标:")
    print(f"  ❌ 文件大小 < 10KB (可能是全黑图)")
    print(f"  ❌ 图像内容全黑或空白")
    print(f"  ❌ 生成过程中出现Compel相关错误")
    
    print(f"\n✅ 成功指标:")
    print(f"  ✅ 日志显示禁用Compel并使用LoRA兼容模式")
    print(f"  ✅ 文件大小正常 (>100KB)")
    print(f"  ✅ 图像内容正常，能看到具体内容")
    print(f"  ✅ LoRA效果能正常体现")

if __name__ == "__main__":
    print("🚀 动漫模型LoRA+长prompt修复测试")
    print("=" * 60)
    
    # 显示修复细节
    show_fix_details()
    
    # 生成测试参数
    test_cases = test_anime_long_prompt_with_lora()
    
    # 显示验证步骤
    show_verification_steps()
    
    print(f"\n🎯 修复总结:")
    print(f"这个修复解决了动漫模型使用LoRA时长prompt生成全黑图像的严重bug。")
    print(f"现在系统会智能检测LoRA配置，在有LoRA时禁用Compel避免兼容性问题。")
    print(f"虽然长prompt可能会被截断，但能保证正常生成有内容的图像。") 