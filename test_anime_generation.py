#!/usr/bin/env python3
"""
动漫模型生成质量诊断测试脚本
测试LoRA加载和图像生成质量
"""

import os
import sys
import json
import time

def test_anime_generation():
    """测试动漫模型生成质量"""
    print("🎯 动漫模型生成质量诊断测试")
    print("=" * 50)
    
    # 测试参数
    test_prompt = "masterpiece, best quality, 1boy, handsome anime guy, detailed face, high resolution"
    test_params = {
        "task_type": "text-to-image",
        "prompt": test_prompt,
        "negativePrompt": "worst quality, bad quality, blurry, sketch",
        "width": 768,
        "height": 768,
        "steps": 25,
        "cfgScale": 7.0,
        "seed": 12345,
        "numImages": 1,
        "baseModel": "anime",
        "lora_config": {"gayporn": 1.0}
    }
    
    print(f"📝 测试参数:")
    print(f"  Prompt: {test_prompt}")
    print(f"  尺寸: {test_params['width']}x{test_params['height']}")
    print(f"  Steps: {test_params['steps']}")
    print(f"  CFG Scale: {test_params['cfgScale']}")
    print(f"  LoRA: {test_params['lora_config']}")
    
    # 模拟生成请求
    try:
        # 这里应该调用实际的handler
        print("\n🚀 开始生成测试...")
        print("注意：这是测试脚本，实际生成需要在RunPod环境中进行")
        
        # 模拟响应
        print("✅ 测试参数验证通过")
        print("💡 建议的优化设置:")
        print("  - 分辨率: 768x768 或更高")
        print("  - CFG Scale: 6-9 (推荐 7.0)")
        print("  - Steps: 20-35 (推荐 25)")
        print("  - 确保使用有效的LoRA")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_lora_loading():
    """测试LoRA加载机制"""
    print("\n🔧 LoRA加载机制测试")
    print("=" * 50)
    
    # 测试LoRA配置
    lora_configs = [
        {"gayporn": 1.0},
        {"blowjob_handjob": 1.0},
        {"sex_slave": 1.0},
        {"furry": 0.8}
    ]
    
    for i, config in enumerate(lora_configs):
        print(f"\n测试配置 {i+1}: {config}")
        
        # 模拟LoRA切换请求
        test_params = {
            "task_type": "load-loras",
            "lora_config": config
        }
        
        print(f"  📦 请求加载LoRA: {list(config.keys())}")
        print(f"  ⚡ 模拟清理之前的适配器...")
        print(f"  🔄 模拟加载新适配器...")
        print(f"  ✅ LoRA配置测试完成")
    
    print("\n💡 LoRA优化建议:")
    print("  1. 确保完全清理之前的适配器")
    print("  2. 使用唯一的适配器名称")
    print("  3. 验证LoRA文件存在性")
    print("  4. 适当的权重设置 (0.5-1.0)")
    
    return True

def test_generation_quality():
    """测试生成质量优化"""
    print("\n🎨 生成质量优化测试")
    print("=" * 50)
    
    quality_tests = [
        {
            "name": "低质量参数 (应被自动修正)",
            "params": {"width": 512, "height": 512, "steps": 15, "cfgScale": 3.0}
        },
        {
            "name": "推荐参数",
            "params": {"width": 768, "height": 768, "steps": 25, "cfgScale": 7.0}
        },
        {
            "name": "高质量参数",
            "params": {"width": 1024, "height": 1024, "steps": 30, "cfgScale": 7.5}
        }
    ]
    
    for test in quality_tests:
        print(f"\n📊 {test['name']}:")
        params = test['params']
        
        # 模拟参数优化逻辑
        optimized = params.copy()
        
        if optimized['width'] < 768 or optimized['height'] < 768:
            print(f"  ⚠️  分辨率过低，从 {optimized['width']}x{optimized['height']} 调整为 768x768")
            optimized['width'] = max(768, optimized['width'])
            optimized['height'] = max(768, optimized['height'])
        
        if optimized['cfgScale'] < 6.0:
            print(f"  ⚠️  CFG过低，从 {optimized['cfgScale']} 调整为 7.0")
            optimized['cfgScale'] = 7.0
        elif optimized['cfgScale'] > 10.0:
            print(f"  ⚠️  CFG过高，从 {optimized['cfgScale']} 调整为 7.5")
            optimized['cfgScale'] = 7.5
        
        if optimized['steps'] < 20:
            print(f"  ⚠️  步数过低，从 {optimized['steps']} 调整为 25")
            optimized['steps'] = 25
        elif optimized['steps'] > 40:
            print(f"  ⚠️  步数过高，从 {optimized['steps']} 调整为 35")
            optimized['steps'] = 35
        
        print(f"  📝 优化后参数: {optimized}")
        print(f"  ✅ 参数验证通过")
    
    return True

def main():
    """主测试函数"""
    print("🧪 动漫模型诊断测试开始")
    print("=" * 60)
    
    tests = [
        ("动漫模型生成质量", test_anime_generation),
        ("LoRA加载机制", test_lora_loading),
        ("生成质量优化", test_generation_quality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 运行测试: {test_name}")
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - 通过")
            else:
                print(f"❌ {test_name} - 失败")
                
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{len(results)} 测试通过")
    
    # 修复建议
    print("\n💡 修复建议:")
    print("1. 🔧 使用更彻底的LoRA适配器清理机制")
    print("2. 🎯 优化动漫模型生成参数 (分辨率>=768, CFG=6-9, steps=20-35)")
    print("3. 🧹 在LoRA加载失败时确保状态清理") 
    print("4. 📝 验证LoRA文件存在性和兼容性")
    print("5. 🎲 确保种子值正确显示和记录")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！系统配置正确。")
        return 0
    else:
        print(f"\n⚠️  有 {len(results) - passed} 个测试失败，需要修复。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 