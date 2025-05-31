#!/usr/bin/env python3
"""
简单的动漫模型测试脚本
测试修复后的动漫模型生成功能
"""

import requests
import json
import time
import base64
from datetime import datetime

# RunPod API配置
RUNPOD_API_URL = "https://api.runpod.ai/v2/vllm-gguf-ggml/runsync"
RUNPOD_API_KEY = "RNPD-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # 替换为实际的API密钥

def test_anime_simple_generation():
    """测试动漫模型简单生成"""
    print("🎯 测试动漫模型简单生成...")
    
    # 最基础的测试参数
    test_payload = {
        "input": {
            "task_type": "text-to-image",
            "prompt": "a man",
            "negativePrompt": "",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfgScale": 7.0,
            "seed": 42,
            "numImages": 1,
            "baseModel": "anime",
            "lora_config": {}  # 无LoRA测试
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        print("📤 发送请求...")
        print(f"  Prompt: {test_payload['input']['prompt']}")
        print(f"  Model: {test_payload['input']['baseModel']}")
        print(f"  LoRA: None (基础模型测试)")
        
        start_time = time.time()
        response = requests.post(RUNPOD_API_URL, 
                               headers=headers, 
                               json=test_payload, 
                               timeout=300)
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API响应成功 (耗时: {response_time:.2f}s)")
            
            if result.get('status') == 'COMPLETED':
                output = result.get('output', {})
                success = output.get('success', False)
                
                if success:
                    data = output.get('data', [])
                    print(f"🎉 生成成功! 生成了 {len(data)} 张图像")
                    
                    for i, img_data in enumerate(data):
                        print(f"  图像 {i+1}:")
                        print(f"    URL: {img_data.get('url', 'N/A')}")
                        print(f"    文件名: {img_data.get('filename', 'N/A')}")
                        print(f"    模型: {img_data.get('model', 'N/A')}")
                        
                    return True
                else:
                    error_msg = output.get('error', '未知错误')
                    print(f"❌ 生成失败: {error_msg}")
                    return False
            else:
                print(f"❌ 任务状态异常: {result.get('status')}")
                return False
        else:
            print(f"❌ API请求失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_anime_with_lora():
    """测试动漫模型+LoRA生成（期望降级）"""
    print("\n🎯 测试动漫模型+LoRA生成...")
    
    test_payload = {
        "input": {
            "task_type": "text-to-image", 
            "prompt": "a handsome man, anime style",
            "negativePrompt": "low quality, blurry",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfgScale": 7.0,
            "seed": 123,
            "numImages": 1,
            "baseModel": "anime",
            "lora_config": {"gayporn": 1.0}  # 测试LoRA降级
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        print("📤 发送LoRA测试请求...")
        print(f"  LoRA: gayporn (预期会降级到基础模型)")
        
        start_time = time.time()
        response = requests.post(RUNPOD_API_URL,
                               headers=headers,
                               json=test_payload,
                               timeout=300)
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('status') == 'COMPLETED':
                output = result.get('output', {})
                success = output.get('success', False)
                
                if success:
                    data = output.get('data', [])
                    print(f"✅ LoRA测试成功! (应该是基础模型生成)")
                    print(f"🎉 生成了 {len(data)} 张图像")
                    return True
                else:
                    error_msg = output.get('error', '未知错误')
                    print(f"⚠️  LoRA测试失败: {error_msg}")
                    print("ℹ️  这可能是预期的，如果LoRA不兼容")
                    return False
            else:
                print(f"❌ LoRA测试状态异常: {result.get('status')}")
                return False
        else:
            print(f"❌ LoRA测试API失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ LoRA测试异常: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n🎯 测试错误处理...")
    
    # 测试空prompt
    test_payload = {
        "input": {
            "task_type": "text-to-image",
            "prompt": "",  # 空prompt测试
            "baseModel": "anime",
            "numImages": 1
        }
    }
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        print("📤 测试空prompt处理...")
        response = requests.post(RUNPOD_API_URL,
                               headers=headers,
                               json=test_payload,
                               timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('status') == 'COMPLETED':
                output = result.get('output', {})
                success = output.get('success', False)
                
                if success:
                    print("✅ 空prompt测试成功 (应该使用默认prompt)")
                    return True
                else:
                    print("⚠️  空prompt测试失败")
                    return False
            
    except Exception as e:
        print(f"❌ 错误处理测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🎮 动漫模型修复验证测试")
    print("=" * 60)
    print(f"🕒 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # 测试1: 基础生成
    print("📋 测试1: 基础动漫模型生成")
    result1 = test_anime_simple_generation()
    results.append(("基础生成", result1))
    
    # 测试2: LoRA降级
    print("\n📋 测试2: LoRA降级机制")
    result2 = test_anime_with_lora()
    results.append(("LoRA降级", result2))
    
    # 测试3: 错误处理
    print("\n📋 测试3: 错误处理机制")
    result3 = test_error_handling()
    results.append(("错误处理", result3))
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🏆 总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 动漫模型修复成功!")
    elif passed > 0:
        print("⚠️  部分测试通过，需要进一步检查")
    else:
        print("❌ 所有测试失败，需要进一步调试")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 