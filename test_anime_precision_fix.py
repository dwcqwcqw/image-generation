#!/usr/bin/env python3
"""
测试动漫模型精度修复
验证LayerNorm Half精度兼容性问题是否已解决
"""

import requests
import json
import time
import base64
from datetime import datetime

# RunPod API配置
RUNPOD_API_URL = "https://api.runpod.ai/v2/vllm-gguf-ggml/runsync"
RUNPOD_API_KEY = "RNPD-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # 替换为实际的API密钥

def test_anime_model():
    """测试动漫模型生成"""
    print("🎨 测试动漫模型精度修复...")
    
    # 测试参数
    test_params = {
        "input": {
            "task_type": "text-to-image",
            "prompt": "masterpiece, best quality, 1boy, handsome man, muscular, shirtless, detailed face, anime style",
            "negativePrompt": "low quality, blurry, bad anatomy",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfgScale": 6.0,
            "seed": 42,
            "numImages": 1,
            "baseModel": "anime"  # 使用动漫模型
        }
    }
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"📝 测试参数:")
    print(f"   模型: {test_params['input']['baseModel']}")
    print(f"   分辨率: {test_params['input']['width']}x{test_params['input']['height']}")
    print(f"   Steps: {test_params['input']['steps']}")
    print(f"   CFG: {test_params['input']['cfgScale']}")
    print(f"   Prompt: {test_params['input']['prompt'][:50]}...")
    
    try:
        print("\n🚀 发送请求到RunPod...")
        start_time = time.time()
        
        response = requests.post(
            RUNPOD_API_URL,
            headers=headers,
            json=test_params,
            timeout=300  # 5分钟超时
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  请求耗时: {duration:.2f}秒")
        print(f"📊 HTTP状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功!")
            
            if result.get('status') == 'COMPLETED':
                output = result.get('output', {})
                if output.get('success'):
                    images = output.get('data', [])
                    print(f"🎉 动漫模型生成成功!")
                    print(f"📸 生成图片数量: {len(images)}")
                    
                    for i, img in enumerate(images):
                        print(f"   图片 {i+1}: {img.get('url', 'No URL')}")
                        print(f"   尺寸: {img.get('width')}x{img.get('height')}")
                        print(f"   种子: {img.get('seed')}")
                    
                    print("\n✅ 动漫模型精度修复验证成功!")
                    print("💡 LayerNorm Half精度问题已解决")
                    return True
                else:
                    error = output.get('error', 'Unknown error')
                    print(f"❌ 生成失败: {error}")
                    
                    # 检查是否还有精度相关错误
                    if 'LayerNormKernelImpl' in error or 'Half' in error:
                        print("🚨 仍然存在精度兼容性问题!")
                        print("💡 需要进一步检查float32设置")
                    
                    return False
            else:
                print(f"❌ 任务状态: {result.get('status')}")
                return False
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ 请求超时 - 可能是模型加载时间较长")
        return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

def test_model_switching():
    """测试模型切换功能"""
    print("\n🔄 测试模型切换功能...")
    
    # 先测试真人模型
    print("1️⃣ 测试真人模型...")
    realistic_params = {
        "input": {
            "task_type": "text-to-image",
            "prompt": "handsome man, realistic photo",
            "width": 768,
            "height": 768,
            "steps": 20,
            "cfgScale": 4.0,
            "seed": 123,
            "numImages": 1,
            "baseModel": "realistic"
        }
    }
    
    # 再测试动漫模型
    print("2️⃣ 测试动漫模型...")
    anime_params = {
        "input": {
            "task_type": "text-to-image",
            "prompt": "anime boy, masterpiece, best quality",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfgScale": 6.0,
            "seed": 456,
            "numImages": 1,
            "baseModel": "anime"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 测试两个模型
    for i, (name, params) in enumerate([("真人模型", realistic_params), ("动漫模型", anime_params)], 1):
        print(f"\n{i}️⃣ 测试{name}...")
        try:
            response = requests.post(RUNPOD_API_URL, headers=headers, json=params, timeout=300)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'COMPLETED' and result.get('output', {}).get('success'):
                    print(f"✅ {name}生成成功!")
                else:
                    print(f"❌ {name}生成失败")
            else:
                print(f"❌ {name}请求失败: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}测试异常: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 动漫模型精度修复验证测试")
    print("=" * 60)
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 主要测试：动漫模型精度修复
    success = test_anime_model()
    
    if success:
        print("\n🎉 主要测试通过!")
        # 额外测试：模型切换
        test_model_switching()
    else:
        print("\n❌ 主要测试失败，需要进一步调试")
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")
    print("=" * 60) 