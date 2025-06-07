#!/usr/bin/env python3
"""
快速API换脸测试脚本
验证API密钥配置和参数格式是否正确
"""

import os
import requests

# RunPod API配置
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
FACE_SWAP_ENDPOINT = os.getenv("FACE_SWAP_ENDPOINT", "https://api.runpod.ai/v2/sbta9w9yx2cc1e")

def test_api_configuration():
    """测试API配置"""
    print("=== 快速API换脸配置测试 ===")
    print(f"🔑 API密钥: {RUNPOD_API_KEY[:10]}...{RUNPOD_API_KEY[-4:]}" if RUNPOD_API_KEY else "❌ 未设置")
    print(f"🌐 API端点: {FACE_SWAP_ENDPOINT}")
    
    if not RUNPOD_API_KEY:
        print("❌ API密钥未设置，请设置RUNPOD_API_KEY环境变量")
        return False
    
    # 测试API端点连通性
    try:
        print("🔍 测试API端点连通性...")
        response = requests.get(FACE_SWAP_ENDPOINT, timeout=10)
        print(f"✅ API端点响应状态: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ API端点测试失败: {e}")
        return False

def test_task_submission():
    """测试任务提交（使用正确的参数格式）"""
    print("\n🧪 测试任务提交格式...")
    
    # 使用正确的 single-image 格式
    test_payload = {
        "input": {
            "process_type": "single-image",  # 注意：使用 single-image 而不是 single_image
            "source_file": "https://example.com/source.jpg",
            "target_file": "https://example.com/target.jpg", 
            "options": {
                "mouth_mask": True,
                "use_face_enhancer": True
            }
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        print("📤 提交测试任务...")
        response = requests.post(
            f"{FACE_SWAP_ENDPOINT}/run",
            json=test_payload,
            headers=headers,
            timeout=30
        )
        
        print(f"📋 响应状态码: {response.status_code}")
        print(f"📋 响应内容: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if 'id' in result:
                print(f"✅ 任务提交成功，ID: {result['id']}")
                print("✅ API密钥认证通过!")
                print("✅ 参数格式正确!")
                return True
            else:
                print(f"⚠️ 响应格式异常: {result}")
                return False
        else:
            print(f"❌ 任务提交失败: {response.status_code}")
            
            # 分析具体错误
            if response.status_code == 401:
                print("   原因：API密钥认证失败")
            elif response.status_code == 400:
                print("   原因：参数格式错误")
                print("   检查 process_type 是否为 'single-image'")
            elif response.status_code == 404:
                print("   原因：API端点不存在")
            
            return False
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def main():
    """主测试函数"""
    # 测试1：配置检查
    config_ok = test_api_configuration()
    
    if not config_ok:
        print("\n❌ 基础配置检查失败")
        return
    
    # 测试2：任务提交
    submit_ok = test_task_submission()
    
    if submit_ok:
        print("\n🎉 所有测试通过!")
        print("💡 API换脸配置正确，可以正常使用")
        print("💡 之前的 'single_image' 错误已修复为 'single-image'")
    else:
        print("\n❌ 任务提交测试失败")
        print("💡 请检查API密钥和参数格式")

if __name__ == "__main__":
    main() 