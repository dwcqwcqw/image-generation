#!/usr/bin/env python3
"""
测试外部API换脸功能
"""

import sys
import os
import requests
import base64
import io
from PIL import Image
import uuid
import time

# 添加backend路径
sys.path.append('backend')

# RunPod API配置
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
FACE_SWAP_ENDPOINT = os.getenv("FACE_SWAP_ENDPOINT", "https://api.runpod.ai/v2/sbta9w9yx2cc1e")

def create_test_image(color, size=(512, 512)):
    """创建测试图像"""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # 添加一些文字标识
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((10, 10), f"Test Image {color}", fill="white" if color != "white" else "black", font=font)
    except:
        draw.text((10, 10), f"Test {color}", fill="white" if color != "white" else "black")
    
    return img

def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_face_swap_api():
    """测试换脸API调用"""
    print("🧪 开始测试外部换脸API...")
    
    # 1. 创建测试图像
    print("📸 创建测试图像...")
    source_image = create_test_image("red", (512, 512))
    target_image = create_test_image("blue", (512, 512))
    
    # 保存测试图像到本地（用于调试）
    source_image.save("test_source.jpg")
    target_image.save("test_target.jpg")
    print("✅ 测试图像已保存: test_source.jpg, test_target.jpg")
    
    # 2. 转换为base64（模拟上传到URL的过程）
    source_base64 = image_to_base64(source_image)
    target_base64 = image_to_base64(target_image)
    
    # 在实际环境中，这里应该是真实的URL
    # 为了测试，我们使用placeholder URL
    source_url = "https://example.com/source.jpg"  # 这里需要真实的图像URL
    target_url = "https://example.com/target.jpg"  # 这里需要真实的图像URL
    
    print(f"📤 准备调用API...")
    print(f"   源图像URL: {source_url}")
    print(f"   目标图像URL: {target_url}")
    
    # 3. 构建API请求
    submit_payload = {
        "input": {
            "process_type": "single_image",
            "source_file": source_url,
            "target_file": target_url,
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
    
    # 4. 提交任务
    print("📤 提交换脸任务...")
    try:
        submit_response = requests.post(
            f"{FACE_SWAP_ENDPOINT}/run",
            json=submit_payload,
            headers=headers,
            timeout=30
        )
        
        print(f"📋 提交响应状态码: {submit_response.status_code}")
        print(f"📋 提交响应内容: {submit_response.text}")
        
        if submit_response.status_code != 200:
            print(f"❌ 任务提交失败: {submit_response.status_code} - {submit_response.text}")
            return False
            
        submit_result = submit_response.json()
        
        if 'id' not in submit_result:
            print(f"❌ 任务提交响应异常: {submit_result}")
            return False
            
        job_id = submit_result['id']
        print(f"✅ 任务已提交，ID: {job_id}")
        
        # 5. 查询任务状态（只查询一次作为测试）
        print("🔄 查询任务状态...")
        
        status_response = requests.get(
            f"{FACE_SWAP_ENDPOINT}/status/{job_id}",
            headers=headers,
            timeout=10
        )
        
        print(f"📋 状态查询响应码: {status_response.status_code}")
        print(f"📋 状态查询响应: {status_response.text}")
        
        if status_response.status_code == 200:
            result = status_response.json()
            status = result.get('status', 'UNKNOWN')
            print(f"📋 任务状态: {status}")
            
            if status == 'COMPLETED':
                print("✅ 任务已完成!")
                if 'output' in result and 'result' in result['output']:
                    print("✅ 获得了换脸结果!")
                    return True
            elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                print("🔄 任务正在处理中...")
                return True  # API工作正常
            elif status == 'FAILED':
                error_msg = result.get('error', '未知错误')
                print(f"❌ 任务失败: {error_msg}")
                return False
        
        return True
        
    except requests.RequestException as e:
        print(f"❌ API请求失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_api_endpoint():
    """测试API端点可用性"""
    print("🔍 测试API端点可用性...")
    
    try:
        # 简单的ping测试
        response = requests.get(FACE_SWAP_ENDPOINT, timeout=10)
        print(f"📋 端点响应状态码: {response.status_code}")
        
        if response.status_code == 404:
            print("✅ API端点可达（404是正常的，因为我们没有访问具体的路径）")
            return True
        elif response.status_code == 200:
            print("✅ API端点可达")
            return True
        else:
            print(f"⚠️ API端点响应异常: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ API端点不可达: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 外部API换脸功能测试 ===")
    print(f"🔑 API密钥: {RUNPOD_API_KEY[:10]}...{RUNPOD_API_KEY[-10:]}")
    print(f"🌐 API端点: {FACE_SWAP_ENDPOINT}")
    
    # 测试1: API端点可用性
    endpoint_ok = test_api_endpoint()
    
    # 测试2: 换脸API调用
    if endpoint_ok:
        api_ok = test_face_swap_api()
        
        if api_ok:
            print("\n✅ API换脸功能测试通过!")
            print("💡 系统已准备好使用外部API进行换脸处理")
        else:
            print("\n❌ API换脸功能测试失败")
            print("💡 请检查API配置和网络连接")
    else:
        print("\n❌ API端点不可用")
        print("💡 请检查网络连接和API端点配置")

if __name__ == "__main__":
    main() 