#!/usr/bin/env python3
"""
测试URL格式换脸结果处理功能
"""

import requests
import base64
from PIL import Image
import io

def test_url_detection():
    """测试URL格式检测"""
    print("=== URL格式检测测试 ===")
    
    # 测试数据
    test_cases = [
        ("https://example.com/image.jpg", True, "HTTPS URL"),
        ("http://example.com/image.jpg", True, "HTTP URL"),
        ("iVBORw0KGgoAAAANSUhEUgAAAA...", False, "Base64数据"),
        ("/9j/4AAQSkZJRgABAQAAAQABAAD...", False, "Base64图像数据"),
        ("ftp://example.com/image.jpg", False, "FTP URL"),
        ("", False, "空字符串"),
        (None, False, "None值")
    ]
    
    for test_data, expected_is_url, description in test_cases:
        if test_data is None:
            is_url = False
        else:
            is_url = isinstance(test_data, str) and test_data.startswith(('http://', 'https://'))
        
        status = "✅" if is_url == expected_is_url else "❌"
        print(f"{status} {description}: {is_url} (期望: {expected_is_url})")

def test_url_download():
    """测试URL下载功能（使用一个真实的测试图片URL）"""
    print("\n=== URL下载测试 ===")
    
    # 使用一个公开的测试图片URL
    test_url = "https://httpbin.org/image/jpeg"
    
    try:
        print(f"📥 测试下载: {test_url}")
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ 下载成功，状态码: {response.status_code}")
            print(f"✅ 内容长度: {len(response.content)} 字节")
            print(f"✅ 内容类型: {response.headers.get('content-type', 'unknown')}")
            
            # 尝试打开为图像
            try:
                image = Image.open(io.BytesIO(response.content))
                print(f"✅ 图像解析成功: {image.size} 像素, 模式: {image.mode}")
                return True
            except Exception as img_error:
                print(f"❌ 图像解析失败: {img_error}")
                return False
        else:
            print(f"❌ 下载失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 下载异常: {e}")
        return False

def test_base64_processing():
    """测试Base64处理功能"""
    print("\n=== Base64处理测试 ===")
    
    # 创建一个小的测试图像并转换为Base64
    try:
        # 创建1x1像素的红色图像
        test_image = Image.new('RGB', (1, 1), color='red')
        
        # 转换为字节
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # 编码为Base64
        base64_data = base64.b64encode(img_bytes).decode('utf-8')
        print(f"✅ 创建测试Base64数据: {len(base64_data)} 字符")
        
        # 解码测试
        try:
            decoded_bytes = base64.b64decode(base64_data)
            decoded_image = Image.open(io.BytesIO(decoded_bytes))
            print(f"✅ Base64解码成功: {decoded_image.size} 像素, 模式: {decoded_image.mode}")
            return True
        except Exception as decode_error:
            print(f"❌ Base64解码失败: {decode_error}")
            return False
            
    except Exception as e:
        print(f"❌ Base64测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 API换脸结果格式处理测试")
    print("=" * 50)
    
    # 运行所有测试
    test_url_detection()
    url_test_passed = test_url_download()
    base64_test_passed = test_base64_processing()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"✅ URL下载测试: {'通过' if url_test_passed else '失败'}")
    print(f"✅ Base64处理测试: {'通过' if base64_test_passed else '失败'}")
    
    overall_success = url_test_passed and base64_test_passed
    print(f"\n🎯 总体测试结果: {'✅ 全部通过' if overall_success else '❌ 有测试失败'}")
    
    return overall_success

if __name__ == "__main__":
    main() 