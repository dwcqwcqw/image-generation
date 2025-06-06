#!/usr/bin/env python3
"""
换脸集成功能测试脚本

测试新的图生图流程：
1. 真人模型：文生图 + 换脸
2. 动漫模型：传统图生图
3. 换脸功能的可用性检查
"""

import sys
import os
import base64
import json
from PIL import Image
import io

# 添加backend路径
sys.path.append('backend')

def create_test_image(width=512, height=512, color=(255, 255, 255)):
    """创建测试图像"""
    img = Image.new('RGB', (width, height), color)
    return img

def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_data = buffer.getvalue()
    return base64.b64encode(img_data).decode('utf-8')

def test_face_swap_availability():
    """测试换脸功能是否可用"""
    print("🔍 测试换脸功能可用性...")
    
    try:
        from backend.face_swap_integration import is_face_swap_available, MODELS_CONFIG
        
        available = is_face_swap_available()
        print(f"✅ 换脸功能可用性: {available}")
        
        print("\n📁 模型配置:")
        for model_name, model_path in MODELS_CONFIG.items():
            exists = os.path.exists(model_path)
            print(f"  - {model_name}: {model_path} {'✅' if exists else '❌'}")
        
        return available
        
    except ImportError as e:
        print(f"❌ 导入换脸模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查换脸可用性失败: {e}")
        return False

def test_face_swap_pipeline():
    """测试换脸流水线"""
    print("\n🎭 测试换脸流水线...")
    
    try:
        from backend.face_swap_integration import process_face_swap_pipeline
        
        # 创建测试图像
        generated_image = create_test_image(512, 512, (200, 200, 200))
        source_image = create_test_image(512, 512, (100, 100, 100))
        
        print("🔄 执行换脸流水线...")
        result_image, success = process_face_swap_pipeline(generated_image, source_image)
        
        print(f"✅ 换脸流水线完成: 成功={success}")
        print(f"📏 结果图像尺寸: {result_image.size}")
        
        return success
        
    except Exception as e:
        print(f"❌ 换脸流水线测试失败: {e}")
        return False

def test_image_to_image_realistic():
    """测试真人模型的图生图功能"""
    print("\n🎯 测试真人模型图生图...")
    
    try:
        from backend.handler import image_to_image
        
        # 创建测试参数
        test_image = create_test_image(512, 512, (150, 150, 150))
        image_base64 = image_to_base64(test_image)
        
        params = {
            'baseModel': 'realistic',
            'prompt': 'a handsome man portrait, realistic, high quality',
            'negativePrompt': 'blurry, low quality',
            'image': image_base64,
            'width': 512,
            'height': 512,
            'steps': 10,  # 减少步数以加快测试
            'cfgScale': 7.0,
            'seed': 42,
            'numImages': 1,
            'denoisingStrength': 0.7
        }
        
        print("🔄 执行真人模型图生图...")
        results = image_to_image(params)
        
        print(f"✅ 真人模型图生图完成: 生成了 {len(results)} 张图像")
        
        for i, result in enumerate(results):
            print(f"  - 图像 {i+1}: {result.get('type', 'unknown')} | 换脸成功: {result.get('faceSwapSuccess', 'N/A')}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ 真人模型图生图测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def test_image_to_image_anime():
    """测试动漫模型的图生图功能"""
    print("\n🎨 测试动漫模型图生图...")
    
    try:
        from backend.handler import image_to_image
        
        # 创建测试参数
        test_image = create_test_image(512, 512, (180, 180, 180))
        image_base64 = image_to_base64(test_image)
        
        params = {
            'baseModel': 'anime',
            'prompt': 'anime character, beautiful art style',
            'negativePrompt': 'ugly, distorted',
            'image': image_base64,
            'width': 512,
            'height': 512,
            'steps': 10,  # 减少步数以加快测试
            'cfgScale': 7.0,
            'seed': 42,
            'numImages': 1,
            'denoisingStrength': 0.7
        }
        
        print("🔄 执行动漫模型图生图...")
        results = image_to_image(params)
        
        print(f"✅ 动漫模型图生图完成: 生成了 {len(results)} 张图像")
        
        for i, result in enumerate(results):
            print(f"  - 图像 {i+1}: {result.get('type', 'unknown')}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ 动漫模型图生图测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n🔧 测试模型加载...")
    
    try:
        from backend.handler import load_specific_model, BASE_MODELS
        
        print("📝 可用模型:")
        for model_name, model_config in BASE_MODELS.items():
            print(f"  - {model_name}: {model_config['name']} ({model_config['model_type']})")
        
        # 测试加载真人模型
        print("\n🔄 测试加载真人模型...")
        load_specific_model('realistic')
        print("✅ 真人模型加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始换脸集成功能测试\n")
    
    results = {}
    
    # 1. 测试换脸功能可用性
    results['face_swap_available'] = test_face_swap_availability()
    
    # 2. 测试换脸流水线
    if results['face_swap_available']:
        results['face_swap_pipeline'] = test_face_swap_pipeline()
    else:
        results['face_swap_pipeline'] = False
        print("⏭️  跳过换脸流水线测试（功能不可用）")
    
    # 3. 测试模型加载
    results['model_loading'] = test_model_loading()
    
    # 4. 测试真人模型图生图
    if results['model_loading']:
        results['realistic_img2img'] = test_image_to_image_realistic()
    else:
        results['realistic_img2img'] = False
        print("⏭️  跳过真人模型测试（模型加载失败）")
    
    # 5. 测试动漫模型图生图
    if results['model_loading']:
        results['anime_img2img'] = test_image_to_image_anime()
    else:
        results['anime_img2img'] = False
        print("⏭️  跳过动漫模型测试（模型加载失败）")
    
    # 总结
    print("\n" + "="*50)
    print("📋 测试结果总结:")
    print("="*50)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！换脸集成功能正常工作。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 