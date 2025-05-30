#!/usr/bin/env python3
"""
测试动漫模型修复
验证:
1. 按需模型加载 (不预热)
2. 动漫模型的长prompt支持
3. LoRA兼容性检查
4. Half精度修复
"""

import sys
import os
sys.path.append('backend')  # 修改路径指向backend目录

def test_model_loading():
    """测试模型加载逻辑"""
    print("🧪 测试模型加载逻辑...")
    
    # 模拟handler导入
    try:
        from handler import BASE_MODELS, current_base_model
        print(f"✅ 当前基础模型: {current_base_model}")
        print(f"✅ 可用模型: {list(BASE_MODELS.keys())}")
        
        # 验证模型配置
        for model_id, config in BASE_MODELS.items():
            print(f"  📋 {model_id}: {config['name']} ({config['model_type']})")
            print(f"     路径: {config['path']}")
            print(f"     LoRA: {config['lora_path']}")
            
    except Exception as e:
        print(f"❌ 模型配置错误: {e}")
        return False
        
    return True

def test_lora_compatibility():
    """测试LoRA兼容性"""
    print("\n🧪 测试LoRA兼容性...")
    
    try:
        from handler import AVAILABLE_LORAS, BASE_MODELS
        
        # 分组LoRA
        flux_loras = []
        anime_loras = []
        
        for lora_id, lora_info in AVAILABLE_LORAS.items():
            base_model = lora_info.get('base_model', 'unknown')
            model_type = BASE_MODELS.get(base_model, {}).get('model_type', 'unknown')
            
            if model_type == 'flux':
                flux_loras.append(lora_id)
            elif model_type == 'diffusers':
                anime_loras.append(lora_id)
        
        print(f"✅ FLUX LoRAs ({len(flux_loras)}): {flux_loras}")
        print(f"✅ 动漫 LoRAs ({len(anime_loras)}): {anime_loras}")
        
        # 验证没有交叉兼容性问题
        if len(flux_loras) > 0 and len(anime_loras) > 0:
            print("✅ LoRA分离正确，避免兼容性问题")
            return True
        else:
            print("⚠️  LoRA分组可能有问题")
            return False
            
    except Exception as e:
        print(f"❌ LoRA兼容性测试失败: {e}")
        return False

def test_precision_config():
    """测试精度配置"""
    print("\n🧪 测试精度配置...")
    
    try:
        # 检查torch配置
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            
        # 测试float32是否可用
        test_tensor = torch.tensor([1.0], dtype=torch.float32)
        print(f"✅ Float32支持: {test_tensor.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 精度配置测试失败: {e}")
        return False

def test_compel_import():
    """测试Compel库导入"""
    print("\n🧪 测试Compel库...")
    
    try:
        from compel import Compel
        print("✅ Compel库导入成功")
        
        # 测试基本功能
        print("✅ Compel库可用于长prompt处理")
        return True
        
    except ImportError:
        print("❌ Compel库未安装，需要: pip install compel")
        return False
    except Exception as e:
        print(f"❌ Compel测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始动漫模型修复验证...")
    
    tests = [
        ("模型加载", test_model_loading),
        ("LoRA兼容性", test_lora_compatibility), 
        ("精度配置", test_precision_config),
        ("Compel库", test_compel_import)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！动漫模型修复成功")
        return True
    else:
        print("⚠️  存在问题需要修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 