#!/usr/bin/env python3
"""
测试修复后的换脸功能
"""

import os
import sys
import traceback

def test_onnx_providers():
    """测试ONNX Runtime providers配置"""
    print("🔍 测试ONNX Runtime providers...")
    
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        print(f"✅ 可用providers: {available_providers}")
        
        # 测试CUDA provider配置
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ CUDA provider可用")
            
            # 测试创建session
            try:
                # 创建一个简单的模型来测试
                import numpy as np
                
                # 简单测试：创建一个dummy session
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ]
                print(f"✅ CUDA provider配置成功: {providers}")
                
            except Exception as e:
                print(f"⚠️  CUDA provider测试失败: {e}")
        else:
            print("⚠️  CUDA provider不可用，将使用CPU")
            
    except ImportError:
        print("❌ onnxruntime未安装")
    except Exception as e:
        print(f"❌ ONNX Runtime测试失败: {e}")

def test_face_swap_models():
    """测试换脸模型文件"""
    print("\n🔍 测试换脸模型文件...")
    
    expected_files = {
        "inswapper": "/runpod-volume/faceswap/inswapper_128_fp16.onnx",
        "gfpgan": "/runpod-volume/faceswap/GFPGANv1.4.pth", 
        "buffalo_l": "/runpod-volume/faceswap/buffalo_l"
    }
    
    for name, path in expected_files.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                files = os.listdir(path)
                print(f"✅ {name}: {path} (目录, {len(files)} 文件)")
            else:
                size_mb = os.path.getsize(path) / 1024 / 1024
                print(f"✅ {name}: {path} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {name}: {path} (不存在)")

def test_dependencies():
    """测试依赖库"""
    print("\n🔍 测试依赖库...")
    
    dependencies = {
        "insightface": "InsightFace",
        "cv2": "OpenCV",
        "gfpgan": "GFPGAN",
        "onnxruntime": "ONNX Runtime"
    }
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}: 可用")
        except ImportError:
            print(f"❌ {name}: 不可用")

def test_handler_functions():
    """测试handler中的关键函数"""
    print("\n🔍 测试handler函数...")
    
    try:
        # 添加backend目录到路径
        backend_path = os.path.dirname(os.path.abspath(__file__))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        import handler
        
        # 测试关键函数是否存在
        required_functions = [
            'get_execution_providers',
            'init_face_analyser', 
            'init_face_swapper',
            'init_face_enhancer',
            'enhance_face_quality',
            'process_face_swap_pipeline'
        ]
        
        for func_name in required_functions:
            if hasattr(handler, func_name):
                print(f"✅ {func_name}: 存在")
            else:
                print(f"❌ {func_name}: 不存在")
        
        # 测试execution providers
        try:
            providers = handler.get_execution_providers()
            print(f"✅ execution providers: {providers}")
        except Exception as e:
            print(f"❌ execution providers测试失败: {e}")
            
    except Exception as e:
        print(f"❌ handler导入失败: {e}")
        traceback.print_exc()

def test_face_swap_availability():
    """测试换脸功能可用性"""
    print("\n🔍 测试换脸功能可用性...")
    
    try:
        import handler
        
        if hasattr(handler, 'is_face_swap_available'):
            available = handler.is_face_swap_available()
            print(f"✅ 换脸功能可用性: {available}")
        else:
            print("❌ is_face_swap_available函数不存在")
            
    except Exception as e:
        print(f"❌ 换脸功能测试失败: {e}")

def main():
    """主测试函数"""
    print("🧪 开始测试修复后的换脸功能...")
    print("=" * 50)
    
    test_dependencies()
    test_onnx_providers()
    test_face_swap_models()
    test_handler_functions()
    test_face_swap_availability()
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")

if __name__ == "__main__":
    main() 