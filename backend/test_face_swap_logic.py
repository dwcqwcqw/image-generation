#!/usr/bin/env python3
"""
简化的换脸功能逻辑测试
主要测试handler中的换脸集成代码结构
"""

import os
import sys
import traceback

def test_handler_import():
    """测试handler模块导入和换脸功能结构"""
    print("🔄 测试handler模块导入...")
    
    try:
        # 添加当前目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        # 导入handler模块
        import handler
        print("✅ handler模块导入成功")
        
        # 检查换脸功能全局变量
        face_swap_vars = [
            'INSIGHTFACE_AVAILABLE', 
            'GFPGAN_AVAILABLE', 
            'OPENCV_AVAILABLE',
            'FACE_SWAP_AVAILABLE'
        ]
        
        print("\n📊 换脸功能状态变量:")
        for var in face_swap_vars:
            if hasattr(handler, var):
                value = getattr(handler, var)
                print(f"   {var}: {value}")
            else:
                print(f"   ❌ {var}: 未定义")
                
        # 检查换脸功能函数
        face_swap_functions = [
            'add_faceswap_path',
            'init_face_analyser', 
            'init_face_swapper',
            'detect_faces',
            'swap_face',
            'process_face_swap_pipeline',
            'pil_to_cv2',
            'cv2_to_pil', 
            'is_face_swap_available'
        ]
        
        print("\n🔧 换脸功能函数:")
        missing_functions = []
        for func in face_swap_functions:
            if hasattr(handler, func):
                print(f"   ✅ {func}: 已定义")
            else:
                print(f"   ❌ {func}: 缺失")
                missing_functions.append(func)
                
        if missing_functions:
            print(f"\n⚠️ 缺失的函数: {missing_functions}")
        else:
            print("\n🎉 所有换脸函数都已正确定义")
            
        # 测试is_face_swap_available函数
        if hasattr(handler, 'is_face_swap_available'):
            try:
                available = handler.is_face_swap_available()
                print(f"\n📈 换脸功能可用性检查: {available}")
            except Exception as e:
                print(f"\n❌ 换脸功能检查失败: {e}")
        
        # 测试依赖路径函数
        if hasattr(handler, 'add_faceswap_path'):
            try:
                handler.add_faceswap_path()
                print("✅ 换脸路径初始化成功")
            except Exception as e:
                print(f"⚠️ 换脸路径初始化警告: {e}")
                
        # 检查handler主要处理函数
        main_functions = ['handler', 'generate_image', 'generate_flux_images']
        print("\n🎯 主要处理函数:")
        for func in main_functions:
            if hasattr(handler, func):
                print(f"   ✅ {func}: 已定义")
            else:
                print(f"   ❌ {func}: 缺失")
                
        return True
        
    except Exception as e:
        print(f"❌ handler模块测试失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return False

def test_dependency_availability():
    """测试依赖可用性"""
    print("\n" + "="*50)
    print("  依赖可用性测试")
    print("="*50)
    
    dependencies = {
        'cv2': 'OpenCV',
        'insightface': 'InsightFace', 
        'onnxruntime': 'ONNX Runtime',
        'gfpgan': 'GFPGAN'
    }
    
    available_deps = []
    missing_deps = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}: 可用")
            available_deps.append(name)
        except ImportError:
            print(f"❌ {name}: 不可用")
            missing_deps.append(name)
            
    print(f"\n📊 依赖统计:")
    print(f"   可用: {len(available_deps)}/{len(dependencies)} 个")
    print(f"   缺失: {missing_deps}")
    
    return len(missing_deps) == 0

def test_face_swap_pipeline_structure():
    """测试换脸流水线结构"""
    print("\n" + "="*50)
    print("  换脸流水线结构测试")
    print("="*50)
    
    try:
        import handler
        
        if not hasattr(handler, 'process_face_swap_pipeline'):
            print("❌ process_face_swap_pipeline 函数不存在")
            return False
            
        # 检查函数签名（不实际调用）
        import inspect
        sig = inspect.signature(handler.process_face_swap_pipeline)
        params = list(sig.parameters.keys())
        
        print(f"✅ process_face_swap_pipeline 函数存在")
        print(f"   参数: {params}")
        
        expected_params = ['generated_images', 'uploaded_file_path']
        missing_params = [p for p in expected_params if p not in params]
        
        if missing_params:
            print(f"⚠️ 可能缺失的参数: {missing_params}")
        else:
            print("✅ 函数参数结构正确")
            
        return True
        
    except Exception as e:
        print(f"❌ 流水线结构测试失败: {e}")
        return False

def test_model_paths():
    """测试模型路径逻辑"""
    print("\n" + "="*50)
    print("  模型路径逻辑测试")
    print("="*50)
    
    # 测试各种可能的模型路径
    possible_paths = [
        "/runpod-volume/faceswap",
        "/workspace/faceswap", 
        "/app/faceswap",
        "./faceswap",
        "../faceswap"
    ]
    
    print("🔍 检查可能的模型路径:")
    found_paths = []
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"   ✅ {path} -> {abs_path}")
            found_paths.append(abs_path)
        else:
            print(f"   ❌ {path} (不存在)")
            
    if found_paths:
        print(f"\n📁 找到的路径: {found_paths}")
    else:
        print("\n⚠️ 未找到任何模型路径")
        
    return len(found_paths) > 0

def generate_deployment_dockerfile():
    """生成优化的Dockerfile配置建议"""
    print("\n" + "="*50)
    print("  Dockerfile优化建议")
    print("="*50)
    
    suggestions = [
        "# 优化的requirements.txt配置:",
        "insightface>=0.7.3",
        "onnxruntime-gpu>=1.16.0",
        "opencv-python>=4.10.0",
        "# gfpgan>=1.3.8  # 如果安装有问题可暂时注释",
        "",
        "# Dockerfile中的模型下载建议:",
        "# 在容器构建时预下载模型文件",
        "RUN mkdir -p /runpod-volume/faceswap",
        "# 或者设置环境变量指向其他路径",
        "ENV FACESWAP_MODEL_PATH=/app/models/faceswap",
        "",
        "# 运行时模型检查建议:",
        "# 在handler.py中添加模型文件存在性检查",
        "# 如果模型不存在，自动下载或回退到无换脸模式"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def main():
    """主测试函数"""
    print("🚀 开始换脸功能逻辑测试...")
    
    results = []
    
    # 执行各项测试
    results.append(("Handler导入测试", test_handler_import()))
    results.append(("依赖可用性测试", test_dependency_availability()))
    results.append(("流水线结构测试", test_face_swap_pipeline_structure()))
    results.append(("模型路径测试", test_model_paths()))
    
    # 输出测试结果总结
    print("\n" + "="*50)
    print("  测试结果总结")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {test_name}")
        if result:
            passed += 1
            
    print(f"\n📊 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！换脸功能结构完整。")
    elif passed >= total // 2:
        print("⚠️ 部分测试通过，需要检查缺失的组件。")
    else:
        print("❌ 多数测试失败，需要修复基础结构问题。")
        
    generate_deployment_dockerfile()

if __name__ == "__main__":
    main() 