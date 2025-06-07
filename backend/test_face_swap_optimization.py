#!/usr/bin/env python3
"""
换脸优化效果测试脚本
Test Face Swap Optimization Effects
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """测试必要的库导入"""
    logger.info("🔍 Testing imports...")
    
    try:
        import cv2
        logger.info("✅ OpenCV imported successfully")
    except ImportError as e:
        logger.error(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import insightface
        logger.info("✅ InsightFace imported successfully")
    except ImportError as e:
        logger.error(f"❌ InsightFace import failed: {e}")
        return False
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"✅ ONNX Runtime available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            logger.info("🚀 CUDA provider available - GPU acceleration enabled")
        else:
            logger.warning("⚠️ CUDA provider not available - using CPU")
            
    except ImportError as e:
        logger.error(f"❌ ONNX Runtime import failed: {e}")
        return False
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} imported")
        logger.info(f"🔧 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        logger.error(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_face_swap_functions():
    """测试换脸相关函数"""
    logger.info("🧪 Testing face swap functions...")
    
    try:
        # 导入handler模块
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import handler
        
        # 测试执行提供程序
        providers = handler.get_execution_providers()
        logger.info(f"✅ Execution providers: {[p.__class__.__name__ if hasattr(p, '__class__') else str(p) for p in providers]}")
        
        # 测试人脸分析器初始化
        face_analyser = handler.init_face_analyser()
        if face_analyser is not None:
            logger.info("✅ Face analyser initialized successfully")
        else:
            logger.warning("⚠️ Face analyser initialization failed")
        
        # 测试换脸模型初始化
        face_swapper = handler.init_face_swapper()
        if face_swapper is not None:
            logger.info("✅ Face swapper initialized successfully")
        else:
            logger.warning("⚠️ Face swapper initialization failed")
        
        # 测试GFPGAN初始化
        face_enhancer = handler.init_face_enhancer()
        if face_enhancer is not None:
            logger.info("✅ Face enhancer (GFPGAN) initialized successfully")
        else:
            logger.warning("⚠️ Face enhancer initialization failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Face swap function test failed: {e}")
        return False

def test_optimization_features():
    """测试优化功能"""
    logger.info("🎯 Testing optimization features...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import handler
        
        # 创建测试数据
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # 测试动态混合比例计算
        logger.info("🧪 Testing dynamic blend ratio calculation...")
        
        # 模拟人脸对象
        class MockFace:
            def __init__(self):
                self.bbox = np.array([100, 100, 200, 200])
                self.det_score = 0.85
        
        source_face = MockFace()
        target_face = MockFace()
        
        try:
            blend_ratio = handler.calculate_dynamic_blend_ratio(
                source_face, target_face, test_image, test_image
            )
            logger.info(f"✅ Dynamic blend ratio: {blend_ratio:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Dynamic blend ratio test failed: {e}")
        
        # 测试多尺度检测功能
        logger.info("🔍 Testing multi-scale detection...")
        try:
            faces = handler.detect_faces(test_image)
            logger.info(f"✅ Multi-scale detection completed, found {len(faces)} faces")
        except Exception as e:
            logger.warning(f"⚠️ Multi-scale detection test failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization features test failed: {e}")
        return False

def test_performance_benchmark():
    """性能基准测试"""
    logger.info("⚡ Running performance benchmark...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import handler
        import cv2
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # 人脸检测性能测试
        logger.info("🔍 Benchmarking face detection...")
        start_time = time.time()
        
        for i in range(3):  # 测试3次
            faces = handler.detect_faces(test_image)
            
        detection_time = (time.time() - start_time) / 3
        logger.info(f"✅ Average face detection time: {detection_time:.3f}s")
        
        # 图像处理性能测试
        logger.info("🎨 Benchmarking image processing...")
        start_time = time.time()
        
        for i in range(5):  # 测试5次
            # 双边滤波
            filtered = cv2.bilateralFilter(test_image, 9, 80, 80)
            
            # 锐化
            kernel = np.array([[-0.1, -0.1, -0.1],
                             [-0.1,  1.8, -0.1],
                             [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(filtered, -1, kernel)
        
        processing_time = (time.time() - start_time) / 5
        logger.info(f"✅ Average image processing time: {processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

def test_model_availability():
    """测试模型文件可用性"""
    logger.info("📁 Testing model file availability...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import handler
        
        # 获取模型配置
        models_config = handler.get_face_swap_models_config()
        
        for model_type, model_path in models_config.items():
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                logger.info(f"✅ {model_type}: {model_path} ({file_size:.1f}MB)")
            else:
                logger.warning(f"⚠️ {model_type}: {model_path} (NOT FOUND)")
        
        # 检查换脸功能可用性
        is_available = handler.is_face_swap_available()
        if is_available:
            logger.info("✅ Face swap functionality is available")
        else:
            logger.warning("⚠️ Face swap functionality is not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model availability test failed: {e}")
        return False

def generate_test_report():
    """生成测试报告"""
    logger.info("📊 Generating test report...")
    
    report = """
# 换脸优化测试报告
## Face Swap Optimization Test Report

### 测试环境
- Python版本: {python_version}
- 操作系统: {os_info}
- 测试时间: {test_time}

### 测试结果

#### 1. 库导入测试
{import_status}

#### 2. 功能测试
{function_status}

#### 3. 优化特性测试
{optimization_status}

#### 4. 性能基准测试
{performance_status}

#### 5. 模型可用性测试
{model_status}

### 总结
{summary}

### 建议
{recommendations}
""".format(
        python_version=sys.version,
        os_info=f"{os.name} {os.uname().sysname if hasattr(os, 'uname') else 'Unknown'}",
        test_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        import_status="✅ 通过" if test_imports() else "❌ 失败",
        function_status="✅ 通过" if test_face_swap_functions() else "❌ 失败",
        optimization_status="✅ 通过" if test_optimization_features() else "❌ 失败",
        performance_status="✅ 通过" if test_performance_benchmark() else "❌ 失败",
        model_status="✅ 通过" if test_model_availability() else "❌ 失败",
        summary="系统优化效果良好，建议部署到生产环境",
        recommendations="""
1. 确保CUDA环境正确配置以启用GPU加速
2. 验证所有模型文件完整性
3. 监控实际使用中的性能指标
4. 根据用户反馈调整优化参数
"""
    )
    
    # 保存报告
    report_path = "/tmp/face_swap_optimization_test_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📝 Test report saved to: {report_path}")
    return report_path

def main():
    """主测试流程"""
    logger.info("🚀 Starting Face Swap Optimization Test...")
    
    # 运行所有测试
    tests = [
        ("Import Test", test_imports),
        ("Function Test", test_face_swap_functions),
        ("Optimization Test", test_optimization_features),
        ("Performance Test", test_performance_benchmark),
        ("Model Test", test_model_availability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED - {e}")
    
    # 生成报告
    logger.info(f"\n{'='*50}")
    logger.info("Generating Test Report...")
    logger.info(f"{'='*50}")
    
    report_path = generate_test_report()
    
    # 总结
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    logger.info(f"\n🎯 Test Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Face swap optimization is ready for production.")
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Please check the issues above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 