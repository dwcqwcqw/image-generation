#!/usr/bin/env python3
"""
CUDA库依赖修复脚本
Fix CUDA Library Dependencies for Face Swap Optimization
"""

import os
import sys
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_cuda_libraries():
    """检查CUDA库的可用性"""
    logger.info("🔍 Checking CUDA libraries...")
    
    required_libs = [
        "libcublasLt.so.12",
        "libcublas.so.12", 
        "libcudnn.so.8",
        "libcurand.so.10",
        "libcusolver.so.11",
        "libcusparse.so.12",
        "libnvjitlink.so.12"
    ]
    
    missing_libs = []
    
    for lib in required_libs:
        try:
            # 尝试使用ldconfig查找库
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if lib not in result.stdout:
                missing_libs.append(lib)
                logger.warning(f"⚠️ Missing library: {lib}")
            else:
                logger.info(f"✅ Found library: {lib}")
        except Exception as e:
            logger.warning(f"⚠️ Error checking {lib}: {e}")
            missing_libs.append(lib)
    
    return missing_libs

def install_cuda_packages():
    """安装CUDA相关包"""
    logger.info("🔧 Installing CUDA packages...")
    
    cuda_packages = [
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12", 
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nvjitlink-cu12"
    ]
    
    for package in cuda_packages:
        try:
            logger.info(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                package, "--no-cache-dir", "--quiet"
            ])
            logger.info(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to install {package}: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Error installing {package}: {e}")

def update_onnxruntime():
    """更新onnxruntime到GPU版本"""
    logger.info("🔧 Updating onnxruntime...")
    
    try:
        # 卸载CPU版本
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", 
            "onnxruntime", "-y", "--quiet"
        ])
        logger.info("✅ Uninstalled CPU onnxruntime")
    except:
        logger.info("ℹ️ CPU onnxruntime not found")
    
    try:
        # 安装GPU版本
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "onnxruntime-gpu", "--no-cache-dir", "--quiet"
        ])
        logger.info("✅ Installed GPU onnxruntime")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Failed to install onnxruntime-gpu: {e}")
        
        # 回退到CPU版本
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "onnxruntime", "--no-cache-dir", "--quiet"
            ])
            logger.info("✅ Fallback: Installed CPU onnxruntime")
        except Exception as fallback_error:
            logger.error(f"❌ Failed to install fallback onnxruntime: {fallback_error}")

def test_onnx_cuda():
    """测试ONNX Runtime CUDA功能"""
    logger.info("🧪 Testing ONNX Runtime CUDA...")
    
    try:
        import onnxruntime as ort
        
        # 检查可用提供程序
        providers = ort.get_available_providers()
        logger.info(f"📝 Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            logger.info("✅ CUDA provider is available")
            
            # 尝试创建CUDA会话
            try:
                # 创建简单的测试会话
                import tempfile
                import numpy as np
                
                # 创建最小的ONNX模型进行测试
                try:
                    import onnx
                    from onnx import helper, TensorProto
                    
                    # 创建恒等操作模型
                    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 64, 64])
                    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
                    identity_node = helper.make_node('Identity', ['input'], ['output'])
                    graph = helper.make_graph([identity_node], 'test_model', [input_tensor], [output_tensor])
                    model = helper.make_model(graph)
                    
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
                        onnx.save(model, temp_file.name)
                        
                        # 测试CUDA会话
                        session = ort.InferenceSession(
                            temp_file.name,
                            providers=['CUDAExecutionProvider']
                        )
                        
                        # 运行推理测试
                        input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)
                        output = session.run(None, {'input': input_data})
                        
                        logger.info("✅ CUDA inference test passed")
                        
                        # 清理临时文件
                        os.unlink(temp_file.name)
                        
                except ImportError:
                    logger.warning("⚠️ ONNX not available for detailed testing")
                except Exception as test_error:
                    logger.warning(f"⚠️ CUDA inference test failed: {test_error}")
                    
            except Exception as session_error:
                logger.warning(f"⚠️ CUDA session creation failed: {session_error}")
                
        else:
            logger.warning("⚠️ CUDA provider not available")
            
    except ImportError:
        logger.error("❌ onnxruntime not available")
    except Exception as e:
        logger.error(f"❌ ONNX Runtime test failed: {e}")

def create_cuda_fix_summary():
    """创建CUDA修复总结"""
    summary = """
# CUDA库修复总结

## 执行的修复操作

### 1. 检查CUDA库依赖
- 检查了关键的CUDA 12库文件
- 识别缺失的库文件

### 2. 安装CUDA包
- nvidia-cublas-cu12: CUDA BLAS库
- nvidia-cudnn-cu12: CUDA深度神经网络库  
- nvidia-cufft-cu12: CUDA FFT库
- nvidia-curand-cu12: CUDA随机数生成库
- nvidia-cusolver-cu12: CUDA线性代数库
- nvidia-cusparse-cu12: CUDA稀疏矩阵库
- nvidia-nvjitlink-cu12: CUDA JIT链接库

### 3. 更新ONNX Runtime
- 卸载CPU版本的onnxruntime
- 安装GPU版本的onnxruntime-gpu
- 如果失败则回退到CPU版本

### 4. 测试CUDA功能
- 验证CUDA提供程序可用性
- 测试CUDA推理功能

## 预期效果

修复后的系统应该能够：
1. **启用GPU加速** - 换脸模型使用CUDA执行
2. **提升性能** - 速度提升10-100倍
3. **提高质量** - GPU计算支持更精细的面部处理
4. **减少错误** - 解决libcublasLt.so.12缺失问题

## 验证方法

运行以下命令验证修复效果：
```bash
python -c "import onnxruntime as ort; print('CUDA available:', 'CUDAExecutionProvider' in ort.get_available_providers())"
```

如果输出 `CUDA available: True`，则修复成功。
"""
    
    with open('/tmp/cuda_fix_summary.md', 'w') as f:
        f.write(summary)
    
    logger.info("📝 CUDA fix summary saved to /tmp/cuda_fix_summary.md")

def main():
    """主修复流程"""
    logger.info("🚀 Starting CUDA library fix...")
    
    # 1. 检查当前状态
    missing_libs = check_cuda_libraries()
    
    # 2. 安装CUDA包
    install_cuda_packages()
    
    # 3. 更新ONNX Runtime
    update_onnxruntime()
    
    # 4. 测试功能
    test_onnx_cuda()
    
    # 5. 创建总结
    create_cuda_fix_summary()
    
    # 6. 重新检查
    logger.info("🔍 Re-checking CUDA libraries after fix...")
    remaining_missing = check_cuda_libraries()
    
    if len(remaining_missing) < len(missing_libs):
        logger.info(f"✅ Improvement: {len(missing_libs) - len(remaining_missing)} libraries fixed")
    
    if not remaining_missing:
        logger.info("🎉 All CUDA libraries are now available!")
    else:
        logger.warning(f"⚠️ Still missing {len(remaining_missing)} libraries: {remaining_missing}")
    
    logger.info("🏁 CUDA library fix completed")

if __name__ == "__main__":
    main() 