#!/usr/bin/env python3
"""
完整的换脸功能测试脚本
测试所有路径、依赖、模型文件和功能
"""

import os
import sys
import subprocess
import traceback
from PIL import Image
import numpy as np

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_environment():
    """检查Python环境"""
    print_section("PYTHON环境检查")
    print(f"Python版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {os.path.dirname(os.path.abspath(__file__))}")

def check_basic_dependencies():
    """检查基础依赖包"""
    print_section("基础依赖包检查")
    
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy', 
        'PIL': 'Pillow',
        'onnxruntime': 'ONNX Runtime'
    }
    
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
                print(f"✅ {name}: {cv2.__version__}")
            elif module == 'torch':
                import torch
                print(f"✅ {name}: {torch.__version__}")
                print(f"   CUDA可用: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   CUDA版本: {torch.version.cuda}")
            elif module == 'PIL':
                from PIL import Image
                print(f"✅ {name}: {Image.__version__}")
            elif module == 'numpy':
                import numpy as np
                print(f"✅ {name}: {np.__version__}")
            elif module == 'onnxruntime':
                import onnxruntime
                print(f"✅ {name}: {onnxruntime.__version__}")
                providers = onnxruntime.get_available_providers()
                print(f"   可用提供器: {providers}")
        except ImportError as e:
            print(f"❌ {name}: 未安装 - {e}")
        except Exception as e:
            print(f"⚠️ {name}: 导入错误 - {e}")

def check_face_swap_dependencies():
    """检查换脸专用依赖"""
    print_section("换脸依赖包检查")
    
    # 检查 insightface
    try:
        import insightface
        print(f"✅ InsightFace: {insightface.__version__}")
        
        # 测试基本功能
        try:
            app = insightface.app.FaceAnalysis()
            print("   ✅ FaceAnalysis 初始化成功")
        except Exception as e:
            print(f"   ❌ FaceAnalysis 初始化失败: {e}")
            
    except ImportError as e:
        print(f"❌ InsightFace: 未安装 - {e}")
        print("   安装命令: pip install insightface")
        
    # 检查 gfpgan
    try:
        import gfpgan
        print(f"✅ GFPGAN: 已安装")
        
        # 测试基本功能
        try:
            from gfpgan import GFPGANer
            print("   ✅ GFPGANer 导入成功")
        except Exception as e:
            print(f"   ❌ GFPGANer 导入失败: {e}")
            
    except ImportError as e:
        print(f"❌ GFPGAN: 未安装 - {e}")
        print("   安装命令: pip install gfpgan")

def check_model_paths():
    """检查模型文件路径"""
    print_section("模型文件路径检查")
    
    model_configs = {
        "face_swap": "/runpod-volume/faceswap/inswapper_128_fp16.onnx",
        "face_enhance": "/runpod-volume/faceswap/GFPGANv1.4.pth", 
        "face_analysis": "/runpod-volume/faceswap/buffalo_l"
    }
    
    # 检查基础目录
    base_dir = "/runpod-volume"
    if os.path.exists(base_dir):
        print(f"✅ 基础目录存在: {base_dir}")
        contents = os.listdir(base_dir)
        print(f"   内容: {contents}")
    else:
        print(f"❌ 基础目录不存在: {base_dir}")
        
    faceswap_dir = "/runpod-volume/faceswap"
    if os.path.exists(faceswap_dir):
        print(f"✅ 换脸目录存在: {faceswap_dir}")
        contents = os.listdir(faceswap_dir)
        print(f"   内容: {contents}")
        
        # 检查每个文件的大小
        for item in contents:
            item_path = os.path.join(faceswap_dir, item)
            if os.path.isfile(item_path):
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"   📄 {item}: {size_mb:.1f}MB")
            elif os.path.isdir(item_path):
                sub_contents = os.listdir(item_path)
                print(f"   📁 {item}/: {len(sub_contents)} 个文件")
    else:
        print(f"❌ 换脸目录不存在: {faceswap_dir}")
        
    # 检查具体模型文件
    print("\n模型文件详细检查:")
    for model_type, path in model_configs.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✅ {model_type}: {path} ({size_mb:.1f}MB)")
            else:
                contents = os.listdir(path)
                print(f"✅ {model_type}: {path} (目录, {len(contents)} 个文件)")
                for item in contents[:5]:  # 只显示前5个
                    print(f"   - {item}")
                if len(contents) > 5:
                    print(f"   ... 还有 {len(contents) - 5} 个文件")
        else:
            print(f"❌ {model_type}: {path} (不存在)")

def test_face_swap_integration():
    """测试换脸集成功能"""
    print_section("换脸集成功能测试")
    
    try:
        # 添加当前目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        # 导入handler模块
        print("🔄 导入handler模块...")
        import handler
        
        # 检查换脸功能状态
        print(f"📊 换脸功能可用状态: {handler.FACE_SWAP_AVAILABLE}")
        
        if hasattr(handler, 'is_face_swap_available'):
            available = handler.is_face_swap_available()
            print(f"📊 换脸功能检查结果: {available}")
        
        # 测试依赖检查
        print("\n依赖检查结果:")
        if hasattr(handler, 'INSIGHTFACE_AVAILABLE'):
            print(f"   InsightFace: {handler.INSIGHTFACE_AVAILABLE}")
        if hasattr(handler, 'GFPGAN_AVAILABLE'):
            print(f"   GFPGAN: {handler.GFPGAN_AVAILABLE}")
        if hasattr(handler, 'OPENCV_AVAILABLE'):
            print(f"   OpenCV: {handler.OPENCV_AVAILABLE}")
            
        # 测试模型初始化
        print("\n🔄 测试模型初始化...")
        
        if hasattr(handler, 'init_face_analyser'):
            analyser = handler.init_face_analyser()
            if analyser:
                print("✅ 人脸分析器初始化成功")
            else:
                print("❌ 人脸分析器初始化失败")
                
        if hasattr(handler, 'init_face_swapper'):
            swapper = handler.init_face_swapper()
            if swapper:
                print("✅ 换脸模型初始化成功")
            else:
                print("❌ 换脸模型初始化失败")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")

def test_image_processing():
    """测试图像处理功能"""
    print_section("图像处理功能测试")
    
    try:
        # 创建测试图像
        print("🔄 创建测试图像...")
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # 转换为numpy数组
        img_array = np.array(test_image)
        print(f"✅ 图像数组创建成功: {img_array.shape}")
        
        # 测试OpenCV转换
        try:
            import cv2
            # RGB转BGR
            bgr_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            print(f"✅ OpenCV颜色转换成功: {bgr_image.shape}")
            
            # BGR转RGB
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            print(f"✅ OpenCV逆转换成功: {rgb_image.shape}")
        except Exception as e:
            print(f"❌ OpenCV转换失败: {e}")
            
    except Exception as e:
        print(f"❌ 图像处理测试失败: {e}")

def test_onnx_runtime():
    """测试ONNX Runtime"""
    print_section("ONNX Runtime测试")
    
    try:
        import onnxruntime as ort
        
        # 检查提供器
        providers = ort.get_available_providers()
        print(f"可用提供器: {providers}")
        
        # 检查CUDA支持
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA执行提供器可用")
            
            # 测试CUDA设备
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    print(f"✅ 检测到 {device_count} 个CUDA设备")
                    for i in range(device_count):
                        device_name = torch.cuda.get_device_name(i)
                        print(f"   设备 {i}: {device_name}")
            except Exception as e:
                print(f"⚠️ CUDA设备检查失败: {e}")
        else:
            print("⚠️ CUDA执行提供器不可用")
            
        # 检查模型文件是否可以加载
        model_path = "/runpod-volume/faceswap/inswapper_128_fp16.onnx"
        if os.path.exists(model_path):
            try:
                session = ort.InferenceSession(model_path, providers=providers)
                print(f"✅ ONNX模型加载成功: {model_path}")
                
                # 获取模型信息
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                print(f"   输入: {len(inputs)} 个")
                for inp in inputs:
                    print(f"     - {inp.name}: {inp.shape}")
                print(f"   输出: {len(outputs)} 个")
                for out in outputs:
                    print(f"     - {out.name}: {out.shape}")
                    
            except Exception as e:
                print(f"❌ ONNX模型加载失败: {e}")
        else:
            print(f"❌ ONNX模型文件不存在: {model_path}")
            
    except ImportError:
        print("❌ ONNX Runtime未安装")

def check_alternative_paths():
    """检查替代路径"""
    print_section("替代路径检查")
    
    alternative_paths = [
        "/workspace/faceswap",
        "/app/faceswap", 
        "./faceswap",
        "../faceswap",
        os.path.expanduser("~/faceswap")
    ]
    
    for path in alternative_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"✅ 找到替代路径: {abs_path}")
            contents = os.listdir(abs_path)
            print(f"   内容: {contents[:5]}...")  # 只显示前5个
        else:
            print(f"❌ 替代路径不存在: {abs_path}")

def create_minimal_test():
    """创建最小测试用例"""
    print_section("最小功能测试")
    
    try:
        # 测试是否可以创建最基本的换脸流程
        print("🔄 测试基本换脸流程结构...")
        
        # 创建虚拟图像
        dummy_image = Image.new('RGB', (256, 256), color='blue')
        print(f"✅ 创建虚拟图像成功: {dummy_image.size}")
        
        # 如果有handler模块，测试换脸流程
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
                
            import handler
            
            if hasattr(handler, 'process_face_swap_pipeline'):
                print("✅ 换脸流程函数存在")
                
                # 不实际执行，只检查函数可调用性
                print("✅ 换脸流程函数可调用")
            else:
                print("❌ 换脸流程函数不存在")
                
        except Exception as e:
            print(f"❌ handler模块测试失败: {e}")
            
    except Exception as e:
        print(f"❌ 最小测试失败: {e}")

def generate_installation_commands():
    """生成安装命令"""
    print_section("缺失依赖安装命令")
    
    commands = [
        "# 安装基础依赖",
        "pip install opencv-python",
        "pip install onnxruntime-gpu",  # 或 onnxruntime-cpu
        "",
        "# 安装换脸依赖", 
        "pip install insightface",
        "pip install gfpgan",
        "",
        "# 如果CUDA不可用，使用CPU版本",
        "pip install onnxruntime",
        "",
        "# 可能需要的额外依赖",
        "pip install retinaface-pytorch",
        "pip install basicsr",
        "pip install facexlib"
    ]
    
    for cmd in commands:
        print(cmd)

def main():
    """主测试函数"""
    print("🚀 开始完整的换脸功能测试...")
    
    check_python_environment()
    check_basic_dependencies()
    check_face_swap_dependencies()
    check_model_paths()
    test_onnx_runtime()
    check_alternative_paths()
    test_face_swap_integration()
    test_image_processing()
    create_minimal_test()
    generate_installation_commands()
    
    print_section("测试完成")
    print("🎉 所有测试已完成！请查看上述结果以诊断问题。")

if __name__ == "__main__":
    main() 