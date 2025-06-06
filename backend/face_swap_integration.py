"""
Face Swap Integration Module for AI Image Generation Backend
将换脸功能集成到AI图像生成后端的模块

基于 faceswap 项目的功能，专门为真人模型的图生图流程设计
"""

import os
import sys
import cv2
import numpy as np
import torch
import random
from typing import Optional, List, Tuple
from PIL import Image
import traceback

# 添加 faceswap 模块路径，支持本地开发和生产环境
def add_faceswap_path():
    """动态添加faceswap模块路径"""
    possible_paths = [
        "/Users/baileyli/Documents/AI同志项目/image generation/faceswap",  # 本地开发
        "/app/faceswap",  # Docker容器
        "/workspace/faceswap",  # RunPod
        "../faceswap",  # 相对路径
        "./faceswap"  # 当前目录
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            print(f"✓ Added faceswap path: {path}")
            return True
    
    print("⚠️ No faceswap path found, using system path")
    return False

add_faceswap_path()

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
    print("✓ InsightFace available for face analysis")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available - face swap will be disabled")

try:
    import gfpgan
    GFPGAN_AVAILABLE = True
    print("✓ GFPGAN available for face enhancement")
except ImportError:
    GFPGAN_AVAILABLE = False
    print("⚠️ GFPGAN not available - face enhancement will be disabled")

# 模型路径配置
MODELS_CONFIG = {
    "face_swap": "/runpod-volume/faceswap/inswapper_128_fp16.onnx",
    "face_enhance": "/runpod-volume/faceswap/GFPGANv1.4.pth", 
    "face_analysis": "/runpod-volume/faceswap/buffalo_l"
}

# 全局模型缓存
_face_analyser = None
_face_swapper = None
_face_enhancer = None

def get_execution_providers():
    """获取执行provider列表，优先使用CUDA"""
    providers = []
    if torch.cuda.is_available():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    return providers

def init_face_analyser():
    """初始化人脸分析器"""
    global _face_analyser
    
    if not INSIGHTFACE_AVAILABLE:
        print("❌ InsightFace not available, face analysis disabled")
        return None
        
    if _face_analyser is None:
        try:
            print("🔄 Initializing face analyser...")
            
            # 检查模型路径
            model_path = MODELS_CONFIG["face_analysis"]
            if not os.path.exists(model_path):
                print(f"❌ Face analysis model not found: {model_path}")
                return None
            
            _face_analyser = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root=os.path.dirname(model_path),
                providers=get_execution_providers()
            )
            _face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ Face analyser initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize face analyser: {e}")
            _face_analyser = None
            
    return _face_analyser

def init_face_swapper():
    """初始化换脸模型"""
    global _face_swapper
    
    if not INSIGHTFACE_AVAILABLE:
        print("❌ InsightFace not available, face swapping disabled")
        return None
        
    if _face_swapper is None:
        try:
            print("🔄 Initializing face swapper...")
            
            # 检查模型路径
            model_path = MODELS_CONFIG["face_swap"]
            if not os.path.exists(model_path):
                print(f"❌ Face swap model not found: {model_path}")
                return None
            
            _face_swapper = insightface.model_zoo.get_model(
                model_path,
                providers=get_execution_providers()
            )
            print("✅ Face swapper initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize face swapper: {e}")
            _face_swapper = None
            
    return _face_swapper

def init_face_enhancer():
    """初始化人脸增强器"""
    global _face_enhancer
    
    if not GFPGAN_AVAILABLE:
        print("❌ GFPGAN not available, face enhancement disabled")
        return None
        
    if _face_enhancer is None:
        try:
            print("🔄 Initializing face enhancer...")
            
            # 检查模型路径
            model_path = MODELS_CONFIG["face_enhance"]
            if not os.path.exists(model_path):
                print(f"❌ Face enhancement model not found: {model_path}")
                return None
            
            # 选择设备
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                
            _face_enhancer = gfpgan.GFPGANer(
                model_path=model_path,
                upscale=1,
                device=device
            )
            print(f"✅ Face enhancer initialized successfully on {device}")
            
        except Exception as e:
            print(f"❌ Failed to initialize face enhancer: {e}")
            _face_enhancer = None
            
    return _face_enhancer

def detect_faces(image: np.ndarray) -> List:
    """
    检测图像中的人脸
    
    Args:
        image: OpenCV格式图像 (BGR)
        
    Returns:
        人脸列表，按检测置信度排序
    """
    try:
        face_analyser = init_face_analyser()
        if face_analyser is None:
            return []
        
        faces = face_analyser.get(image)
        if faces is None:
            return []
        
        # 按检测置信度排序
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        print(f"🔍 Detected {len(faces)} faces")
        
        return faces
        
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return []

def swap_face(source_face, target_face, target_image: np.ndarray) -> np.ndarray:
    """
    执行人脸置换
    
    Args:
        source_face: 源人脸特征
        target_face: 目标人脸特征
        target_image: 目标图像 (BGR格式)
        
    Returns:
        换脸后的图像
    """
    try:
        face_swapper = init_face_swapper()
        if face_swapper is None:
            print("❌ Face swapper not available")
            return target_image
        
        print("🔄 Performing face swap...")
        swapped_image = face_swapper.get(target_image, target_face, source_face, paste_back=True)
        print("✅ Face swap completed")
        
        return swapped_image
        
    except Exception as e:
        print(f"❌ Face swap error: {e}")
        return target_image

def enhance_faces(image: np.ndarray) -> np.ndarray:
    """
    增强图像中的人脸质量
    
    Args:
        image: 输入图像 (BGR格式)
        
    Returns:
        增强后的图像
    """
    try:
        face_enhancer = init_face_enhancer()
        if face_enhancer is None:
            print("❌ Face enhancer not available")
            return image
        
        print("🔄 Enhancing faces...")
        _, _, enhanced_image = face_enhancer.enhance(image, paste_back=True)
        print("✅ Face enhancement completed")
        
        return enhanced_image
        
    except Exception as e:
        print(f"❌ Face enhancement error: {e}")
        return image

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """将PIL图像转换为OpenCV格式"""
    # PIL是RGB，OpenCV是BGR
    rgb_image = np.array(pil_image)
    if len(rgb_image.shape) == 3:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = rgb_image
    return bgr_image

def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """将OpenCV图像转换为PIL格式"""
    # OpenCV是BGR，PIL是RGB
    if len(cv2_image.shape) == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2_image
    return Image.fromarray(rgb_image)

def process_face_swap_pipeline(generated_image: Image.Image, source_image: Image.Image) -> Tuple[Image.Image, bool]:
    """
    执行完整的换脸流程
    
    Args:
        generated_image: AI生成的图像 (PIL格式)
        source_image: 用户上传的源图像 (PIL格式)
        
    Returns:
        (处理后的图像, 是否成功换脸)
    """
    try:
        print("🎯 Starting face swap pipeline...")
        
        # 转换为OpenCV格式
        generated_cv2 = pil_to_cv2(generated_image)
        source_cv2 = pil_to_cv2(source_image)
        
        # 检测源图像中的人脸
        print("🔍 Detecting faces in source image...")
        source_faces = detect_faces(source_cv2)
        if not source_faces:
            print("❌ No faces detected in source image")
            return generated_image, False
        
        # 选择最佳源人脸（置信度最高）
        source_face = source_faces[0]
        print(f"✅ Selected source face with confidence: {source_face.det_score:.3f}")
        
        # 检测生成图像中的人脸
        print("🔍 Detecting faces in generated image...")
        target_faces = detect_faces(generated_cv2)
        if not target_faces:
            print("❌ No faces detected in generated image")
            return generated_image, False
        
        print(f"🎯 Found {len(target_faces)} faces in generated image")
        
        # 如果有多个人脸，随机选择一个进行换脸
        if len(target_faces) > 1:
            target_face = random.choice(target_faces)
            print(f"🎲 Randomly selected face {target_faces.index(target_face) + 1} out of {len(target_faces)}")
        else:
            target_face = target_faces[0]
            print("🎯 Using the only detected face")
        
        print(f"✅ Selected target face with confidence: {target_face.det_score:.3f}")
        
        # 执行换脸
        result_cv2 = swap_face(source_face, target_face, generated_cv2)
        
        # 增强人脸质量
        print("✨ Enhancing face quality...")
        result_cv2 = enhance_faces(result_cv2)
        
        # 转换回PIL格式
        result_pil = cv2_to_pil(result_cv2)
        
        print("🎉 Face swap pipeline completed successfully!")
        return result_pil, True
        
    except Exception as e:
        print(f"❌ Face swap pipeline error: {e}")
        print(f"📝 Error traceback: {traceback.format_exc()}")
        return generated_image, False

def download_buffalo_l_model():
    """自动下载buffalo_l人脸分析模型"""
    try:
        import requests
        import zipfile
        import tempfile
        
        model_dir = MODELS_CONFIG["face_analysis"]
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            print(f"✅ buffalo_l model already exists at {model_dir}")
            return True
        
        print("📥 Downloading buffalo_l model from GitHub...")
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # buffalo_l模型文件列表
        model_files = [
            "1k3d68.onnx",
            "2d106det.onnx", 
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx"
        ]
        
        base_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        
        # 下载并解压
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            response = requests.get(base_url, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            
            tmp_file.flush()
            
            # 解压到目标目录
            with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(model_dir))
            
            os.unlink(tmp_file.name)
        
        print(f"✅ buffalo_l model downloaded to {model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download buffalo_l model: {e}")
        return False

def is_face_swap_available() -> bool:
    """检查换脸功能是否可用"""
    if not INSIGHTFACE_AVAILABLE or not GFPGAN_AVAILABLE:
        return False
    
    # 检查模型文件是否存在
    for model_name, model_path in MODELS_CONFIG.items():
        if model_name == "face_analysis":
            # buffalo_l是文件夹，检查是否存在且不为空
            if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
                print(f"❌ Missing model directory: {model_name} at {model_path}")
                print("🔄 Attempting to download buffalo_l model...")
                if not download_buffalo_l_model():
                    return False
        else:
            # 其他是文件
            if not os.path.exists(model_path):
                print(f"❌ Missing model file: {model_name} at {model_path}")
                return False
    
    return True

# 模块初始化检查
if __name__ == "__main__":
    print("🔍 Checking face swap availability...")
    if is_face_swap_available():
        print("✅ Face swap functionality is available")
    else:
        print("❌ Face swap functionality is not available") 