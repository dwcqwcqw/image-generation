import os
import base64
import io
import time
import traceback
import uuid
import sys  # 添加缺失的sys导入
import re  # 添加regex模块用于长prompt处理
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
import runpod

# AI和图像处理库
from diffusers import (
    FluxPipeline, 
    FluxImg2ImgPipeline,  # <-- Add this import
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,  # <-- Add import
    StableDiffusionXLImg2ImgPipeline,  # <-- Add import
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)

from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config # 添加Config导入

# 🔧 兼容性修复：添加回退的torch.get_default_device函数
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    torch.get_default_device = get_default_device
    print("✓ Added fallback torch.get_default_device() function")

# 导入compel用于处理长提示词
try:
    from compel import Compel
    COMPEL_AVAILABLE = True
    print("✓ Compel library loaded for long prompt support")
except ImportError:
    COMPEL_AVAILABLE = False
    print("⚠️  Compel library not available - long prompt support limited")

# 导入换脸集成模块
try:
    # 确保当前目录在Python路径中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 多个可能的文件位置
    possible_locations = [
        os.path.join(current_dir, 'face_swap_integration.py'),  # 同目录
        os.path.join(os.getcwd(), 'face_swap_integration.py'),  # 工作目录
        '/app/face_swap_integration.py',  # 容器绝对路径
        './face_swap_integration.py'  # 相对路径
    ]
    
    face_swap_file = None
    for location in possible_locations:
        if os.path.exists(location):
            face_swap_file = location
            break
    
    if not face_swap_file:
        raise ImportError(f"face_swap_integration.py not found in any of these locations: {possible_locations}")
    
    print(f"🔍 Loading face swap integration from: {face_swap_file}")
    from face_swap_integration import process_face_swap_pipeline, is_face_swap_available
    
    FACE_SWAP_AVAILABLE = is_face_swap_available()
    if FACE_SWAP_AVAILABLE:
        print("✓ Face swap integration loaded successfully")
    else:
        print("⚠️ Face swap models not available - face swap will be disabled")
except ImportError as e:
    FACE_SWAP_AVAILABLE = False
    print(f"⚠️ Face swap integration not available: {e}")
    print(f"📁 Current directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"📁 Working directory: {os.getcwd()}")
    try:
        print(f"📁 Files in current directory: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
        print(f"📁 Files in working directory: {os.listdir(os.getcwd())}")
    except Exception as list_error:
        print(f"📁 Could not list directory contents: {list_error}")
except Exception as e:
    FACE_SWAP_AVAILABLE = False
    print(f"⚠️ Face swap integration error: {e}")
    import traceback
    print(f"📝 Error traceback: {traceback.format_exc()}")

# 添加启动日志
print("=== Starting AI Image Generation Backend ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 验证关键环境变量
required_env_vars = [
    "CLOUDFLARE_R2_ACCESS_KEY",
    "CLOUDFLARE_R2_SECRET_KEY", 
    "CLOUDFLARE_R2_BUCKET",
    "CLOUDFLARE_R2_ENDPOINT"
]

missing_vars = []
for var in required_env_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"WARNING: Missing environment variables: {missing_vars}")
    print("Container will start but R2 upload may fail")

# 环境变量
CLOUDFLARE_R2_ACCESS_KEY = os.getenv("CLOUDFLARE_R2_ACCESS_KEY")
CLOUDFLARE_R2_SECRET_KEY = os.getenv("CLOUDFLARE_R2_SECRET_KEY") 
CLOUDFLARE_R2_BUCKET = os.getenv("CLOUDFLARE_R2_BUCKET")
CLOUDFLARE_R2_ENDPOINT = os.getenv("CLOUDFLARE_R2_ENDPOINT")
CLOUDFLARE_R2_PUBLIC_DOMAIN = os.getenv("CLOUDFLARE_R2_PUBLIC_DOMAIN")  # 可选：自定义公共域名

# 模型路径
FLUX_BASE_PATH = "/runpod-volume/flux_base"
FLUX_LORA_BASE_PATH = "/runpod-volume/lora"

# 配置基础模型类型和路径
BASE_MODELS = {
    "realistic": {
        "name": "真人风格",
        "model_path": "/runpod-volume/flux_base",
        "model_type": "flux", 
        "lora_path": None,  # 🚨 修复：不自动加载默认LoRA
        "lora_id": None     # 🚨 修复：让用户选择决定LoRA
    },
    "anime": {
        "name": "动漫风格", 
        "model_path": "/runpod-volume/cartoon/Anime_NSFW.safetensors",
        "model_type": "diffusers",
        "lora_path": None,  # 🚨 修复：不自动加载默认LoRA
        "lora_id": None     # 🚨 修复：让用户选择决定LoRA
    }
}

# 修复：移除默认LoRA配置，让用户选择决定
DEFAULT_LORA_CONFIG = {}

# 全局变量存储模型
txt2img_pipe = None
img2img_pipe = None
current_lora_config = {}  # 修复：初始化为空
current_base_model = None
device_mapping_enabled = False
current_selected_lora = None  # 修复：初始化为None

# 全局变量存储compel处理器
compel_proc = None
compel_proc_neg = None

# 支持的LoRA模型列表 - 更新为支持不同基础模型
AVAILABLE_LORAS = None
LORAS_LAST_SCAN = 0
LORAS_CACHE_DURATION = 300  # 5分钟缓存

# 初始化 Cloudflare R2 客户端
r2_client = None
if all([CLOUDFLARE_R2_ACCESS_KEY, CLOUDFLARE_R2_SECRET_KEY, CLOUDFLARE_R2_BUCKET, CLOUDFLARE_R2_ENDPOINT]):
    try:
        r2_client = boto3.client(
            's3',
            endpoint_url=CLOUDFLARE_R2_ENDPOINT,
            aws_access_key_id=CLOUDFLARE_R2_ACCESS_KEY,
            aws_secret_access_key=CLOUDFLARE_R2_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        print("✓ R2 client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize R2 client: {e}")
        r2_client = None
else:
    print("✗ R2 configuration incomplete - R2 upload will be disabled")

def get_device():
    """获取设备，兼容不同PyTorch版本"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_models():
    """按需加载模型，不预热"""
    global txt2img_pipe, img2img_pipe, current_base_model
    
    print("✓ 模型系统初始化完成，将按需加载模型")
    print(f"📝 支持的模型类型: {list(BASE_MODELS.keys())}")
    
    # 不预热任何模型，等待用户请求时加载
    current_base_model = None
    print("🎯 系统就绪，等待模型加载请求...")

def load_flux_model(base_path: str, device: str) -> tuple:
    """加载FLUX模型"""
    global device_mapping_enabled
    
    # 内存优化配置
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
    }
    
    # 🚨 禁用device mapping以避免模型切换时的device conflicts
    print("⚠️  禁用FLUX device mapping以避免模型切换冲突")
    device_mapping_enabled = False
    
    # 直接加载到指定设备，不使用device mapping
    txt2img_pipe = FluxPipeline.from_pretrained(
        base_path,
        **model_kwargs
    ).to(device)
    
    # 启用优化
    try:
        txt2img_pipe.enable_attention_slicing()
        print("✅ Attention slicing enabled")
    except Exception as e:
        print(f"⚠️  Attention slicing not available: {e}")
        
    # 🚨 跳过CPU offload以避免device conflicts
    print("⚠️  跳过FLUX CPU offload以避免device冲突")
    
    try:
        txt2img_pipe.enable_vae_slicing()
        txt2img_pipe.enable_vae_tiling()
        print("✅ VAE optimizations enabled")
    except Exception as e:
        print(f"⚠️  VAE optimizations not available: {e}")
    
    # 创建图生图管道
    print("🔗 Creating FLUX image-to-image pipeline (sharing components)...")
    img2img_pipe = FluxImg2ImgPipeline(
        vae=txt2img_pipe.vae,
        text_encoder=txt2img_pipe.text_encoder,
        text_encoder_2=txt2img_pipe.text_encoder_2,
        tokenizer=txt2img_pipe.tokenizer,
        tokenizer_2=txt2img_pipe.tokenizer_2,
        transformer=txt2img_pipe.transformer,
        scheduler=txt2img_pipe.scheduler,
    ).to(device)
    
    return txt2img_pipe, img2img_pipe

def load_diffusers_model(base_path: str, device: str) -> tuple:
    """加载标准diffusers模型 - 支持SDXL目录加载"""
    print(f"🎨 Loading diffusers model from {base_path}")
    
    model_filename = os.path.basename(base_path)
    is_anime_nsfw_model = model_filename == "Anime_NSFW.safetensors"

    if is_anime_nsfw_model:
        print(f"💡 特定配置: 为 {model_filename} 使用 StableDiffusionXLPipeline 和 float16")
        torch_dtype = torch.float16
        variant = "fp16"
        pipeline_class = StableDiffusionXLPipeline
        img2img_pipeline_class = StableDiffusionXLImg2ImgPipeline
        # 暂时禁用offload以匹配notebook行为，后续可根据内存情况调整
        enable_offload = False 
    else:
        print(f"💡 标准配置: 为 {model_filename} 使用 StableDiffusionPipeline 和 float32 (兼容性优先)")
        torch_dtype = torch.float32
        variant = None # variant不用于通用SDPipeline或目录加载
        pipeline_class = StableDiffusionPipeline
        img2img_pipeline_class = StableDiffusionImg2ImgPipeline
        enable_offload = True # 对其他模型保持offload

    print(f"💡 使用 {torch_dtype} 精度加载模型")
    
    try:
        if os.path.isdir(base_path):
            print(f"📁 检测到目录，使用from_pretrained加载模型 ({pipeline_class.__name__})")
            if pipeline_class == StableDiffusionXLPipeline:
                txt2img_pipeline = pipeline_class.from_pretrained(
                    base_path,
                    torch_dtype=torch_dtype,
                    variant=variant if variant else None,
                    use_safetensors=True,
                    # safety_checker, requires_safety_checker not valid for SDXL from_pretrained
                ).to(device)
            else:
                txt2img_pipeline = pipeline_class.from_pretrained(
                    base_path,
                    torch_dtype=torch_dtype,
                    variant=variant if variant else None, 
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
        else:
            print(f"📄 检测到单文件，使用from_single_file加载 ({pipeline_class.__name__})")
            if pipeline_class == StableDiffusionXLPipeline:
                txt2img_pipeline = pipeline_class.from_single_file(
                    base_path,
                    torch_dtype=torch_dtype,
                    variant=variant if variant else None,
                    use_safetensors=True,
                    # safety_checker, requires_safety_checker, load_safety_checker not valid for SDXL from_single_file
                ).to(device)
            else:
                txt2img_pipeline = pipeline_class.from_single_file(
                    base_path,
                    torch_dtype=torch_dtype,
                    variant=variant if variant else None, 
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False,
                    load_safety_checker=False 
                ).to(device)
        
        # 优化内存使用
        txt2img_pipeline.enable_attention_slicing()
        if enable_offload:
            print("📦 启用模型CPU Offload")
            txt2img_pipeline.enable_model_cpu_offload()
        else:
            print("🚫 模型CPU Offload已禁用 (特定于Anime_NSFW.safetensors测试)")

        if img2img_pipeline_class == StableDiffusionXLImg2ImgPipeline:
            # SDXL img2img管道不接受safety_checker参数
            img2img_pipeline = img2img_pipeline_class(
                vae=txt2img_pipeline.vae,
                text_encoder=txt2img_pipeline.text_encoder,
                text_encoder_2=txt2img_pipeline.text_encoder_2,
                tokenizer=txt2img_pipeline.tokenizer,
                tokenizer_2=txt2img_pipeline.tokenizer_2,
                unet=txt2img_pipeline.unet,
                scheduler=txt2img_pipeline.scheduler,
                feature_extractor=getattr(txt2img_pipeline, 'feature_extractor', None),
            ).to(device)
        else:
            # 标准SD img2img管道接受safety_checker参数
            img2img_pipeline = img2img_pipeline_class(
                vae=txt2img_pipeline.vae,
                text_encoder=getattr(txt2img_pipeline, 'text_encoder', None),
                text_encoder_2=getattr(txt2img_pipeline, 'text_encoder_2', None),
                tokenizer=getattr(txt2img_pipeline, 'tokenizer', None),
                tokenizer_2=getattr(txt2img_pipeline, 'tokenizer_2', None),
                unet=txt2img_pipeline.unet,
                scheduler=txt2img_pipeline.scheduler,
                safety_checker=None,
                feature_extractor=getattr(txt2img_pipeline, 'feature_extractor', None),
                requires_safety_checker=False
            ).to(device)
        
        txt2img_pipeline.safety_checker = None
        txt2img_pipeline.requires_safety_checker = False
        img2img_pipeline.safety_checker = None
        img2img_pipeline.requires_safety_checker = False
        
        img2img_pipeline.enable_attention_slicing()
        if enable_offload: # Apply to img2img pipe as well
             print("📦 为img2img管道启用模型CPU Offload")
             img2img_pipeline.enable_model_cpu_offload()
        else:
            print("🚫 img2img管道模型CPU Offload已禁用")
        
        print(f"✅ {pipeline_class.__name__} 模型加载成功: {base_path}")
        return txt2img_pipeline, img2img_pipeline  # 🚨 修复：返回正确的img2img_pipeline而不是img2img_pipe
        
    except Exception as e:
        print(f"❌ Error loading diffusers model ({pipeline_class.__name__}): {str(e)}")
        raise e

def load_specific_model(base_model_type: str):
    """加载指定的基础模型 - 修复：不自动加载LoRA"""
    global txt2img_pipe, img2img_pipe, current_base_model, current_lora_config, current_selected_lora
    
    if base_model_type not in BASE_MODELS:
        raise ValueError(f"Unknown base model type: {base_model_type}")
    
    # 🚨 彻底清理之前的模型，避免device conflicts
    if txt2img_pipe is not None:
        print("🧹 清理之前的txt2img模型...")
        try:
            del txt2img_pipe
        except:
            pass
        txt2img_pipe = None
    
    if img2img_pipe is not None:
        print("🧹 清理之前的img2img模型...")
        try:
            del img2img_pipe
        except:
            pass
        img2img_pipe = None
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU内存已清理")
    
    model_config = BASE_MODELS[base_model_type]
    device = get_device()
    
    print(f"🎯 Loading {model_config['name']} model...")
    
    try:
        model_start_time = datetime.now()
        
        # 加载基础模型
        if model_config["model_type"] == "flux":
            txt2img_pipe, img2img_pipe = load_flux_model(model_config["model_path"], device)
        elif model_config["model_type"] == "diffusers":
            txt2img_pipe, img2img_pipe = load_diffusers_model(model_config["model_path"], device)
        
        current_base_model = base_model_type
        
        # 🚨 修复：不自动加载默认LoRA，保持清洁状态，等待用户选择
        print("ℹ️  基础模型加载完成，无默认LoRA，等待用户选择LoRA")
        current_lora_config = {}
        current_selected_lora = None
        
        model_time = (datetime.now() - model_start_time).total_seconds()
        print(f"🎉 {model_config['name']} model loaded successfully in {model_time:.2f}s!")
        
        # 🚨 跳过动漫模型的预热推理，避免精度问题
        if model_config["model_type"] == "diffusers":
            print("⚡ 跳过动漫模型预热推理(避免精度兼容性问题)")
            print("✅ 动漫模型ready for generation (no warmup needed)")
        else:
            # 对FLUX模型进行预热
            try:
                print("🔥 Warming up model with test inference...")
                warmup_start = datetime.now()
                with torch.no_grad():
                    test_result = txt2img_pipe(
                        prompt="test",
                        width=512,
                        height=512,
                        num_inference_steps=1,
                        guidance_scale=1.0
                    )
                warmup_time = (datetime.now() - warmup_start).total_seconds()
                print(f"✅ Model warmup completed in {warmup_time:.2f}s")
            except Exception as warmup_error:
                print(f"⚠️  Model warmup failed, but model should still work: {warmup_error}")
        
        print(f"🚀 {model_config['name']} system ready for image generation!")
        
    except Exception as e:
        print(f"❌ Failed to load {model_config['name']} model: {e}")
        # 重置全局变量
        txt2img_pipe = None
        img2img_pipe = None
        current_base_model = None
        current_lora_config = {}
        current_selected_lora = None
        raise

def upload_to_r2(image_data: bytes, filename: str) -> str:
    """上传图片到 Cloudflare R2"""
    try:
        # 检查R2客户端是否可用
        if r2_client is None:
            raise RuntimeError("R2 client not available - check environment variables")
            
        # 确保 image_data 是 bytes 类型
        if not isinstance(image_data, bytes):
            raise TypeError(f"Expected bytes, got {type(image_data)}")
            
        # 验证文件大小
        if len(image_data) == 0:
            raise ValueError("Image data is empty")
            
        if len(image_data) > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError(f"Image too large: {len(image_data)} bytes")
            
        print(f"Uploading {len(image_data)} bytes to R2 as {filename}")
        
        r2_client.put_object(
            Bucket=CLOUDFLARE_R2_BUCKET,
            Key=filename,
            Body=image_data,
            ContentType='image/png',
            ACL='public-read'
        )
        
        # 构建正确的公共 URL 格式 - 优先使用 R2 Public Domain
        # 使用提供的 R2 public domain (最稳定的解决方案)
        r2_public_domain = os.getenv("CLOUDFLARE_R2_PUBLIC_BUCKET_DOMAIN")
        
        if r2_public_domain:
            # 使用 R2 public domain (.r2.dev 格式)
            public_url = f"https://{r2_public_domain}/{filename}"
            print(f"✓ Successfully uploaded to (R2 public domain): {public_url}")
        elif CLOUDFLARE_R2_PUBLIC_DOMAIN:
            # 备选：自定义域名
            public_url = f"{CLOUDFLARE_R2_PUBLIC_DOMAIN.rstrip('/')}/{filename}"
            print(f"✓ Successfully uploaded to (custom domain): {public_url}")
        else:
            # 最后回退到标准R2格式
            # 正确格式: https://{bucket}.{account_id}.r2.cloudflarestorage.com/{filename}
            # 从endpoint URL中提取account ID
            account_id = CLOUDFLARE_R2_ENDPOINT.split('//')[1].split('.')[0]
            public_url = f"https://{CLOUDFLARE_R2_BUCKET}.{account_id}.r2.cloudflarestorage.com/{filename}"
            print(f"✓ Successfully uploaded to (standard R2): {public_url}")
            print(f"⚠️  注意：如果出现CORS错误，建议设置 CLOUDFLARE_R2_PUBLIC_BUCKET_DOMAIN 环境变量")
        
        return public_url
        
    except Exception as e:
        print(f"✗ Error uploading to R2: {str(e)}")
        print(f"Image data type: {type(image_data)}, size: {len(image_data) if hasattr(image_data, '__len__') else 'unknown'}")
        
        # 对于演示目的，返回一个占位符URL而不是失败
        # 在生产环境中，您可能希望抛出异常
        placeholder_url = f"https://via.placeholder.com/512x512/cccccc/666666?text=Upload+Failed"
        print(f"Returning placeholder URL: {placeholder_url}")
        return placeholder_url

def image_to_bytes(image: Image.Image) -> bytes:
    """将 PIL Image 转换为字节"""
    try:
        buffer = io.BytesIO()
        # 确保图像是RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='PNG', quality=95, optimize=True)
        buffer.seek(0)  # 重置buffer位置
        return buffer.getvalue()
    except Exception as e:
        print(f"Error converting image to bytes: {str(e)}")
        raise e

def base64_to_image(base64_str: str) -> Image.Image:
    """将 base64 字符串转换为 PIL Image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image.convert('RGB')

def process_long_prompt(prompt: str, max_clip_tokens: int = 75, max_t5_tokens: int = 500) -> tuple:
    """
    处理长提示词，为FLUX的双编码器系统优化
    
    Args:
        prompt: 输入提示词
        max_clip_tokens: CLIP编码器最大token数（默认75，留2个特殊token空间）
        max_t5_tokens: T5编码器最大token数（默认500，留空间给特殊token）
    
    Returns:
        tuple: (clip_prompt, t5_prompt)
    """
    if not prompt:
        return "", ""
    
    # 🎯 更准确的token估算：考虑标点符号和特殊字符
    # 简单分词：按空格、逗号、标点符号分割
    token_pattern = r'\w+|[^\w\s]'  # 提取regex模式避免f-string中的反斜杠
    tokens = re.findall(token_pattern, prompt.lower())
    estimated_tokens = len(tokens)
    
    print(f"📏 Prompt analysis: {len(prompt)} chars, ~{estimated_tokens} tokens (improved estimation)")
    
    if estimated_tokens <= max_clip_tokens:
        # 短prompt：两个编码器都使用完整prompt
        print("✅ Short prompt: using full prompt for both CLIP and T5")
        return prompt, prompt
    else:
        # 长prompt：CLIP使用截断版本，T5使用完整版本
        if estimated_tokens <= max_t5_tokens:
            # 🎯 更智能的CLIP截断：保持完整的语义单元
            words = prompt.split()
            
            # 从前往后累积token，确保不超过限制
            clip_words = []
            current_tokens = 0
            
            for word in words:
                # 估算当前单词的token数（考虑标点符号）
                word_tokens = len(re.findall(token_pattern, word.lower()))
                
                if current_tokens + word_tokens <= max_clip_tokens:
                    clip_words.append(word)
                    current_tokens += word_tokens
                else:
                    break
            
            # 如果截断点不理想，尝试在句号或逗号处截断
            if len(clip_words) > 10:  # 只在有足够词汇时优化截断点
                for i in range(len(clip_words) - 1, max(0, len(clip_words) - 5), -1):
                    if clip_words[i].endswith(('.', ',', ';', '!')):
                        clip_words = clip_words[:i+1]
                        break
            
            clip_prompt = ' '.join(clip_words)
            clip_token_count = len(re.findall(token_pattern, clip_prompt.lower()))
            
            print(f"📝 Long prompt optimization:")
            print(f"   CLIP prompt: ~{len(clip_words)} words → {clip_token_count} tokens (safe truncation)")
            print(f"   T5 prompt: ~{estimated_tokens} tokens (full prompt)")
            return clip_prompt, prompt
        else:
            # 超长prompt：两个编码器都需要截断
            words = prompt.split()
            
            # CLIP截断
            clip_words = []
            current_tokens = 0
            for word in words:
                word_tokens = len(re.findall(token_pattern, word.lower()))
                if current_tokens + word_tokens <= max_clip_tokens:
                    clip_words.append(word)
                    current_tokens += word_tokens
                else:
                    break
            
            # T5截断
            t5_words = []
            current_tokens = 0
            for word in words:
                word_tokens = len(re.findall(token_pattern, word.lower()))
                if current_tokens + word_tokens <= max_t5_tokens:
                    t5_words.append(word)
                    current_tokens += word_tokens
                else:
                    break
            
            # 优化截断点
            if len(clip_words) > 10:
                for i in range(len(clip_words) - 1, max(0, len(clip_words) - 5), -1):
                    if clip_words[i].endswith(('.', ',', ';')):
                        clip_words = clip_words[:i+1]
                        break
                        
            if len(t5_words) > 20:
                for i in range(len(t5_words) - 1, max(0, len(t5_words) - 10), -1):
                    if t5_words[i].endswith(('.', ',', ';')):
                        t5_words = t5_words[:i+1]
                        break
            
            clip_prompt = ' '.join(clip_words)
            t5_prompt = ' '.join(t5_words)
            
            clip_token_count = len(re.findall(token_pattern, clip_prompt.lower()))
            t5_token_count = len(re.findall(token_pattern, t5_prompt.lower()))
            
            print(f"⚠️  Ultra-long prompt: both encoders truncated intelligently")
            print(f"   CLIP prompt: ~{len(clip_words)} words → {clip_token_count} tokens")
            print(f"   T5 prompt: ~{len(t5_words)} words → {t5_token_count} tokens")
            return clip_prompt, t5_prompt

def generate_flux_images(prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int, num_images: int, base_model: str) -> list:
    """FLUX模型图像生成"""
    global txt2img_pipe, device_mapping_enabled
    
    # FLUX模型原生支持长提示词，使用优化的embedding处理
    generation_kwargs = {
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "generator": None,  # 稍后设置
    }

    # Generate embeds using the pipeline's own encoder for robustness
    print("🧬 Generating FLUX prompt embeddings using pipeline.encode_prompt()...")
    try:
        device = get_device()
        
        # 🚨 修复：简化FLUX长prompt处理，避免device冲突
        print(f"💾 GPU Memory before encoding: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
        # 🎯 优化长提示词处理：为FLUX双编码器系统优化
        clip_prompt, t5_prompt = process_long_prompt(prompt)
        print(f"📝 FLUX prompt processing:")
        print(f"   CLIP prompt: {len(clip_prompt)} chars")
        print(f"   T5 prompt: {len(t5_prompt)} chars")
        
        # 🚨 修复：直接使用pipeline encode_prompt，不进行CPU/GPU切换
        with torch.cuda.amp.autocast(enabled=False):
            prompt_embeds_obj = txt2img_pipe.encode_prompt(
                prompt=clip_prompt,    # CLIP编码器使用优化后的prompt
                prompt_2=t5_prompt,    # T5编码器使用完整prompt
                device=device,
                num_images_per_prompt=1 
            )
        
        # 处理embeddings
        if hasattr(prompt_embeds_obj, 'prompt_embeds'):
            prompt_embeds = prompt_embeds_obj.prompt_embeds
            pooled_prompt_embeds = prompt_embeds_obj.pooled_prompt_embeds if hasattr(prompt_embeds_obj, 'pooled_prompt_embeds') else None
        else:
            # Handle tuple case
            prompt_embeds = prompt_embeds_obj[0] if isinstance(prompt_embeds_obj, tuple) else None
            pooled_prompt_embeds = prompt_embeds_obj[1] if isinstance(prompt_embeds_obj, tuple) and len(prompt_embeds_obj) > 1 else None
        
        # 设置embeddings到generation_kwargs
        if prompt_embeds is not None:
            generation_kwargs["prompt_embeds"] = prompt_embeds
            print("✅ FLUX prompt embeddings生成成功")
        
        if pooled_prompt_embeds is not None:
            generation_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
            print("✅ FLUX pooled embeddings生成成功")

        # FLUX使用传统的guidance_scale参数
        generation_kwargs["guidance_scale"] = cfg_scale
        print(f"🎛️ Using guidance_scale: {cfg_scale}")
        print(f"💾 GPU Memory after encoding: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    except Exception as e:
        print(f"⚠️ FLUX pipeline.encode_prompt() failed: {e}. Using raw prompts.")
        generation_kwargs["prompt"] = prompt
        # 🚨 FLUX模型不支持negative_prompt，移除此参数
        # generation_kwargs["negative_prompt"] = negative_prompt  # <-- 注释掉这行

    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=txt2img_pipe.device).manual_seed(seed)
    generation_kwargs["generator"] = generator

    return generate_images_common(generation_kwargs, prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, base_model, "text-to-image")

def generate_diffusers_images(prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int, num_images: int, base_model: str) -> list:
    """使用标准diffusers管道生成图像 - 支持长提示词处理和WAI-NSFW-illustrious-SDXL优化参数"""
    global txt2img_pipe
    import traceback  # 🚨 修复：确保traceback已导入
    
    if txt2img_pipe is None:
        raise RuntimeError("Diffusers pipeline not loaded")
    
    print(f"📝 Processing anime model generation...")
    
    # 🚨 全面的参数安全检查和修复
    if not prompt or prompt is None:
        prompt = "masterpiece, best quality, amazing quality, 1boy, handsome man, anime style"
        print(f"⚠️  修复空prompt: {prompt}")
    
    if negative_prompt is None:
        negative_prompt = ""
        print(f"⚠️  修复None negative_prompt")
    
    # 确保prompt和negative_prompt都是字符串类型
    prompt = str(prompt).strip()
    negative_prompt = str(negative_prompt).strip()
    
    # 🚨 根据CivitAI WAI-NSFW-illustrious-SDXL推荐设置
    # 强制使用1024x1024或更大尺寸
    if width < 1024 or height < 1024:
        print(f"⚠️  WAI-NSFW-illustrious-SDXL模型需要1024x1024或更大 ({width}x{height})，调整为1024x1024")
        width = max(1024, width)
        height = max(1024, height)
    
    # CFG Scale: 5-7 (CivitAI推荐)
    if cfg_scale < 5.0:
        print(f"⚠️  WAI-NSFW-illustrious-SDXL模型CFG过低 ({cfg_scale})，调整为6.0 (推荐5-7)")
        cfg_scale = 6.0
    elif cfg_scale > 7.0:
        print(f"⚠️  WAI-NSFW-illustrious-SDXL模型CFG过高 ({cfg_scale})，调整为6.5 (推荐5-7)")
        cfg_scale = 6.5
    
    # Steps: 15-30 (v14), 25-40 (older versions) - 我们使用25-30
    if steps < 15:
        print(f"⚠️  WAI-NSFW-illustrious-SDXL模型steps过低 ({steps})，调整为20 (推荐15-30)")
        steps = 20
    elif steps > 35:
        print(f"⚠️  WAI-NSFW-illustrious-SDXL模型steps过高 ({steps})，调整为30 (推荐15-30)")
        steps = 30
    
    # 🚨 修复：添加WAI-NSFW-illustrious-SDXL推荐的质量标签
    # 🚨 修复：添加推荐的负面提示
    recommended_negative = "bad quality, worst quality, worst detail, sketch, censor"
    # 🚨 修复：防止重复添加推荐negative prompt
    # 🔧 可选：如果用户没有输入任何负面提示词，可以跳过自动添加
    auto_add_negative = False  # 设为False可完全禁用自动添加
    
    if auto_add_negative and recommended_negative not in negative_prompt:
        if negative_prompt and negative_prompt.strip():
            # 如果用户有自定义负面提示，添加到推荐负面提示之后
            negative_prompt = recommended_negative + ", " + negative_prompt
        else:
            # 如果没有自定义负面提示，使用推荐的
            negative_prompt = recommended_negative
        print(f"🛡️ 添加WAI-NSFW-illustrious-SDXL推荐负面提示")
    else:
        if auto_add_negative:
            print(f"🛡️ 已包含推荐负面提示，跳过添加")
        else:
            print(f"🔧 自动添加负面提示已禁用，保持用户原始输入")
    
    print(f"🔍 最终参数检查:")
    print(f"  prompt: {repr(prompt)} (type: {type(prompt)})")
    print(f"  negative_prompt: {repr(negative_prompt)} (type: {type(negative_prompt)})")
    print(f"  dimensions: {width}x{height}")
    print(f"  steps: {steps}, cfg_scale: {cfg_scale}")
    
    # 🎯 SDXL长提示词处理 - 动漫模型避免Compel，使用智能压缩
    processed_prompt = prompt
    processed_negative_prompt = negative_prompt
    
    try:
        # 🚨 修复：使用更准确的token估算方法
        import re
        token_pattern = r'\w+|[^\w\s]'
        estimated_tokens = len(re.findall(token_pattern, prompt.lower()))
        
        # 🚨 修复：动漫模型始终使用智能压缩，避免Compel导致的黑图问题
        print(f"💡 动漫模型始终使用智能压缩模式 (估计token: {estimated_tokens})")
        
        # 压缩正向prompt
        if estimated_tokens > 75:
            print(f"📝 压缩长prompt: {estimated_tokens} tokens -> 75 tokens")
            processed_prompt = compress_prompt_to_77_tokens(processed_prompt, max_tokens=75)
            print(f"✅ prompt压缩完成")
        else:
            print("✅ prompt已在75 token限制内，无需压缩")
        
        # 压缩negative prompt
        negative_tokens = len(re.findall(r'\w+|[^\w\s]', processed_negative_prompt.lower()))
        if negative_tokens > 75:
            print(f"🔧 压缩negative prompt: {negative_tokens} tokens -> 75 tokens")
            processed_negative_prompt = compress_prompt_to_77_tokens(processed_negative_prompt, max_tokens=75)
            print(f"✅ negative prompt压缩完成")
        
        # 使用标准处理方式，避免Compel
        generation_kwargs = {
            'prompt': processed_prompt,
            'negative_prompt': processed_negative_prompt,
            'height': height,
            'width': width,
            'num_inference_steps': steps,
            'guidance_scale': cfg_scale,
            'num_images_per_prompt': 1,
            'output_type': 'pil',
            'return_dict': True
        }
        
        # 🚨 修复递归调用 - 直接使用generate_images_common统一处理
        print(f"🎨 使用 {base_model} diffusers模型生成图像...")
        print("💡 动漫模型推荐1024x1024以上分辨率")
        print("🔧 动漫模型优化参数(CivitAI推荐): steps=20, cfg_scale=6, size=1024x1024")
        
        return generate_images_common(generation_kwargs, prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, base_model, "text-to-image")
        
    except Exception as long_prompt_error:
        print(f"⚠️  智能压缩处理失败: {long_prompt_error}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        print("📝 回退到标准处理模式")
        
        generation_kwargs = {
            "prompt": processed_prompt,
            "negative_prompt": processed_negative_prompt,
            "height": int(height),
            "width": int(width),
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg_scale),
            "num_images_per_prompt": 1,
            "output_type": "pil",
            "return_dict": True
        }
        print("✅ 回退到标准处理")
        
        return generate_images_common(generation_kwargs, prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, base_model, "text-to-image")

def generate_images_common(generation_kwargs: dict, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int, num_images: int, base_model: str, task_type: str) -> list:
    """通用图像生成逻辑 - 支持真正的多张生成"""
    global txt2img_pipe, current_base_model
    
    # 🚨 修复：确保所有参数都不为None，避免NoneType错误
    if prompt is None or prompt == "":
        prompt = "masterpiece, best quality, 1boy"
        print(f"⚠️  空prompt，使用默认: {prompt}")
    if negative_prompt is None:
        negative_prompt = ""
        print(f"⚠️  negative_prompt为None，使用空字符串")
    
    print(f"🔍 Debug - prompt: {repr(prompt)}, negative_prompt: {repr(negative_prompt)}")
    
    results = []
    
    # 获取当前模型类型以确定autocast策略
    model_config = BASE_MODELS.get(current_base_model, {})
    model_type = model_config.get("model_type", "unknown")
    
    # 🚨 动漫模型禁用autocast避免LayerNorm精度问题
    use_autocast = model_type == "flux"  # 只有FLUX模型使用autocast
    
    print(f"🎨 开始生成 {num_images} 张图像 (模型: {model_type})")
    
    # 🎯 修复：循环生成真正的多张图片
    for i in range(num_images):
        try:
            # 为每张图片设置不同的随机种子
            current_generation_kwargs = generation_kwargs.copy()
            
            if seed != -1:
                # 基于原始种子生成不同的种子
                current_seed = seed + i
                import torch
                generator = torch.Generator(device=txt2img_pipe.device).manual_seed(int(current_seed))
                current_generation_kwargs["generator"] = generator
                print(f"🎲 图像 {i+1} 种子: {current_seed}")
            else:
                # 🚨 修复：为随机种子生成具体的种子值并显示
                import random
                current_seed = random.randint(0, 2147483647)  # 使用32位整数范围
                import torch
                generator = torch.Generator(device=txt2img_pipe.device).manual_seed(int(current_seed))
                current_generation_kwargs["generator"] = generator
                print(f"🎲 图像 {i+1} 种子: {current_seed} (随机生成)")
            
            # 生成图像 - 根据模型类型选择是否使用autocast
            if use_autocast:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    result = txt2img_pipe(**current_generation_kwargs)
            else:
                print(f"💡 动漫模型: 跳过autocast使用float32精度 (图像 {i+1})")
                result = txt2img_pipe(**current_generation_kwargs)
            
            # 处理结果
            if hasattr(result, 'images') and result.images and len(result.images) > 0:
                image = result.images[0]  # 取第一张图片
                if image is not None:
                    try:
                        # 上传到R2
                        filename = f"txt2img_{current_base_model}_{int(time.time())}_{i}.png"
                        image_url = upload_to_r2(image_to_bytes(image), filename)
                        
                        results.append({
                            'url': image_url,
                            'filename': filename,
                            'prompt': prompt,
                            'model': current_base_model,
                            'width': width,
                            'height': height,
                            'steps': steps,
                            'cfg_scale': cfg_scale,
                            'seed': current_seed  # 🚨 修复：总是包含具体的种子值
                        })
                        print(f"✅ 图像 {i+1}/{num_images} 生成成功: {filename}")
                    except Exception as upload_error:
                        print(f"❌ 上传图像 {i+1} 失败: {upload_error}")
                        continue
                else:
                    print(f"⚠️  图像 {i+1} 生成结果为空")
            else:
                print(f"⚠️  图像 {i+1} 管道返回空结果或无图像")
                
        except Exception as e:
            print(f"❌ 生成图像 {i+1} 失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            continue
    
    # 删除重复的日志输出 - 已在generate_images_common中统一处理
    print(f"🎯 总共成功生成了 {len(results)} 张图像")
    return results

def text_to_image(prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, steps: int = 25, cfg_scale: float = 5.0, seed: int = -1, num_images: int = 1, base_model: str = "realistic", lora_config: dict = None) -> list:
    """文本生成图像 - 支持多种模型类型"""
    global current_base_model, txt2img_pipe
    
    print(f"🎯 请求模型: {base_model}, 当前加载模型: {current_base_model}")
    
    # 🚨 修复：确保lora_config有默认值
    if lora_config is None:
        lora_config = {}
    
    # 🚨 修复：先检查模型切换，再处理LoRA配置
    # 检查模型是否需要切换
    if base_model != current_base_model:
        print(f"🎯 请求模型: {base_model}, 当前加载模型: {current_base_model}")
        print(f"🔄 需要切换模型: {current_base_model} -> {base_model}")
        try:
            load_specific_model(base_model)
            print(f"✅ 成功切换到 {base_model} 模型")
        except Exception as switch_error:
            print(f"❌ 模型切换失败: {switch_error}")
            return {
                'success': False,
                'error': f'Failed to switch to {base_model} model: {str(switch_error)}'
            }
    # 再切换LoRA
    if lora_config and isinstance(lora_config, dict) and len(lora_config) > 0:
        lora_id = next(iter(lora_config.keys()))
        print(f"🎨 切换LoRA: {lora_id}")
        switch_single_lora(lora_id)
    else:
        print("ℹ️  没有LoRA配置，使用基础模型生成")
    
    # 生成图像
    try:
        print(f"🎨 使用 {current_base_model} 模型生成图像...")
        model_config = BASE_MODELS.get(current_base_model, {})
        model_type = model_config.get("model_type", "unknown")
        
        # 🚨 修复：直接调用对应的生成函数，避免递归调用
        if model_type == "flux":
            print("🎯 调用generate_flux_images函数 (FLUX)")
            return generate_flux_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                num_images=num_images,
                base_model=current_base_model
            )
        elif model_type == "diffusers":
            print("🎯 调用generate_diffusers_images函数 (diffusers)")
            return generate_diffusers_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                num_images=num_images,
                base_model=current_base_model
            )
        else:
            print(f"❌ 未知模型类型: {model_type}")
            return {
                'success': False,
                'error': f'Unknown model type: {model_type}'
            }
        
    except Exception as generation_error:
        print(f"❌ 图像生成过程出错: {generation_error}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Image generation failed: {str(generation_error)}'
        }

def image_to_image(params: dict) -> list:
    """
    图生图生成 - 新版本，支持换脸功能
    
    对于真人模型：
    1. 先进行文生图
    2. 分析用户上传图片的人脸
    3. 将用户人脸置换到生成图片中
    4. 如果换脸失败，返回原始生成图片
    
    对于动漫模型：
    保持原有的图生图逻辑
    """
    global txt2img_pipe, img2img_pipe, current_base_model
    
    # 检查并自动加载模型
    base_model = params.get('baseModel', 'realistic')
    
    if txt2img_pipe is None or current_base_model != base_model:
        print(f"📝 模型未加载或需要切换，当前: {current_base_model} -> 请求: {base_model}")
        try:
            load_specific_model(base_model)
            print(f"✅ 成功加载模型: {base_model}")
        except Exception as model_error:
            print(f"❌ 模型加载失败: {model_error}")
            raise ValueError(f"Failed to load model '{base_model}': {str(model_error)}")
    
    # 确保模型加载成功
    if txt2img_pipe is None:
        raise ValueError("Model failed to load properly")
    
    print(f"✅ 模型已就绪: {current_base_model}")
    
    # 提取参数
    prompt = params.get('prompt', '')
    negative_prompt = params.get('negativePrompt', '')
    image_data = params.get('image', '')
    width = params.get('width', 512)
    height = params.get('height', 512)
    steps = params.get('steps', 20)
    cfg_scale = params.get('cfgScale', 7.0)
    seed = params.get('seed', -1)
    num_images = params.get('numImages', 1)
    denoising_strength = params.get('denoisingStrength', 0.7)
    lora_config = params.get('lora_config', {})
    
    # 确保prompt和negative_prompt不为None
    if prompt is None:
        prompt = ""
    if negative_prompt is None:
        negative_prompt = ""
    
    print(f"📝 图生图处理 - 提示词: {len(prompt)} 字符")
    print(f"📐 图像尺寸: {width}x{height}, 步数: {steps}, CFG: {cfg_scale}")
    
    # 检查是否需要更新LoRA配置
    if lora_config and isinstance(lora_config, dict) and len(lora_config) > 0:
        lora_id = next(iter(lora_config.keys()))
        print(f"🎨 切换LoRA: {lora_id}")
        switch_single_lora(lora_id)
    
    # 处理输入图像
    try:
        if isinstance(image_data, str):
            source_image = base64_to_image(image_data)
        else:
            raise ValueError("Invalid image data format")
    except Exception as e:
        print(f"❌ 图像解码失败: {e}")
        raise ValueError(f"Failed to decode input image: {str(e)}")
    
    # 获取当前模型类型
    model_config = BASE_MODELS.get(current_base_model, {})
    model_type = model_config.get("model_type", "unknown")
    
    print(f"🎯 当前模型类型: {model_type}")
    
    # 🚀 新逻辑：真人模型使用"文生图+换脸"，动漫模型使用传统图生图
    if current_base_model == "realistic":
        print("🎭 使用真人模型换脸流程：文生图 + 换脸")
        if not FACE_SWAP_AVAILABLE:
            print("⚠️ 换脸功能暂时不可用，将回退到文生图+基础图像处理")
        return _process_realistic_with_face_swap(
            prompt, negative_prompt, source_image, width, height, 
            steps, cfg_scale, seed, num_images, base_model
        )
    else:
        print("🎨 使用传统图生图流程")
        return _process_traditional_img2img(
            prompt, negative_prompt, source_image, width, height, 
            steps, cfg_scale, seed, num_images, denoising_strength, base_model
        )

def _process_realistic_with_face_swap(prompt: str, negative_prompt: str, source_image: Image.Image, 
                                    width: int, height: int, steps: int, cfg_scale: float, 
                                    seed: int, num_images: int, base_model: str) -> list:
    """真人模型：文生图 + 换脸流程"""
    try:
        print("🎯 第一步：使用真人模型进行文生图...")
        
        # 使用文生图生成初始图像
        txt2img_results = text_to_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            num_images=num_images,
            base_model=base_model
        )
        
        if not txt2img_results or len(txt2img_results) == 0:
            raise ValueError("Text-to-image generation failed")
        
        print(f"✅ 文生图完成，生成了 {len(txt2img_results)} 张图像")
        
        # 第二步：对每张生成的图像进行换脸处理
        print("🎭 第二步：进行换脸处理...")
        
        final_results = []
        
        for i, result_item in enumerate(txt2img_results):
            try:
                print(f"🔄 处理第 {i+1}/{len(txt2img_results)} 张图像...")
                
                # 从结果中获取图像
                if isinstance(result_item, dict) and 'url' in result_item:
                    # 如果结果包含URL，需要下载图像
                    import requests
                    response = requests.get(result_item['url'])
                    generated_image = Image.open(io.BytesIO(response.content))
                elif hasattr(result_item, 'images') and len(result_item.images) > 0:
                    generated_image = result_item.images[0]
                else:
                    print(f"⚠️ 无法从结果中提取图像，跳过第 {i+1} 张")
                    continue
                
                # 执行换脸（如果可用）
                if FACE_SWAP_AVAILABLE:
                    face_swapped_image, swap_success = process_face_swap_pipeline(
                        generated_image, source_image
                    )
                    
                    if swap_success:
                        print(f"✅ 第 {i+1} 张图像换脸成功")
                    else:
                        print(f"⚠️ 第 {i+1} 张图像换脸失败，使用原始生成图像")
                        face_swapped_image = generated_image
                        swap_success = False
                else:
                    print(f"⚠️ 换脸功能不可用，使用原始生成图像")
                    face_swapped_image = generated_image
                    swap_success = False
                
                # 上传处理后的图像
                image_id = str(uuid.uuid4())
                image_bytes = image_to_bytes(face_swapped_image)
                image_url = upload_to_r2(image_bytes, f"{image_id}.jpg")
                
                # 构建结果
                result_dict = {
                    'id': image_id,
                    'url': image_url,
                    'prompt': prompt,
                    'negativePrompt': negative_prompt,
                    'seed': seed + i if seed != -1 else torch.randint(0, 2**32 - 1, (1,)).item(),
                    'width': width,
                    'height': height,
                    'steps': steps,
                    'cfgScale': cfg_scale,
                    'createdAt': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'type': 'text-to-image-with-faceswap',  # 更新类型名称
                    'baseModel': base_model,
                    'faceSwapAvailable': FACE_SWAP_AVAILABLE,
                    'faceSwapSuccess': swap_success
                }
                
                final_results.append(result_dict)
                print(f"✅ 第 {i+1} 张图像处理完成: {image_url}")
                
            except Exception as e:
                print(f"❌ 第 {i+1} 张图像处理失败: {e}")
                continue
        
        print(f"🎉 换脸流程完成，成功处理 {len(final_results)} 张图像")
        return final_results
        
    except Exception as e:
        print(f"❌ 换脸流程失败: {e}")
        # 如果换脸流程失败，回退到传统图生图
        print("🔄 回退到传统图生图流程...")
        return _process_traditional_img2img(
            prompt, negative_prompt, source_image, width, height, 
            steps, cfg_scale, seed, num_images, 0.7, base_model
        )

def _process_traditional_img2img(prompt: str, negative_prompt: str, source_image: Image.Image,
                               width: int, height: int, steps: int, cfg_scale: float,
                               seed: int, num_images: int, denoising_strength: float, 
                               base_model: str) -> list:
    """传统图生图流程"""
    global img2img_pipe, current_base_model
    
    # 确保img2img_pipe已加载
    if img2img_pipe is None:
        print("🔄 img2img_pipe未加载，重新加载...")
        load_specific_model(current_base_model)
    
    if img2img_pipe is None:
        raise ValueError("Image-to-image pipeline failed to load")
    
    # 调整图像尺寸
    try:
        source_image = source_image.resize((width, height), Image.Resampling.LANCZOS)
        print(f"✅ 图像尺寸调整为: {width}x{height}")
    except Exception as e:
        print(f"❌ 图像尺寸调整失败: {e}")
        raise ValueError(f"Failed to resize image: {str(e)}")
    
    # 获取当前模型类型
    model_config = BASE_MODELS.get(current_base_model, {})
    model_type = model_config.get("model_type", "unknown")
    
    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=img2img_pipe.device).manual_seed(seed)
    
    # 根据模型类型使用不同的生成逻辑
    results = []
    
    try:
        if model_type == "flux":
            # FLUX模型 - 支持长提示词，不支持negative_prompt
            print("🎯 使用FLUX图生图管道")
            
            # FLUX模型压缩提示词
            if len(prompt) > 400:  # FLUX可以处理更长的提示词
                compressed_prompt = compress_prompt_to_77_tokens(prompt, max_tokens=75)
                print(f"📏 FLUX图生图提示词压缩: {len(prompt)} -> {len(compressed_prompt)} 字符")
                prompt = compressed_prompt
            
            for i in range(num_images):
                try:
                    current_seed = seed + i if seed != -1 else torch.randint(0, 2**32 - 1, (1,)).item()
                    current_generator = torch.Generator(device=img2img_pipe.device).manual_seed(current_seed)
                    
                    print(f"🖼️ 生成FLUX图生图 {i+1}/{num_images} (种子: {current_seed})")
                    
                    result = img2img_pipe(
                        prompt=prompt,
                        # 注意：FLUX不支持negative_prompt参数
                        image=source_image,
                        strength=denoising_strength,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        generator=current_generator,
                        num_images_per_prompt=1
                    )
                    
                    if hasattr(result, 'images') and len(result.images) > 0:
                        image = result.images[0]
                        # 上传到R2
                        image_id = str(uuid.uuid4())
                        image_bytes = image_to_bytes(image)
                        image_url = upload_to_r2(image_bytes, f"{image_id}.jpg")
                        
                        # 🚨 修复：返回格式与前端期望一致
                        results.append({
                            'id': image_id,  # 前端期望的字段名
                            'url': image_url,  # 前端期望的字段名
                            'prompt': prompt,
                            'negativePrompt': negative_prompt,
                            'seed': current_seed,
                            'width': width,
                            'height': height,
                            'steps': steps,
                            'cfgScale': cfg_scale,
                            'denoisingStrength': denoising_strength,
                            'createdAt': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                            'type': 'image-to-image',
                            'baseModel': base_model
                        })
                        print(f"✅ FLUX图生图 {i+1} 生成成功: {image_url}")
                    else:
                        print(f"❌ FLUX图生图 {i+1} 生成失败：无图像结果")
                        
                except Exception as e:
                    print(f"❌ FLUX图生图 {i+1} 生成失败: {e}")
                    continue
                    
        elif model_type == "diffusers":
            # SDXL/标准diffusers模型
            print("🎯 使用标准Diffusers图生图管道")
            
            # 压缩提示词
            if len(prompt) > 200:
                compressed_prompt = compress_prompt_to_77_tokens(prompt, max_tokens=75)
                print(f"📏 Diffusers图生图提示词压缩: {len(prompt)} -> {len(compressed_prompt)} 字符")
                prompt = compressed_prompt
                
            if len(negative_prompt) > 200:
                compressed_negative = compress_prompt_to_77_tokens(negative_prompt, max_tokens=75)
                print(f"📏 Diffusers图生图负面提示词压缩: {len(negative_prompt)} -> {len(compressed_negative)} 字符")
                negative_prompt = compressed_negative
            
            # 🚨 动漫模型禁用autocast避免LayerNorm精度问题
            use_autocast = model_type == "flux"  # 只有FLUX模型使用autocast
            
            for i in range(num_images):
                try:
                    current_seed = seed + i if seed != -1 else torch.randint(0, 2**32 - 1, (1,)).item()
                    current_generator = torch.Generator(device=img2img_pipe.device).manual_seed(current_seed)
                    
                    print(f"🖼️ 生成Diffusers图生图 {i+1}/{num_images} (种子: {current_seed})")
                    
                    # 根据模型类型选择是否使用autocast
                    if use_autocast:
                        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                            result = img2img_pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                image=source_image,
                                strength=denoising_strength,
                                width=width,
                                height=height,
                                num_inference_steps=steps,
                                guidance_scale=cfg_scale,
                                generator=current_generator,
                                num_images_per_prompt=1
                            )
                    else:
                        # 动漫模型不使用autocast
                        print("💡 动漫模型图生图: 使用float32精度")
                        result = img2img_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=source_image,
                            strength=denoising_strength,
                            width=width,
                            height=height,
                            num_inference_steps=steps,
                            guidance_scale=cfg_scale,
                            generator=current_generator,
                            num_images_per_prompt=1
                        )
                    
                    if hasattr(result, 'images') and len(result.images) > 0:
                        image = result.images[0]
                        # 上传到R2
                        image_id = str(uuid.uuid4())
                        image_bytes = image_to_bytes(image)
                        image_url = upload_to_r2(image_bytes, f"{image_id}.jpg")
                        
                        # 🚨 修复：返回格式与前端期望一致
                        results.append({
                            'id': image_id,  # 前端期望的字段名
                            'url': image_url,  # 前端期望的字段名
                            'prompt': prompt,
                            'negativePrompt': negative_prompt,
                            'seed': current_seed,
                            'width': width,
                            'height': height,
                            'steps': steps,
                            'cfgScale': cfg_scale,
                            'denoisingStrength': denoising_strength,
                            'createdAt': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                            'type': 'image-to-image',
                            'baseModel': base_model
                        })
                        print(f"✅ Diffusers图生图 {i+1} 生成成功: {image_url}")
                    else:
                        print(f"❌ Diffusers图生图 {i+1} 生成失败：无图像结果")
                        
                except Exception as e:
                    print(f"❌ Diffusers图生图 {i+1} 生成失败: {e}")
                    continue
        else:
            raise ValueError(f"Unsupported model type for image-to-image: {model_type}")
            
    except Exception as e:
        print(f"❌ 图生图生成过程出错: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        raise RuntimeError(f"Image-to-image generation failed: {str(e)}")
    
    if len(results) == 0:
        raise RuntimeError("No images were generated successfully")
    
    print(f"🎉 图生图完成: 成功生成 {len(results)}/{num_images} 张图像")
    return results

def get_available_loras() -> dict:
    """获取可用的LoRA模型列表 - 简化版本（前端静态显示）"""
    # 前端已经有静态列表，这里只返回基本信息
    return {
        "message": "前端使用静态LoRA列表，后端动态搜索文件",
        "search_paths": LORA_SEARCH_PATHS,
        "current_selected": current_selected_lora,
        "current_base_model": current_base_model
    }

def get_loras_by_base_model() -> dict:
    """获取按基础模型分组的LoRA列表 - 简化版本"""
    return {
        "realistic": [
            {"id": "flux_nsfw", "name": "FLUX NSFW", "description": "NSFW真人内容生成模型"},
            {"id": "chastity_cage", "name": "Chastity Cage", "description": "贞操笼主题内容生成"},
            {"id": "dynamic_penis", "name": "Dynamic Penis", "description": "动态男性解剖生成"},
            {"id": "masturbation", "name": "Masturbation", "description": "自慰主题内容生成"},
            {"id": "puppy_mask", "name": "Puppy Mask", "description": "小狗面具主题内容"},
            {"id": "butt_and_feet", "name": "Butt and Feet", "description": "臀部和足部主题内容"},
            {"id": "cumshots", "name": "Cumshots", "description": "射精主题内容生成"},
            {"id": "uncutpenis", "name": "Uncut Penis", "description": "未割包皮主题内容"},
            {"id": "doggystyle", "name": "Doggystyle", "description": "后入式主题内容"},
            {"id": "fisting", "name": "Fisting", "description": "拳交主题内容生成"},
            {"id": "on_off", "name": "On Off", "description": "穿衣/脱衣对比内容"},
            {"id": "blowjob", "name": "Blowjob", "description": "口交主题内容生成"},
            {"id": "cum_on_face", "name": "Cum on Face", "description": "颜射主题内容生成"},
            {"id": "anal_sex", "name": "Anal Sex", "description": "肛交主题内容生成"}
        ],
        "anime": [
            {"id": "gayporn", "name": "Gayporn", "description": "男同动漫风格内容生成"},
            {"id": "blowjob_handjob", "name": "Blowjob Handjob", "description": "口交和手交动漫内容"},
            {"id": "furry", "name": "Furry", "description": "兽人风格动漫内容"},
            {"id": "sex_slave", "name": "Sex Slave", "description": "性奴主题动漫内容"},
            {"id": "comic", "name": "Comic", "description": "漫画风格内容生成"},
            {"id": "glory_wall", "name": "Glory Wall", "description": "荣耀墙主题内容"},
            {"id": "multiple_views", "name": "Multiple Views", "description": "多视角动漫内容"},
            {"id": "pet_play", "name": "Pet Play", "description": "宠物扮演主题内容"}
        ],
        # 🚨 修复：移除固定的current_selected，让前端决定初始选择
        "message": "LoRA列表获取成功 - 静态配置版本"
    }

def switch_single_lora(lora_id: str) -> bool:
    """切换到单个LoRA（彻底无多LoRA/adapter_name残留，动漫模型兼容）"""
    global txt2img_pipe, img2img_pipe, current_lora_config, current_selected_lora, current_base_model

    if txt2img_pipe is None:
        raise ValueError("No pipeline loaded, cannot switch LoRA")

    # 动态搜索LoRA文件
    lora_path = find_lora_file(lora_id, current_base_model)
    if not lora_path:
        raise ValueError(f"LoRA文件未找到: {lora_id}")

    # 如果已经是当前LoRA，直接返回
    if lora_id == current_selected_lora:
        print(f"LoRA {lora_id} 已经加载 - 跳过切换")
        return True

    try:
        print(f"🔄 切换LoRA到: {lora_id}")
        print(f"📁 文件路径: {lora_path}")
        # 卸载当前LoRA（如有）
        if hasattr(txt2img_pipe, 'unload_lora_weights'):
            try:
                txt2img_pipe.unload_lora_weights()
                print("🧹 已卸载之前的LoRA (txt2img)")
            except Exception as e:
                print(f"⚠️  卸载txt2img LoRA时出错: {e}")
        # 加载新LoRA
        txt2img_pipe.load_lora_weights(lora_path)
        print("✅ 新LoRA加载成功 (txt2img)")
        # img2img管道同步
        if img2img_pipe and hasattr(img2img_pipe, 'load_lora_weights'):
            try:
                if hasattr(img2img_pipe, 'unload_lora_weights'):
                    img2img_pipe.unload_lora_weights()
                img2img_pipe.load_lora_weights(lora_path)
                print("✅ img2img管道LoRA同步成功")
            except Exception as e:
                print(f"⚠️  img2img管道LoRA同步失败: {e}")
        # 更新当前LoRA配置
        current_lora_config = {lora_id: 1.0}
        current_selected_lora = lora_id
        print(f"🎉 成功切换到LoRA: {lora_id}")
        return True
    except Exception as e:
        print(f"❌ LoRA切换失败: {str(e)}")
        # 强制清理，防止后续死锁
        if hasattr(txt2img_pipe, 'unload_lora_weights'):
            try:
                txt2img_pipe.unload_lora_weights()
            except:
                pass
        if img2img_pipe and hasattr(img2img_pipe, 'unload_lora_weights'):
            try:
                img2img_pipe.unload_lora_weights()
            except:
                pass
        raise RuntimeError(f"LoRA切换失败: {str(e)}")

def switch_base_model(base_model_type: str) -> bool:
    """切换基础模型"""
    global current_base_model
    
    if base_model_type not in BASE_MODELS:
        raise ValueError(f"Unknown base model type: {base_model_type}")
    
    if current_base_model == base_model_type:
        print(f"Base model {BASE_MODELS[base_model_type]['name']} is already loaded")
        return True
    
    try:
        print(f"Switching base model from {BASE_MODELS[current_base_model]['name']} to {BASE_MODELS[base_model_type]['name']}")
        
        # 释放当前模型内存
        global txt2img_pipe, img2img_pipe
        if txt2img_pipe is not None:
            del txt2img_pipe
        if img2img_pipe is not None:
            del img2img_pipe
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 加载新的基础模型
        load_specific_model(base_model_type)
        
        print(f"Successfully switched to {BASE_MODELS[base_model_type]['name']}")
        return True
        
    except Exception as e:
        print(f"Failed to switch base model: {str(e)}")
        # 尝试恢复到之前的模型
        try:
            load_specific_model(current_base_model)
            print(f"Recovered to previous model: {BASE_MODELS[current_base_model]['name']}")
        except Exception as recovery_error:
            print(f"Failed to recover base model: {recovery_error}")
        raise RuntimeError(f"Failed to switch base model: {str(e)}")

def handler(job):
    """RunPod 处理函数 - 优化版本"""
    try:
        job_input = job['input']
        task_type = job_input.get('task_type')
        
        if task_type == 'get-loras':
            # 获取可用LoRA列表（保持兼容性）
            available_loras = get_available_loras()
            return {
                'success': True,
                'data': {
                    'loras': available_loras,
                    'current_config': current_lora_config
                }
            }
            
        elif task_type == 'get-loras-by-model':
            # 获取按基础模型分组的LoRA列表（新的单选UI）
            loras_by_model = get_loras_by_base_model()
            return {
                'success': True,
                'data': loras_by_model
            }
            
        elif task_type == 'switch-single-lora':
            # 切换单个LoRA模型（新的单选模式）
            lora_id = job_input.get('lora_id')
            if not lora_id:
                return {
                    'success': False,
                    'error': 'lora_id is required'
                }
            
            success = switch_single_lora(lora_id)
            
            if success:
                return {
                    'success': True,
                    'data': {
                        'current_selected_lora': current_selected_lora,
                        'current_config': current_lora_config,
                        'message': f'Switched to {AVAILABLE_LORAS[lora_id]["name"]}'
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to switch to {lora_id}'
                }
            
        elif task_type == 'switch-lora':
            # 切换LoRA模型（单个LoRA兼容性支持）
            lora_id = job_input.get('lora_id')
            if not lora_id:
                return {
                    'success': False,
                    'error': 'lora_id is required'
                }
            
            # 兼容单LoRA切换
            single_lora_config = {lora_id: 1.0}
            success = switch_single_lora(lora_id)
            
            if success:
                return {
                    'success': True,
                    'data': {
                        'current_config': current_lora_config,
                        'message': f'Switched to {AVAILABLE_LORAS[lora_id]["name"]}'
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to switch to {lora_id}'
                }
        
        elif task_type == 'load-loras':
            # 加载多个LoRA模型配置
            lora_config = job_input.get('lora_config', {})
            if not lora_config:
                return {
                    'success': False,
                    'error': 'lora_config is required'
                }
            
            success = switch_single_lora(next(iter(lora_config.keys())))
            
            if success:
                return {
                    'success': True,
                    'data': {
                        'current_config': current_lora_config,
                        'message': f'Loaded {len([k for k, v in lora_config.items() if v > 0])} LoRA models'
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to load LoRA configuration'
                }
        
        elif task_type == 'text-to-image':
            # 文本转图像生成
            print("📝 Processing text-to-image request...")
            
            # 提取参数
            prompt = job_input.get('prompt', '')
            negative_prompt = job_input.get('negativePrompt', '') 
            width = job_input.get('width', 1024)
            height = job_input.get('height', 1024)
            steps = job_input.get('steps', 25)
            cfg_scale = job_input.get('cfgScale', 5.0)
            seed = job_input.get('seed', -1)
            num_images = job_input.get('numImages', 1)
            base_model = job_input.get('baseModel', 'realistic')
            lora_config = job_input.get('lora_config', {})
            
            # 🚨 修复：直接调用text_to_image，避免重复处理
            print("🎯 Handler直接调用text_to_image函数")
            results = text_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                num_images=num_images,
                base_model=base_model,
                lora_config=lora_config
            )
            
            return {
                'success': True,
                'data': results
            }
            
        elif task_type == 'image-to-image':
            # 图像转图像生成 - 支持单LoRA
            print("📝 Processing image-to-image request...")
            params = {
                'prompt': job_input.get('prompt', ''),
                'negativePrompt': job_input.get('negativePrompt', ''),
                'image': job_input.get('image', ''),
                'width': job_input.get('width', 512),
                'height': job_input.get('height', 512),
                'steps': job_input.get('steps', 20),
                'cfgScale': job_input.get('cfgScale', 7.0),
                'seed': job_input.get('seed', -1),
                'numImages': job_input.get('numImages', 1),
                'denoisingStrength': job_input.get('denoisingStrength', 0.7),
                'baseModel': job_input.get('baseModel', 'realistic'),
                'lora_config': job_input.get('lora_config', {})
            }
            # 先切换模型
            base_model = params.get('baseModel', 'realistic')
            if img2img_pipe is None or current_base_model != base_model:
                print(f"📝 Handler自动切换模型: {current_base_model} -> {base_model}")
                load_specific_model(base_model)
            # 再切换LoRA
            requested_lora_config = params.get('lora_config', current_lora_config)
            if requested_lora_config and isinstance(requested_lora_config, dict) and len(requested_lora_config) > 0:
                lora_id = next(iter(requested_lora_config.keys()))
                print(f"Auto-loading LoRA config for generation: {lora_id}")
                switch_single_lora(lora_id)
            results = image_to_image(params)
            return {
                'success': True,
                'data': results
            }
            
        elif task_type == 'switch-base-model':
            # 切换基础模型
            base_model_type = job_input.get('base_model_type')
            if not base_model_type:
                return {
                    'success': False,
                    'error': 'base_model_type is required'
                }
            
            success = switch_base_model(base_model_type)
            
            if success:
                return {
                    'success': True,
                    'data': {
                        'current_base_model': current_base_model
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to switch to {base_model_type}'
                }
        
        else:
            return {
                'success': False,
                'error': f'Unknown task type: {task_type}'
            }
            
    except Exception as e:
        print(f"Handler error: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

# Note: The serverless worker will be started by start_debug.py
# This allows for better debugging and health checks before startup 
# This allows for better debugging and health checks before startup 

# 简化的LoRA配置 - 前端静态显示，后端动态搜索文件
LORA_SEARCH_PATHS = {
    "realistic": [
        "/runpod-volume/lora",
        "/runpod-volume/lora/flux_nsfw",
        "/runpod-volume/lora/realistic"
    ],
    "anime": [
        "/runpod-volume/cartoon/lora",
        "/runpod-volume/anime/lora",
        "/runpod-volume/cartoon"
    ]
}

# LoRA名称到可能文件名的映射
LORA_FILE_PATTERNS = {
    # 真人风格LoRA
    "flux_nsfw": ["flux_nsfw", "flux_nsfw.safetensors"],
    "chastity_cage": ["Chastity_Cage.safetensors", "chastity_cage.safetensors", "ChastityCase.safetensors"],
    "dynamic_penis": ["DynamicPenis.safetensors", "dynamic_penis.safetensors"],
    "masturbation": ["Masturbation.safetensors", "masturbation.safetensors"],
    "puppy_mask": ["Puppy_mask.safetensors", "puppy_mask.safetensors", "PuppyMask.safetensors"],
    "butt_and_feet": ["butt-and-feet.safetensors", "butt_and_feet.safetensors", "ButtAndFeet.safetensors"],
    "cumshots": ["cumshots.safetensors", "Cumshots.safetensors"],
    "uncutpenis": ["uncutpenis.safetensors", "UncutPenis.safetensors", "uncut_penis.safetensors"],
    "doggystyle": ["Doggystyle.safetensors", "doggystyle.safetensors", "doggy_style.safetensors"],
    "fisting": ["Fisting.safetensors", "fisting.safetensors"],
    "on_off": ["OnOff.safetensors", "on_off.safetensors", "onoff.safetensors"],
    "blowjob": ["blowjob.safetensors", "Blowjob.safetensors", "blow_job.safetensors"],
    "cum_on_face": ["cumonface.safetensors", "cum_on_face.safetensors", "CumOnFace.safetensors"],
    "anal_sex": ["Anal_sex.safetensors", "anal_sex.safetensors", "AnalSex.safetensors", "analsex.safetensors"],
    
    # 动漫风格LoRA - 移除anime_nsfw，因为它现在是底层模型
    "gayporn": ["Gayporn.safetensors", "gayporn.safetensors", "GayPorn.safetensors"],
    "blowjob_handjob": ["Blowjob_Handjob.safetensors", "blowjob_handjob.safetensors", "BlowjobHandjob.safetensors"],
    "furry": ["Furry.safetensors", "furry.safetensors", "FURRY.safetensors"],
    "sex_slave": ["Sex_slave.safetensors", "sex_slave.safetensors", "SexSlave.safetensors"],
    "comic": ["comic.safetensors", "Comic.safetensors", "COMIC.safetensors"],
    "glory_wall": ["glory_wall.safetensors", "Glory_wall.safetensors", "GloryWall.safetensors"],
    "multiple_views": ["multiple_views.safetensors", "Multiple_views.safetensors", "MultipleViews.safetensors"],
    "pet_play": ["pet_play.safetensors", "Pet_play.safetensors", "PetPlay.safetensors"]
}

def find_lora_file(lora_id: str, base_model: str) -> str:
    """动态搜索LoRA文件路径 - 增强搜索逻辑"""
    search_paths = LORA_SEARCH_PATHS.get(base_model, [])
    file_patterns = LORA_FILE_PATTERNS.get(lora_id, [lora_id])
    
    print(f"🔍 搜索LoRA文件: {lora_id} (模型: {base_model})")
    
    for base_path in search_paths:
        if not os.path.exists(base_path):
            print(f"  ❌ 路径不存在: {base_path}")
            continue
            
        print(f"  📁 搜索目录: {base_path}")
        
        # 尝试精确匹配
        for pattern in file_patterns:
            full_path = os.path.join(base_path, pattern)
            if os.path.exists(full_path):
                print(f"  ✅ 找到文件: {full_path}")
                return full_path
        
        # 尝试模糊匹配（文件名包含lora_id）
        try:
            for filename in os.listdir(base_path):
                if filename.endswith(('.safetensors', '.safetensor', '.ckpt', '.pt')):
                    # 检查文件名是否包含lora_id的关键词
                    name_lower = filename.lower()
                    lora_lower = lora_id.lower().replace('_', '').replace('-', '')
                    
                    if lora_lower in name_lower.replace('_', '').replace('-', ''):
                        full_path = os.path.join(base_path, filename)
                        print(f"  ✅ 模糊匹配找到: {full_path}")
                        return full_path
        except Exception as e:
            print(f"  ❌ 搜索错误: {e}")
    
    print(f"  ❌ 未找到LoRA文件: {lora_id}")
    return None

# 移除复杂的动态扫描，使用简单的静态配置
# AVAILABLE_LORAS = None
# LORAS_LAST_SCAN = 0
# LORAS_CACHE_DURATION = 300  # 5分钟缓存
# 动漫模型新增LoRA列表
ANIME_ADDITIONAL_LORAS = {
    "blowjob_handjob": "/runpod-volume/cartoon/lora/Blowjob_Handjob.safetensors",
    "furry": "/runpod-volume/cartoon/lora/Furry.safetensors", 
    "sex_slave": "/runpod-volume/cartoon/lora/Sex_slave.safetensors",
    "comic": "/runpod-volume/cartoon/lora/comic.safetensors",
    "glory_wall": "/runpod-volume/cartoon/lora/glory_wall.safetensors",
    "multiple_views": "/runpod-volume/cartoon/lora/multiple_views.safetensors",
    "pet_play": "/runpod-volume/cartoon/lora/pet_play.safetensors"
}

def completely_clear_lora_adapters():
    """完全清理所有LoRA适配器 - 最彻底的清理方法"""
    global txt2img_pipe, img2img_pipe
    
    print("🧹 开始完全清理LoRA适配器...")
    
    # 清理管道列表
    pipelines = [txt2img_pipe]
    if img2img_pipe:
        pipelines.append(img2img_pipe)
    
    for i, pipe in enumerate(pipelines):
        if pipe is None:
            continue
        
        pipeline_name = "txt2img" if i == 0 else "img2img"
        
        try:
            # 第1层：标准的unload_lora_weights方法
            if hasattr(pipe, 'unload_lora_weights'):
                pipe.unload_lora_weights()
                print("✅ 标准unload_lora_weights完成")
            
            # 第2层：清理UNet中的特定LoRA配置
            if hasattr(pipe, 'unet') and pipe.unet:
                unet = pipe.unet
                
                # 清理_hf_peft_config_loaded
                if hasattr(unet, '_hf_peft_config_loaded'):
                    delattr(unet, '_hf_peft_config_loaded')
                    print("🔧 清理UNet._hf_peft_config_loaded")
                
                # 🚨 新增：清理PEFT相关的适配器缓存
                if hasattr(unet, 'peft_config') and unet.peft_config:
                    unet.peft_config.clear()
                    print("🔧 清理UNet.peft_config")
                
                # 🚨 新增：清理适配器名称缓存
                if hasattr(unet, '_lora_adapters'):
                    unet._lora_adapters.clear()
                    print("🔧 清理UNet._lora_adapters")
                
                # 🚨 新增：强制清理所有可能的适配器残留
                adapter_attrs = ['_lora_adapters', 'peft_config', '_adapter_names', '_active_adapters']
                for attr in adapter_attrs:
                    if hasattr(unet, attr):
                        try:
                            attr_obj = getattr(unet, attr)
                            if hasattr(attr_obj, 'clear'):
                                attr_obj.clear()
                            elif isinstance(attr_obj, dict):
                                attr_obj.clear()
                            elif isinstance(attr_obj, list):
                                attr_obj.clear()
                            print(f"🔧 清理UNet.{attr}")
                        except Exception as attr_error:
                            print(f"⚠️  清理UNet.{attr}时出错: {attr_error}")
                
                # 🚨 新增：清理Text Encoder适配器（如果存在）
                if hasattr(pipe, 'text_encoder') and pipe.text_encoder:
                    text_encoder = pipe.text_encoder
                    for attr in adapter_attrs:
                        if hasattr(text_encoder, attr):
                            try:
                                attr_obj = getattr(text_encoder, attr)
                                if hasattr(attr_obj, 'clear'):
                                    attr_obj.clear()
                                elif isinstance(attr_obj, dict):
                                    attr_obj.clear()
                                elif isinstance(attr_obj, list):
                                    attr_obj.clear()
                                print(f"🔧 清理TextEncoder.{attr}")
                            except Exception as attr_error:
                                print(f"⚠️  清理TextEncoder.{attr}时出错: {attr_error}")
                
                # 🚨 新增：如果有第二个Text Encoder（SDXL）
                if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2:
                    text_encoder_2 = pipe.text_encoder_2
                    for attr in adapter_attrs:
                        if hasattr(text_encoder_2, attr):
                            try:
                                attr_obj = getattr(text_encoder_2, attr)
                                if hasattr(attr_obj, 'clear'):
                                    attr_obj.clear()
                                elif isinstance(attr_obj, dict):
                                    attr_obj.clear()
                                elif isinstance(attr_obj, list):
                                    attr_obj.clear()
                                print(f"🔧 清理TextEncoder2.{attr}")
                            except Exception as attr_error:
                                print(f"⚠️  清理TextEncoder2.{attr}时出错: {attr_error}")
                
        except Exception as pipeline_error:
            print(f"⚠️  清理{pipeline_name}管道时出错: {pipeline_error}")
    
    # 第3层：强制清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU内存已清理")
    
    print("✅ LoRA适配器完全清理完成")

def compress_prompt_to_77_tokens(prompt: str, max_tokens: int = 75) -> str:
    """
    智能压缩prompt到指定token数量以内 - 使用CLIPTokenizer真实计数
    """
    import re
    # 初始化tokenizer（只初始化一次）
    global _clip_tokenizer
    if '_clip_tokenizer' not in globals() or _clip_tokenizer is None:
        _clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = _clip_tokenizer
    # 先粗略分词
    token_pattern = r'\w+|[^\w\s]'
    words = prompt.split()
    # 循环截断，直到tokenizer计数<=max_tokens
    for end in range(len(words), 0, -1):
        candidate = ' '.join(words[:end])
        token_count = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if token_count <= max_tokens:
            print(f"✅ 智能压缩完成: {token_count} tokens (目标: {max_tokens})")
            print(f"   压缩内容: '{candidate[:100]}{'...' if len(candidate) > 100 else ''}'")
            return candidate
    # 如果全部都超，返回最短
    return ' '.join(words[:1])
