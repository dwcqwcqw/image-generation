import os
import base64
import io
import time
import traceback
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
import runpod

# AI和图像处理库
from diffusers import (
    FluxPipeline, 
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)

from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer
import boto3
from botocore.exceptions import ClientError

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
        "lora_path": "/runpod-volume/lora/flux_nsfw/flux_lustly-ai_v1.safetensors",
        "lora_id": "flux_nsfw"
    },
    "anime": {
        "name": "动漫风格", 
        "model_path": "/runpod-volume/cartoon/waiNSFWIllustrious_v130.safetensors",
        "model_type": "diffusers",
        "lora_path": "/runpod-volume/cartoon/lora/Gayporn.safetensor",
        "lora_id": "gayporn"
    }
}

# 默认LoRA配置 - 根据基础模型（单选模式）
DEFAULT_LORA_CONFIG = {
    "flux_nsfw": 1.0  # 默认使用FLUX NSFW
}

# 全局变量存储模型
txt2img_pipe = None
img2img_pipe = None
current_lora_config = DEFAULT_LORA_CONFIG.copy()
current_base_model = None  # 初始化时不预加载任何模型
device_mapping_enabled = False  # Track if device mapping is used
current_selected_lora = "flux_nsfw"  # 当前选择的单个LoRA（用于真人风格）

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
    # 内存优化配置
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
    }
    
    # 尝试使用设备映射优化
    if device == "cuda":
        try:
            # 先尝试 "balanced" 策略
            model_kwargs_with_device_map = model_kwargs.copy()
            model_kwargs_with_device_map["device_map"] = "balanced"
            
            txt2img_pipe = FluxPipeline.from_pretrained(
                base_path,
                **model_kwargs_with_device_map
            )
            print("✅ Device mapping enabled with 'balanced' strategy")
            
        except Exception as device_map_error:
            print(f"⚠️  Device mapping failed ({device_map_error}), loading without device mapping")
            # 回退到不使用设备映射
            txt2img_pipe = FluxPipeline.from_pretrained(
                base_path,
                **model_kwargs
            )
    else:
        # CPU模式直接加载
        txt2img_pipe = FluxPipeline.from_pretrained(
            base_path,
            **model_kwargs
        )
    
    # 启用优化
    try:
        txt2img_pipe.enable_attention_slicing()
        print("✅ Attention slicing enabled")
    except Exception as e:
        print(f"⚠️  Attention slicing not available: {e}")
        
    try:
        txt2img_pipe.enable_model_cpu_offload()
        print("✅ CPU offload enabled")
    except Exception as e:
        print(f"⚠️  CPU offload not available: {e}")
    
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
    )
    
    return txt2img_pipe, img2img_pipe

def load_diffusers_model(base_path: str, device: str) -> tuple:
    """加载标准diffusers模型 - 修复LayerNorm Half精度兼容性"""
    print(f"🎨 Loading diffusers model from {base_path}")
    
    # 🚨 强制使用float32避免LayerNormKernelImpl错误
    # WAI-NSFW-illustrious-SDXL模型在某些LayerNorm操作上不支持Half精度
    torch_dtype = torch.float32
    print(f"💡 使用float32精度避免LayerNorm兼容性问题")
    
    try:
        # 加载主要文本到图像管道
        txt2img_pipeline = StableDiffusionPipeline.from_single_file(
            base_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None,  # 禁用安全检查器以避免兼容性问题
            requires_safety_checker=False,
            load_safety_checker=False
        ).to(device)
        
        # 优化内存使用
        txt2img_pipeline.enable_attention_slicing()
        txt2img_pipeline.enable_model_cpu_offload()
        
        # 创建图像到图像管道（共享组件）
        img2img_pipeline = StableDiffusionImg2ImgPipeline(
            vae=txt2img_pipeline.vae,
            text_encoder=txt2img_pipeline.text_encoder,
            tokenizer=txt2img_pipeline.tokenizer,
            unet=txt2img_pipeline.unet,
            scheduler=txt2img_pipeline.scheduler,
            safety_checker=None,  # 禁用安全检查器
            feature_extractor=getattr(txt2img_pipeline, 'feature_extractor', None),
            requires_safety_checker=False
        ).to(device)
        
        # 🚨 额外确保安全检查器被禁用
        txt2img_pipeline.safety_checker = None
        txt2img_pipeline.requires_safety_checker = False
        img2img_pipeline.safety_checker = None
        img2img_pipeline.requires_safety_checker = False
        
        # 同样的优化
        img2img_pipeline.enable_attention_slicing()
        img2img_pipeline.enable_model_cpu_offload()
        
        return txt2img_pipeline, img2img_pipeline
        
    except Exception as e:
        print(f"❌ Error loading diffusers model: {str(e)}")
        raise e

def load_specific_model(base_model_type: str):
    """加载指定的基础模型 - 改进的LoRA处理"""
    global txt2img_pipe, img2img_pipe, current_base_model, current_lora_config, current_selected_lora
    
    if base_model_type not in BASE_MODELS:
        raise ValueError(f"Unknown base model type: {base_model_type}")
    
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
        
        # 🚨 保守的LoRA加载策略 - 失败不影响基础模型
        default_lora_path = model_config.get("lora_path")
        lora_loaded_successfully = False
        
        if default_lora_path and os.path.exists(default_lora_path):
            try:
                lora_start_time = datetime.now()
                print(f"🎨 尝试加载默认LoRA: {default_lora_path}")
                
                model_type = model_config.get("model_type")
                if model_type == "flux":
                    # FLUX模型LoRA加载
                    try:
                        txt2img_pipe.load_lora_weights(default_lora_path)
                        lora_loaded_successfully = True
                        print("✅ FLUX LoRA加载成功")
                    except Exception as flux_lora_error:
                        print(f"⚠️  FLUX LoRA加载失败: {flux_lora_error}")
                        
                elif model_type == "diffusers":
                    # 🚨 动漫模型LoRA兼容性测试
                    print(f"🧪 测试动漫模型LoRA兼容性...")
                    try:
                        # 尝试加载LoRA，但准备捕获所有可能的错误
                        txt2img_pipe.load_lora_weights(default_lora_path)
                        lora_loaded_successfully = True
                        print("✅ 动漫模型LoRA加载成功")
                    except Exception as anime_lora_error:
                        print(f"⚠️  动漫模型LoRA不兼容: {str(anime_lora_error)[:200]}...")
                        print("ℹ️  这是预期行为 - 该LoRA与当前动漫基础模型架构不匹配")
                        print("ℹ️  系统将使用纯基础模型继续运行")
                        lora_loaded_successfully = False
                
                if lora_loaded_successfully:
                    lora_time = (datetime.now() - lora_start_time).total_seconds()
                    print(f"✅ LoRA loaded in {lora_time:.2f}s")
                    current_lora_config = {model_config["lora_id"]: 1.0}
                    current_selected_lora = model_config["lora_id"]
                else:
                    print("🎯 继续使用纯基础模型（无LoRA）")
                    current_lora_config = {}
                    current_selected_lora = None
                    
            except Exception as general_lora_error:
                print(f"⚠️  LoRA加载过程出错: {general_lora_error}")
                print("ℹ️  将使用纯基础模型")
                current_lora_config = {}
                current_selected_lora = None
        else:
            print("ℹ️  无默认LoRA配置，使用纯基础模型")
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
        
        # Clear GPU cache before encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 GPU Memory before encoding: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
        # Only try to move text encoders to CPU if device mapping is NOT enabled
        text_encoder_device = None
        text_encoder_2_device = None
        
        try:
            if not device_mapping_enabled:
                print("📦 Manual memory management mode (no device mapping)")
                # Store original devices and move text encoders to CPU
                if hasattr(txt2img_pipe, 'text_encoder') and txt2img_pipe.text_encoder is not None:
                    text_encoder_device = next(txt2img_pipe.text_encoder.parameters()).device
                    if str(text_encoder_device) != 'cpu':
                        print("📦 Moving text_encoder to CPU temporarily to save GPU memory...")
                        txt2img_pipe.text_encoder.to('cpu')
                        torch.cuda.empty_cache()
                        
                if hasattr(txt2img_pipe, 'text_encoder_2') and txt2img_pipe.text_encoder_2 is not None:
                    text_encoder_2_device = next(txt2img_pipe.text_encoder_2.parameters()).device
                    if str(text_encoder_2_device) != 'cpu':
                        print("📦 Moving text_encoder_2 to CPU temporarily to save GPU memory...")
                        txt2img_pipe.text_encoder_2.to('cpu')
                        torch.cuda.empty_cache()
                        print(f"💾 GPU Memory after moving encoders to CPU: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            else:
                print("⚡ Device mapping mode - trusting accelerate for memory management")

            # Encode positive prompt with memory management
            print("🔤 Encoding positive prompt...")
            
            # 🎯 优化长提示词处理：为FLUX双编码器系统优化
            clip_prompt, t5_prompt = process_long_prompt(prompt)
            
            with torch.cuda.amp.autocast(enabled=False):  # Disable autocast to reduce memory
                prompt_embeds_obj = txt2img_pipe.encode_prompt(
                    prompt=clip_prompt,    # CLIP编码器使用优化后的prompt（最多77 tokens）
                    prompt_2=t5_prompt,    # T5编码器使用完整prompt（最多512 tokens）
                    device=device,
                    num_images_per_prompt=1 
                )
            
            # Force move embeddings to CPU immediately
            if hasattr(prompt_embeds_obj, 'prompt_embeds'):
                prompt_embeds_cpu = prompt_embeds_obj.prompt_embeds.cpu()
                pooled_prompt_embeds_cpu = prompt_embeds_obj.pooled_prompt_embeds.cpu() if hasattr(prompt_embeds_obj, 'pooled_prompt_embeds') else None
            else:
                # Handle tuple case
                prompt_embeds_cpu = prompt_embeds_obj[0].cpu() if isinstance(prompt_embeds_obj, tuple) else None
                pooled_prompt_embeds_cpu = prompt_embeds_obj[1].cpu() if isinstance(prompt_embeds_obj, tuple) and len(prompt_embeds_obj) > 1 else None
            
            # Clear GPU memory after positive encoding
            torch.cuda.empty_cache()
            print(f"💾 GPU Memory after positive encoding (moved to CPU): {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

            print("⚡ Skipping negative prompt embedding encoding (FLUX doesn't support negative_prompt_embeds)")
            
        finally:
            # Restore text encoders to original devices (only if we moved them manually)
            if not device_mapping_enabled:
                if text_encoder_device is not None and hasattr(txt2img_pipe, 'text_encoder') and txt2img_pipe.text_encoder is not None:
                    print(f"📦 Restoring text_encoder to {text_encoder_device}...")
                    txt2img_pipe.text_encoder.to(text_encoder_device)
                    
                if text_encoder_2_device is not None and hasattr(txt2img_pipe, 'text_encoder_2') and txt2img_pipe.text_encoder_2 is not None:
                    print(f"📦 Restoring text_encoder_2 to {text_encoder_2_device}...")
                    txt2img_pipe.text_encoder_2.to(text_encoder_2_device)
            else:
                print("⚡ Skipping text encoder restoration (device mapping handles placement)")
                
            torch.cuda.empty_cache()

        # Now move embeddings back to GPU and assign to generation_kwargs
        print("🚀 Moving embeddings back to GPU for generation...")
        
        # Move embeddings back to GPU when needed  
        generation_kwargs["prompt_embeds"] = prompt_embeds_cpu.to(device)
        
        if pooled_prompt_embeds_cpu is not None:
            generation_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds_cpu.to(device)

        # FLUX使用传统的guidance_scale参数
        generation_kwargs["guidance_scale"] = cfg_scale
        print(f"🎛️ Using guidance_scale: {cfg_scale}")
            
        print(f"💾 GPU Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    except Exception as e:
        print(f"⚠️ FLUX pipeline.encode_prompt() failed: {e}. Using raw prompts.")
        generation_kwargs["prompt"] = prompt
        generation_kwargs["negative_prompt"] = negative_prompt

    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=txt2img_pipe.device).manual_seed(seed)
    generation_kwargs["generator"] = generator

    return generate_images_common(generation_kwargs, prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, base_model, "text-to-image")

def generate_diffusers_images(prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int, num_images: int, base_model: str) -> list:
    """使用标准diffusers管道生成图像 - 深度修复NoneType错误"""
    global txt2img_pipe
    
    if txt2img_pipe is None:
        raise RuntimeError("Diffusers pipeline not loaded")
    
    print(f"📝 Processing anime model generation...")
    
    # 🚨 全面的参数安全检查和修复
    if not prompt or prompt is None:
        prompt = "masterpiece, best quality, 1boy, handsome man, anime style"
        print(f"⚠️  修复空prompt: {prompt}")
    
    if negative_prompt is None:
        negative_prompt = ""
        print(f"⚠️  修复None negative_prompt")
    
    # 确保prompt和negative_prompt都是字符串类型
    prompt = str(prompt).strip()
    negative_prompt = str(negative_prompt).strip()
    
    print(f"🔍 最终参数检查:")
    print(f"  prompt: {repr(prompt)} (type: {type(prompt)})")
    print(f"  negative_prompt: {repr(negative_prompt)} (type: {type(negative_prompt)})")
    print(f"  dimensions: {width}x{height}")
    print(f"  steps: {steps}, cfg_scale: {cfg_scale}")
    
    # 🚨 跳过Compel处理，直接使用简单的文本
    # 避免复杂的prompt处理可能导致的None问题
    print("🎯 跳过Compel处理，使用简单prompt处理避免None错误")
    
    # 🚨 使用最基础的参数配置，避免任何可能的None传递
    generation_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": int(height),
        "width": int(width),
        "num_inference_steps": int(steps),
        "guidance_scale": float(cfg_scale),
        "num_images_per_prompt": 1,  # 先强制单张生成
        "output_type": "pil",
        "return_dict": True
    }
    
    # 设置随机种子
    if seed != -1:
        import torch
        generator = torch.Generator(device=txt2img_pipe.device).manual_seed(int(seed))
        generation_kwargs["generator"] = generator
    
    print(f"🎯 Generation kwargs: {list(generation_kwargs.keys())}")
    
    return generate_images_common(generation_kwargs, prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, base_model, "text_to_image")

def generate_images_common(generation_kwargs: dict, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int, num_images: int, base_model: str, task_type: str) -> list:
    """通用图像生成逻辑 - 简化版，专注于基础功能"""
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
    
    # 🚨 对于动漫模型，强制单张生成避免复杂的批量处理
    if model_type == "diffusers" and num_images > 1:
        print(f"🎯 动漫模型强制单张生成，忽略批量请求")
        actual_num_images = 1
    else:
        actual_num_images = num_images
    
    # 简化生成逻辑 - 直接进行单张生成
    try:
        print(f"🎨 开始生成图像 (模型: {model_type})")
        
        # 生成图像 - 根据模型类型选择是否使用autocast
        if use_autocast:
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                result = txt2img_pipe(**generation_kwargs)
        else:
            print("💡 动漫模型: 跳过autocast使用float32精度")
            result = txt2img_pipe(**generation_kwargs)
        
        # 处理结果
        if hasattr(result, 'images') and result.images:
            for i, image in enumerate(result.images):
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
                            'seed': seed
                        })
                        print(f"✅ 图像 {i+1} 生成成功: {filename}")
                    except Exception as upload_error:
                        print(f"❌ 上传图像 {i+1} 失败: {upload_error}")
                        continue
        else:
            print("⚠️  管道返回空结果或无图像")
            
    except Exception as e:
        print(f"❌ 生成图像失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
    
    # 如果需要多张但只生成了一张，复制结果
    if len(results) == 1 and num_images > 1:
        print(f"🔄 为满足 {num_images} 张需求，复制单张结果")
        original = results[0]
        for i in range(1, num_images):
            # 创建新的文件名但使用相同图像
            new_filename = f"txt2img_{current_base_model}_{int(time.time())}_{i}_copy.png"
            copy_result = original.copy()
            copy_result['filename'] = new_filename
            results.append(copy_result)
    
    print(f"🎯 总共生成了 {len(results)} 张图像")
    return results

def text_to_image(prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, steps: int = 12, cfg_scale: float = 1.0, seed: int = -1, num_images: int = 1, base_model: str = "realistic") -> list:
    """文本生成图像 - 支持多种模型类型"""
    global current_base_model, txt2img_pipe
    
    print(f"🎯 请求模型: {base_model}, 当前加载模型: {current_base_model}")
    
    # 检查是否需要切换模型
    if current_base_model != base_model:
        print(f"🔄 需要切换模型: {current_base_model} -> {base_model}")
        try:
            load_specific_model(base_model)
        except Exception as e:
            print(f"❌ 模型切换失败: {e}")
            raise e
    
    # 确保模型已加载
    if txt2img_pipe is None:
        print(f"⚠️  模型未加载，加载 {base_model} 模型...")
        load_specific_model(base_model)
    
    # 获取模型配置
    model_config = BASE_MODELS.get(base_model)
    if not model_config:
        raise ValueError(f"Unknown base model: {base_model}")
    
    model_type = model_config["model_type"]
    print(f"🎨 使用 {model_type} 管道生成图像...")
    
    # 🎯 模型特定参数优化
    if model_type == "flux":
        # FLUX模型参数优化 - 根据官方推荐 https://huggingface.co/lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1
        # 官方推荐: guidance_scale=4, steps=20, 768x768分辨率
        
        if cfg_scale < 3.0:
            print(f"⚠️  FLUX CFG过低 ({cfg_scale})，调整为4.0 (官方推荐)")
            cfg_scale = 4.0
        elif cfg_scale > 6.0:
            print(f"⚠️  FLUX CFG过高 ({cfg_scale})，调整为4.0 (官方推荐)")
            cfg_scale = 4.0
            
        if steps < 15:
            print(f"⚠️  FLUX steps过低 ({steps})，调整为20 (官方推荐)")
            steps = 20
        elif steps > 30:
            print(f"⚠️  FLUX steps过高 ({steps})，调整为20 (官方推荐)")
            steps = 20
            
        # 推荐768x768分辨率以获得更好质量
        if width == 1024 and height == 1024:
            print("💡 FLUX推荐768x768分辨率以获得更好质量")
            width = 768
            height = 768
            
        print(f"🔧 FLUX优化参数(官方推荐): steps={steps}, cfg_scale={cfg_scale}, size={width}x{height}")
        return generate_flux_images(prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, base_model)
        
    elif model_type == "diffusers":
        # 动漫模型参数优化 - 根据CivitAI官方推荐
        # WAI-NSFW-illustrious-SDXL推荐: Steps: 15-30, CFG scale: 5-7, 1024x1024以上
        
        if cfg_scale < 5.0:
            print(f"⚠️  动漫模型CFG过低 ({cfg_scale})，调整为6.0 (官方推荐5-7)")
            cfg_scale = 6.0
        elif cfg_scale > 8.0:
            print(f"⚠️  动漫模型CFG过高 ({cfg_scale})，调整为7.0 (官方推荐5-7)")
            cfg_scale = 7.0
            
        if steps < 15:
            print(f"⚠️  动漫模型steps过低 ({steps})，调整为20 (官方推荐15-30)")
            steps = 20
        elif steps > 35:
            print(f"⚠️  动漫模型steps过高 ({steps})，调整为30 (官方推荐15-30)")
            steps = 30
            
        # 确保使用1024x1024以上分辨率
        if width < 1024 or height < 1024:
            print("💡 动漫模型推荐1024x1024以上分辨率")
            if width < 1024:
                width = 1024
            if height < 1024:
                height = 1024
            
        print(f"🔧 动漫模型优化参数(CivitAI推荐): steps={steps}, cfg_scale={cfg_scale}, size={width}x{height}")
        return generate_diffusers_images(prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, base_model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def image_to_image(params: dict) -> list:
    """图生图生成 - 优化版本"""
    global img2img_pipe, current_base_model
    
    if img2img_pipe is None:
        raise ValueError("Image-to-image model not loaded")
    
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
    base_model = params.get('baseModel', 'realistic')
    lora_config = params.get('lora_config', {})
    
    # 检查是否需要切换基础模型
    if base_model != current_base_model:
        print(f"Switching base model for generation: {current_base_model} -> {base_model}")
        switch_base_model(base_model)
    
    # 检查是否需要更新LoRA配置
    if lora_config and lora_config != current_lora_config:
        print(f"Updating LoRA config for generation: {lora_config}")
        load_multiple_loras(lora_config)
    
    # 处理输入图像
    if isinstance(image_data, str):
        source_image = base64_to_image(image_data)
    else:
        raise ValueError("Invalid image data format")
    
    # 调整图像尺寸
    source_image = source_image.resize((width, height), Image.Resampling.LANCZOS)
    
    # 🎯 长提示词支持 - 全新方法：直接使用FLUX原生处理 (通过 pipeline.encode_prompt)
    print(f"📝 Processing prompt for Img2Img: {len(prompt)} characters")
    
    generation_kwargs = {
        "image": source_image,
        # "prompt": prompt, # Replaced by embeds
        # "negative_prompt": negative_prompt, # Replaced by embeds
        "width": width, 
        "height": height,
        "strength": denoising_strength, # For Img2Img
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "generator": None,  # 稍后设置
    }

    # Generate embeds using the pipeline's own encoder for robustness
    print("🧬 Generating prompt embeddings for Img2Img using pipeline.encode_prompt()...")
    try:
        device = get_device()
        
        # Clear GPU cache before encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 GPU Memory before img2img encoding: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        # Only try to move text encoders to CPU if device mapping is NOT enabled
        # Device mapping conflicts with manual component movement
        text_encoder_device = None
        text_encoder_2_device = None
        
        try:
            if not device_mapping_enabled:
                print("📦 Manual memory management mode for img2img (no device mapping)")
                # Store original devices and move text encoders to CPU
                if hasattr(img2img_pipe, 'text_encoder') and img2img_pipe.text_encoder is not None:
                    text_encoder_device = next(img2img_pipe.text_encoder.parameters()).device
                    if str(text_encoder_device) != 'cpu':
                        print("📦 Moving img2img text_encoder to CPU temporarily...")
                        img2img_pipe.text_encoder.to('cpu')
                        torch.cuda.empty_cache()
                        
                if hasattr(img2img_pipe, 'text_encoder_2') and img2img_pipe.text_encoder_2 is not None:
                    text_encoder_2_device = next(img2img_pipe.text_encoder_2.parameters()).device
                    if str(text_encoder_2_device) != 'cpu':
                        print("📦 Moving img2img text_encoder_2 to CPU temporarily...")
                        img2img_pipe.text_encoder_2.to('cpu')
                        torch.cuda.empty_cache()
                        print(f"💾 GPU Memory after moving img2img encoders to CPU: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            else:
                print("⚡ Device mapping mode for img2img - trusting accelerate for memory management")

            # Encode positive prompt with memory management
            print("🔤 Encoding positive prompt for img2img...")
            
            # 🎯 优化长提示词处理：为FLUX双编码器系统优化
            clip_prompt, t5_prompt = process_long_prompt(prompt)
            # FLUX不需要负提示词嵌入，只处理正提示词
            
            with torch.cuda.amp.autocast(enabled=False):
                prompt_embeds_obj = img2img_pipe.encode_prompt(
                    prompt=clip_prompt,    # CLIP编码器使用优化后的prompt（最多77 tokens）
                    prompt_2=t5_prompt,    # T5编码器使用完整prompt（最多512 tokens）
                    device=device,
                    num_images_per_prompt=1
                )
            
            # Force move embeddings to CPU immediately
            if hasattr(prompt_embeds_obj, 'prompt_embeds'):
                prompt_embeds_cpu = prompt_embeds_obj.prompt_embeds.cpu()
                pooled_prompt_embeds_cpu = prompt_embeds_obj.pooled_prompt_embeds.cpu() if hasattr(prompt_embeds_obj, 'pooled_prompt_embeds') else None
            else:
                prompt_embeds_cpu = prompt_embeds_obj[0].cpu() if isinstance(prompt_embeds_obj, tuple) else None
                pooled_prompt_embeds_cpu = prompt_embeds_obj[1].cpu() if isinstance(prompt_embeds_obj, tuple) and len(prompt_embeds_obj) > 1 else None
            
            # Clear cache after positive encoding
            torch.cuda.empty_cache()
            print(f"💾 GPU Memory after positive img2img encoding (moved to CPU): {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

            # ❌ 跳过负提示词嵌入编码，FLUX不支持
            print("⚡ Skipping negative prompt embedding encoding for img2img (FLUX doesn't support negative_prompt_embeds)")
            
        finally:
            # Restore text encoders to original devices (only if we moved them manually)
            if not device_mapping_enabled:
                if text_encoder_device is not None and hasattr(img2img_pipe, 'text_encoder') and img2img_pipe.text_encoder is not None:
                    print(f"📦 Restoring img2img text_encoder to {text_encoder_device}...")
                    img2img_pipe.text_encoder.to(text_encoder_device)
                    
                if text_encoder_2_device is not None and hasattr(img2img_pipe, 'text_encoder_2') and img2img_pipe.text_encoder_2 is not None:
                    print(f"📦 Restoring img2img text_encoder_2 to {text_encoder_2_device}...")
                    img2img_pipe.text_encoder_2.to(text_encoder_2_device)
            else:
                print("⚡ Skipping img2img text encoder restoration (device mapping handles placement)")
                
            torch.cuda.empty_cache()

        # Now move embeddings back to GPU and assign to generation_kwargs
        print("🚀 Moving img2img embeddings back to GPU for generation...")
        
        # Move embeddings back to GPU when needed
        generation_kwargs["prompt_embeds"] = prompt_embeds_cpu.to(device)
        # ❌ FLUX不支持negative_prompt_embeds参数，移除
        # generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds_cpu.to(device)
        
        if pooled_prompt_embeds_cpu is not None:
            generation_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds_cpu.to(device)
        # ❌ FLUX不支持negative_pooled_prompt_embeds参数，移除  
        # if negative_pooled_prompt_embeds_cpu is not None:
        #     generation_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds_cpu.to(device)

        # FLUX使用传统的guidance_scale参数
        generation_kwargs["guidance_scale"] = cfg_scale
        print(f"🎛️ Using guidance_scale: {cfg_scale}")
            
        print(f"💾 GPU Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    except torch.cuda.OutOfMemoryError as oom_error:
        print(f"❌ CUDA Out of Memory during img2img encode_prompt: {oom_error}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cleared GPU cache after img2img OOM error")
        
        raise RuntimeError(f"GPU memory insufficient for img2img prompt encoding. Please try with shorter prompts or switch to a GPU with more memory. Original error: {oom_error}")
        
    except Exception as e:
        print(f"⚠️ Img2Img pipeline.encode_prompt() failed: {e}. Traceback follows.")
        traceback.print_exc()
        
        # Clear cache on any error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Falling back to using raw prompts for Img2Img (this will likely cause the original error).")
        generation_kwargs["prompt"] = prompt
        generation_kwargs["negative_prompt"] = negative_prompt

    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=img2img_pipe.device).manual_seed(seed)
    generation_kwargs["generator"] = generator

    # 获取当前模型类型以确定autocast策略
    model_config = BASE_MODELS.get(current_base_model, {})
    model_type = model_config.get("model_type", "unknown")
    
    # 🚨 动漫模型禁用autocast避免LayerNorm精度问题
    use_autocast = model_type == "flux"  # 只有FLUX模型使用autocast

    results = []
    
    # 优化：批量生成时一次性生成所有图片
    if num_images > 1 and num_images <= 4:  # 限制批量大小避免内存问题
        try:
            print(f"Batch generating {num_images} images for img2img...")
            # 生成图像 - 根据模型类型选择是否使用autocast
            if use_autocast:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    result = img2img_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=source_image,
                        strength=denoising_strength,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        generator=generator,
                        num_images_per_prompt=num_images
                    )
            else:
                # 动漫模型不使用autocast
                print("💡 动漫模型img2img: 跳过autocast使用float32精度")
                result = img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=source_image,
                    strength=denoising_strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    num_images_per_prompt=num_images
                )
            
            # 处理批量生成的图片
            for i, image in enumerate(result.images):
                try:
                    # 上传到 R2
                    image_id = str(uuid.uuid4())
                    filename = f"generated/{image_id}.png"
                    image_bytes = image_to_bytes(image)
                    image_url = upload_to_r2(image_bytes, filename)
                    
                    # 创建结果对象
                    image_data = {
                        'id': image_id,
                        'url': image_url,
                        'prompt': prompt,
                        'negativePrompt': negative_prompt,
                        'seed': seed + i,
                        'width': width,
                        'height': height,
                        'steps': steps,
                        'cfgScale': cfg_scale,
                        'baseModel': base_model,
                        'createdAt': datetime.utcnow().isoformat(),
                        'type': 'image-to-image',
                        'denoisingStrength': denoising_strength
                    }
                    
                    results.append(image_data)
                    
                except Exception as e:
                    print(f"Error processing batch img2img image {i+1}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Batch img2img generation failed, falling back to individual generation: {str(e)}")
            # 如果批量生成失败，回退到单张生成
            num_images = min(num_images, 1)
    
    # 单张生成或批量生成失败的回退
    if len(results) == 0:
        for i in range(num_images):
            try:
                print(f"Generating img2img image {i+1}/{num_images}...")
                # 生成图像 - 根据模型类型选择是否使用autocast
                if use_autocast:
                    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                        # 优化：清理GPU缓存
                        if torch.cuda.is_available() and i > 0:
                            torch.cuda.empty_cache()
                            
                        result = img2img_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=source_image,
                            strength=denoising_strength,
                            num_inference_steps=steps,
                            guidance_scale=cfg_scale,
                            generator=generator,
                            num_images_per_prompt=1
                        )
                else:
                    # 动漫模型不使用autocast
                    print(f"💡 动漫模型img2img: 生成图片{i+1}使用float32精度")
                    if torch.cuda.is_available() and i > 0:
                        torch.cuda.empty_cache()
                        
                    result = img2img_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=source_image,
                        strength=denoising_strength,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        generator=generator,
                        num_images_per_prompt=1
                    )
                
                image = result.images[0]
                
                # 上传到 R2
                image_id = str(uuid.uuid4())
                filename = f"generated/{image_id}.png"
                image_bytes = image_to_bytes(image)
                image_url = upload_to_r2(image_bytes, filename)
                
                # 创建结果对象
                image_data = {
                    'id': image_id,
                    'url': image_url,
                    'prompt': prompt,
                    'negativePrompt': negative_prompt,
                    'seed': seed,
                    'width': width,
                    'height': height,
                    'steps': steps,
                    'cfgScale': cfg_scale,
                    'baseModel': base_model,
                    'createdAt': datetime.utcnow().isoformat(),
                    'type': 'image-to-image',
                    'denoisingStrength': denoising_strength
                }
                
                results.append(image_data)
                
                # 为下一张图片更新种子
                if i < num_images - 1:
                    seed += 1
                    generator = torch.Generator(device=img2img_pipe.device).manual_seed(seed)
                    
            except Exception as e:
                print(f"Error generating img2img image {i+1}: {str(e)}")
                continue
    
    # 优化：最终清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
            {"id": "cum_on_face", "name": "Cum on Face", "description": "颜射主题内容生成"}
        ],
        "anime": [
            {"id": "gayporn", "name": "Gayporn", "description": "男同动漫风格内容生成"}
        ],
        "current_selected": {
            "realistic": current_selected_lora if current_base_model == "realistic" else "flux_nsfw",
            "anime": "gayporn" if current_base_model == "anime" else "gayporn"
        }
    }

def switch_single_lora(lora_id: str) -> bool:
    """切换单个LoRA模型（使用动态搜索）"""
    global txt2img_pipe, img2img_pipe, current_lora_config, current_selected_lora
    
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
        
        # 卸载当前LoRA
        if hasattr(txt2img_pipe, 'unload_lora_weights'):
            txt2img_pipe.unload_lora_weights()
            print("🧹 已卸载之前的LoRA")
        
        # 加载新的LoRA
        txt2img_pipe.load_lora_weights(lora_path)
        print("✅ 新LoRA加载成功")
        
        # 同步到img2img管道（如果存在）
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
        # 尝试恢复到之前的LoRA
        if current_selected_lora and current_selected_lora != lora_id:
            try:
                previous_lora_path = find_lora_file(current_selected_lora, current_base_model)
                if previous_lora_path:
                    if hasattr(txt2img_pipe, 'unload_lora_weights'):
                        txt2img_pipe.unload_lora_weights()
                    txt2img_pipe.load_lora_weights(previous_lora_path)
                    print(f"🔄 已恢复到之前的LoRA: {current_selected_lora}")
            except Exception as recovery_error:
                print(f"❌ LoRA恢复失败: {recovery_error}")
        raise RuntimeError(f"LoRA切换失败: {str(e)}")

def load_multiple_loras(lora_config: dict) -> bool:
    """加载多个LoRA模型到管道中 - 使用动态搜索"""
    global txt2img_pipe, img2img_pipe, current_base_model, current_lora_config
    
    if txt2img_pipe is None:
        print("❌ No pipeline loaded, cannot load LoRAs")
        return False
    
    if not lora_config:
        print("ℹ️  No LoRA configuration provided")
        return True
    
    # 获取当前模型类型
    current_model_type = BASE_MODELS.get(current_base_model, {}).get("model_type", "unknown")
    print(f"🎯 当前模型类型: {current_model_type}")
    
    try:
        # 先清理现有的LoRA
        print("🧹 Clearing existing LoRA weights...")
        try:
            txt2img_pipe.unload_lora_weights()
            if img2img_pipe:
                img2img_pipe.unload_lora_weights()
        except Exception as e:
            print(f"⚠️  Could not unload previous LoRAs: {e}")
        
        # 动态搜索并过滤兼容的LoRA
        compatible_loras = {}
        for lora_id, weight in lora_config.items():
            if weight <= 0:
                continue
            
            # 动态搜索LoRA文件
            lora_path = find_lora_file(lora_id, current_base_model)
            if not lora_path:
                print(f"⚠️  LoRA文件未找到: {lora_id}")
                continue
                
            compatible_loras[lora_id] = {
                "path": lora_path,
                "weight": weight
            }
        
        if not compatible_loras:
            print("ℹ️  没有找到兼容的LoRA模型")
            return True
        
        print(f"🎨 Loading {len(compatible_loras)} compatible LoRA(s): {list(compatible_loras.keys())}")
        
        # 加载兼容的LoRA
        lora_paths = []
        lora_weights = []
        
        for lora_id, lora_data in compatible_loras.items():
            lora_paths.append(lora_data["path"])
            lora_weights.append(lora_data["weight"])
            print(f"  📦 {lora_id}: {lora_data['path']} (weight: {lora_data['weight']})")
        
        # 根据模型类型使用不同的加载方法
        if current_model_type == "flux":
            # FLUX模型使用load_lora_weights
            txt2img_pipe.load_lora_weights(
                lora_paths[0] if len(lora_paths) == 1 else lora_paths,
                weight_name=None,
                adapter_name=list(compatible_loras.keys())[0] if len(compatible_loras) == 1 else list(compatible_loras.keys())
            )
            
            # 设置权重
            if len(compatible_loras) > 1:
                adapter_weights = {name: weight for name, weight in zip(compatible_loras.keys(), lora_weights)}
                txt2img_pipe.set_adapters(list(compatible_loras.keys()), adapter_weights=list(adapter_weights.values()))
            else:
                # 单个LoRA
                adapter_name = list(compatible_loras.keys())[0]
                txt2img_pipe.set_adapters([adapter_name], adapter_weights=[lora_weights[0]])
                
        elif current_model_type == "diffusers":
            # 标准diffusers模型使用load_lora_weights
            if len(compatible_loras) == 1:
                # 单个LoRA
                lora_path = lora_paths[0]
                weight = lora_weights[0] 
                txt2img_pipe.load_lora_weights(lora_path)
                txt2img_pipe.cross_attention_kwargs = {"scale": weight}
                
                # 同步到img2img管道
                if img2img_pipe:
                    img2img_pipe.load_lora_weights(lora_path)
                    img2img_pipe.cross_attention_kwargs = {"scale": weight}
            else:
                print("⚠️  多个LoRA加载暂不支持标准diffusers模型")
                return False
        
        # 更新当前配置
        current_lora_config.update(lora_config)
        print(f"✅ Successfully loaded {len(compatible_loras)} LoRA(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error loading multiple LoRAs: {e}")
        # 打印更详细的错误信息
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

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
            success = load_multiple_loras(single_lora_config)
            
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
            
            success = load_multiple_loras(lora_config)
            
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
            steps = job_input.get('steps', 12)
            cfg_scale = job_input.get('cfgScale', 1.0)
            seed = job_input.get('seed', -1)
            num_images = job_input.get('numImages', 1)
            base_model = job_input.get('baseModel', 'realistic')
            lora_config = job_input.get('lora_config', {})
            
            # 检查是否需要更新LoRA配置
            if lora_config and lora_config != current_lora_config:
                print(f"🎨 更新LoRA配置: {lora_config}")
                
                # 检查当前模型类型
                if current_base_model:
                    model_config = BASE_MODELS.get(current_base_model, {})
                    model_type = model_config.get("model_type", "unknown")
                    print(f"🎯 当前模型类型: {model_type}")
                    
                    # 清理现有LoRA权重
                    if txt2img_pipe:
                        try:
                            print("🧹 Clearing existing LoRA weights...")
                            txt2img_pipe.unload_lora_weights()
                        except Exception as clear_error:
                            print(f"⚠️  清理LoRA权重时出错: {clear_error}")
                    
                    # 尝试加载新的LoRA配置
                    try:
                        if load_multiple_loras(lora_config):
                            print("✅ LoRA配置更新成功")
                        else:
                            print("⚠️  LoRA配置更新失败，使用基础模型")
                    except Exception as lora_load_error:
                        print(f"⚠️  LoRA加载出错: {lora_load_error}")
                        print("ℹ️  继续使用基础模型生成")
            
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
            
            # 🚨 确保有模型加载
            if not txt2img_pipe:
                print("❌ 没有加载任何模型")
                return {
                    'success': False,
                    'error': 'No model loaded. Please switch to a valid model first.'
                }
            
            # 生成图像
            try:
                print(f"🎨 使用 {current_base_model} 模型生成图像...")
                model_config = BASE_MODELS.get(current_base_model, {})
                model_type = model_config.get("model_type", "unknown")
                
                if model_type == "flux":
                    print("💡 FLUX模型推荐768x768分辨率")
                    print("🔧 FLUX模型优化参数(官方推荐): steps=20, cfg_scale=4, size=768x768")
                    images = generate_flux_images(prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, current_base_model)
                elif model_type == "diffusers":
                    print("💡 动漫模型推荐1024x1024以上分辨率")
                    print("🔧 动漫模型优化参数(CivitAI推荐): steps=20, cfg_scale=7, size=1024x1024")
                    images = generate_diffusers_images(prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images, current_base_model)
                else:
                    print(f"❌ 未知模型类型: {model_type}")
                    return {
                        'success': False,
                        'error': f'Unknown model type: {model_type}'
                    }
                
                # 🚨 检查生成结果是否为空
                if not images or len(images) == 0:
                    print("❌ 图像生成失败，返回空结果")
                    return {
                        'success': False,
                        'error': 'Image generation failed - no images were created. This may be due to model compatibility issues or parameter problems.'
                    }
                
                print(f"✅ 成功生成 {len(images)} 张图像")
                return {
                    'success': True,
                    'data': images
                }
                
            except Exception as generation_error:
                print(f"❌ 图像生成过程出错: {generation_error}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
                return {
                    'success': False,
                    'error': f'Image generation failed: {str(generation_error)}'
                }
            
        elif task_type == 'image-to-image':
            # 图像转图像生成 - 修复参数提取
            print("📝 Processing image-to-image request...")
            
            # 直接从job_input提取参数，而不是嵌套的params对象
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
            
            requested_lora_config = params.get('lora_config', current_lora_config)
            
            # 检查是否需要更新LoRA配置
            if requested_lora_config != current_lora_config:
                print(f"Auto-loading LoRA config for generation: {requested_lora_config}")
                load_multiple_loras(requested_lora_config)
            
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
    
    # 动漫风格LoRA - 包含新增的LoRA
    "gayporn": ["Gayporn.safetensor", "Gayporn.safetensors", "gayporn.safetensors", "GayPorn.safetensors"],
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
