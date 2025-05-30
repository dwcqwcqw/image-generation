import runpod
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from PIL import Image
import base64
import io
import json
import os
import uuid
from datetime import datetime
import boto3
from botocore.config import Config
import sys
import traceback

# 导入compel用于处理长提示词
try:
    from compel import Compel
    COMPEL_AVAILABLE = True
    print("✓ Compel library loaded for long prompt support")
except ImportError:
    COMPEL_AVAILABLE = False
    print("⚠️  Compel library not available - long prompt support limited")

# 兼容性修复：为旧版本PyTorch添加get_default_device函数
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        """Fallback implementation for torch.get_default_device()"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    # 将函数添加到torch模块中
    torch.get_default_device = get_default_device
    print("✓ Added fallback torch.get_default_device() function")

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

# 基础模型配置
BASE_MODELS = {
    "realistic": {
        "name": "真人风格",
        "base_path": "/runpod-volume/flux_base",
        "lora_path": "/runpod-volume/lora/flux_nsfw",
        "lora_id": "flux_nsfw"
    },
    "anime": {
        "name": "动漫风格",
        "base_path": "/runpod-volume/cartoon/waiNSFWIllustrious_v130.safetensors",
        "lora_path": "/runpod-volume/cartoon/lora/Gayporn.safetensor",
        "lora_id": "gayporn"
    }
}

# 支持的LoRA模型列表 - 更新为支持不同基础模型
AVAILABLE_LORAS = {
    "flux_nsfw": {
        "name": "FLUX NSFW",
        "path": "/runpod-volume/lora/flux_nsfw",
        "description": "NSFW真人内容生成模型",
        "default_weight": 1.0,
        "base_model": "realistic"
    },
    "gayporn": {
        "name": "Gayporn",
        "path": "/runpod-volume/cartoon/lora/Gayporn.safetensor",
        "description": "NSFW动漫内容生成模型",
        "default_weight": 1.0,
        "base_model": "anime"
    },
    # 保留其他LoRA以备扩展使用
    "UltraRealPhoto": {
        "name": "Ultra Real Photo",
        "path": "/runpod-volume/lora/UltraRealPhoto.safetensors",
        "description": "Ultra realistic photo generation",
        "default_weight": 1.0,
        "base_model": "realistic"
    },
    "Chastity_Cage": {
        "name": "Chastity Cage",
        "path": "/runpod-volume/lora/Chastity_Cage.safetensors",
        "description": "Chastity device focused generation",
        "default_weight": 0.5,
        "base_model": "realistic"
    },
    "DynamicPenis": {
        "name": "Dynamic Penis",
        "path": "/runpod-volume/lora/DynamicPenis.safetensors",
        "description": "Dynamic male anatomy generation",
        "default_weight": 0.5,
        "base_model": "realistic"
    },
    "OnOff": {
        "name": "On Off",
        "path": "/runpod-volume/lora/OnOff.safetensors",
        "description": "Clothing on/off variations",
        "default_weight": 0.5,
        "base_model": "realistic"
    },
    "Puppy_mask": {
        "name": "Puppy Mask",
        "path": "/runpod-volume/lora/Puppy_mask.safetensors",
        "description": "Puppy mask and pet play content",
        "default_weight": 0.5,
        "base_model": "realistic"
    },
    "asianman": {
        "name": "Asian Man",
        "path": "/runpod-volume/lora/asianman.safetensors",
        "description": "Asian male character generation",
        "default_weight": 0.5,
        "base_model": "realistic"
    },
    "butt-and-feet": {
        "name": "Butt and Feet",
        "path": "/runpod-volume/lora/butt-and-feet.safetensors",
        "description": "Focus on lower body parts",
        "default_weight": 0.5,
        "base_model": "realistic"
    },
    "cumshots": {
        "name": "Cumshots",
        "path": "/runpod-volume/lora/cumshots.safetensors",
        "description": "Adult climax content generation",
        "default_weight": 0.5,
        "base_model": "realistic"
    }
}

# 默认LoRA配置 - 根据基础模型
DEFAULT_LORA_CONFIG = {
    "flux_nsfw": 1.0
}

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

# 全局变量存储模型
txt2img_pipe = None
img2img_pipe = None
current_lora_config = DEFAULT_LORA_CONFIG.copy()
current_base_model = "realistic"  # 当前加载的基础模型

# 全局变量存储compel处理器
compel_proc = None
compel_proc_neg = None

def get_device():
    """获取设备，兼容不同PyTorch版本"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_models():
    """加载 FLUX 模型 - 大幅性能优化版本"""
    global txt2img_pipe, img2img_pipe, current_base_model
    
    print("🚀 Loading FLUX models with optimizations...")
    start_time = datetime.now()
    
    # 默认加载真人风格模型
    base_model_type = "realistic"
    load_specific_model(base_model_type)

def load_specific_model(base_model_type: str):
    """加载指定的基础模型"""
    global txt2img_pipe, img2img_pipe, current_base_model
    
    if base_model_type not in BASE_MODELS:
        raise ValueError(f"Unknown base model type: {base_model_type}")
    
    model_config = BASE_MODELS[base_model_type]
    base_path = model_config["base_path"]
    
    print(f"🎨 Loading {model_config['name']} model from {base_path}")
    start_time = datetime.now()
    
    # CUDA兼容性检查和修复
    if torch.cuda.is_available():
        try:
            # 测试CUDA是否可用
            test_tensor = torch.tensor([1.0]).cuda()
            print("✓ CUDA test successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  CUDA error detected: {e}")
            print("⚠️  Falling back to CPU mode")
            # 强制使用CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # 检查 CUDA 可用性
    device = get_device()
    print(f"📱 Using device: {device}")
    
    # GPU内存优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"💾 GPU Memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    
    try:
        # 🎯 优化1: 使用低内存模式和优化配置
        print("⚡ Loading text-to-image pipeline with optimizations...")
        
        # 内存优化配置
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,  # 低CPU内存使用
        }
        
        # 尝试使用设备映射优化 - 修复兼容性问题
        device_mapping_used = False  # 标志跟踪是否使用设备映射
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
                device_mapping_used = True
                
            except Exception as device_map_error:
                print(f"⚠️  Device mapping failed ({device_map_error}), loading without device mapping")
                # 回退到不使用设备映射
                txt2img_pipe = FluxPipeline.from_pretrained(
                    base_path,
                    **model_kwargs
                )
                device_mapping_used = False
        else:
            # CPU模式直接加载
            txt2img_pipe = FluxPipeline.from_pretrained(
                base_path,
                **model_kwargs
            )
            device_mapping_used = False
        
        loading_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  Base model loaded in {loading_time:.2f}s")
        
        # 🎯 优化2: 启用内存高效注意力
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
        
        # 🎯 优化3: VAE内存优化
        try:
            txt2img_pipe.enable_vae_slicing()
            txt2img_pipe.enable_vae_tiling()
            print("✅ VAE optimizations enabled")
        except Exception as e:
            print(f"⚠️  VAE optimizations not available: {e}")
        
        # 加载对应的默认 LoRA 权重
        lora_start_time = datetime.now()
        default_lora_path = model_config["lora_path"]
        if os.path.exists(default_lora_path):
            print(f"🎨 Loading default LoRA for {model_config['name']}: {default_lora_path}")
            try:
                txt2img_pipe.load_lora_weights(default_lora_path)
                lora_time = (datetime.now() - lora_start_time).total_seconds()
                print(f"✅ LoRA loaded in {lora_time:.2f}s")
                
                # 更新当前LoRA配置
                global current_lora_config
                current_lora_config = {model_config["lora_id"]: 1.0}
                
            except ValueError as e:
                if "PEFT backend is required" in str(e):
                    print("❌ ERROR: PEFT backend is required for LoRA support")
                    print("   Please install: pip install peft>=0.8.0")
                    raise RuntimeError("PEFT library is required but not installed")
                else:
                    print(f"❌ ERROR: Failed to load LoRA weights: {e}")
                    raise RuntimeError(f"Failed to load required LoRA model: {e}")
            except Exception as e:
                print(f"❌ ERROR: Failed to load LoRA weights: {e}")
                raise RuntimeError(f"Failed to load required LoRA model: {e}")
        else:
            print(f"❌ ERROR: Default LoRA weights not found at {default_lora_path}")
            raise RuntimeError(f"Required LoRA model not found for {model_config['name']}")
        
        # 🎯 优化4: 智能设备移动（仅在未使用设备映射时）
        if not device_mapping_used:
            device_start_time = datetime.now()
            print("🚚 Moving pipeline to device...")
            
            if device == "cuda":
                # 渐进式移动到GPU，避免内存峰值
                txt2img_pipe = txt2img_pipe.to(device)
            else:
                txt2img_pipe = txt2img_pipe.to(device)
            
            device_time = (datetime.now() - device_start_time).total_seconds()
            print(f"✅ Device transfer completed in {device_time:.2f}s")
        else:
            print("⚡ Skipping manual device transfer (using device mapping)")
        
        # 🎯 优化5: 图生图模型使用共享组件 (零拷贝)
        print("🔗 Creating image-to-image pipeline (sharing components)...")
        img_start_time = datetime.now()
        
        img2img_pipe = FluxImg2ImgPipeline(
            vae=txt2img_pipe.vae,
            text_encoder=txt2img_pipe.text_encoder,
            text_encoder_2=txt2img_pipe.text_encoder_2,
            tokenizer=txt2img_pipe.tokenizer,
            tokenizer_2=txt2img_pipe.tokenizer_2,
            transformer=txt2img_pipe.transformer,
            scheduler=txt2img_pipe.scheduler,
        )
        
        # 不需要再次移动到设备，因为共享组件已经在设备上
        img_time = (datetime.now() - img_start_time).total_seconds()
        print(f"✅ Image-to-image pipeline created in {img_time:.2f}s")
        
        # 更新当前基础模型
        current_base_model = base_model_type
        
        # 最终内存状态
        if torch.cuda.is_available():
            print(f"💾 GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            print(f"💾 GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"🎉 {model_config['name']} model loaded successfully in {total_time:.2f}s!")
        
        # 🎯 优化6: 预热推理 (可选)
        try:
            print("🔥 Warming up models with test inference...")
            warmup_start = datetime.now()
            with torch.no_grad():
                # 小尺寸预热推理
                test_result = txt2img_pipe(
                    prompt="test",
                    width=512,
                    height=512,
                    num_inference_steps=1,
                    guidance_scale=1.0
                )
            warmup_time = (datetime.now() - warmup_start).total_seconds()
            print(f"✅ Model warmup completed in {warmup_time:.2f}s")
        except Exception as e:
            print(f"⚠️  Model warmup failed (不影响正常使用): {e}")
        
        print(f"🚀 {model_config['name']} system ready for image generation!")
        
        # compel_proc and advanced long prompt support is handled by pipeline.encode_prompt directly
        # No need for separate Compel instances here for basic embedding generation

    except Exception as e:
        print(f"❌ Error loading {model_config['name']} model: {str(e)}")
        traceback.print_exc()
        raise e

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
        
        # 构建正确的公共 URL 格式 - Cloudflare R2 public URL format
        # 优先使用自定义公共域名（推荐，可避免CORS问题）
        if CLOUDFLARE_R2_PUBLIC_DOMAIN:
            public_url = f"{CLOUDFLARE_R2_PUBLIC_DOMAIN.rstrip('/')}/{filename}"
            print(f"✓ Successfully uploaded to (custom domain): {public_url}")
        else:
            # 回退到标准R2格式
            # 正确格式: https://{bucket}.{account_id}.r2.cloudflarestorage.com/{filename}
            # 从endpoint URL中提取account ID
            account_id = CLOUDFLARE_R2_ENDPOINT.split('//')[1].split('.')[0]
            public_url = f"https://{CLOUDFLARE_R2_BUCKET}.{account_id}.r2.cloudflarestorage.com/{filename}"
            print(f"✓ Successfully uploaded to (standard R2): {public_url}")
        
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

def text_to_image(params: dict) -> list:
    """文生图生成 - 优化版本 with long prompt support"""
    global txt2img_pipe, current_base_model
    
    if txt2img_pipe is None:
        raise ValueError("Text-to-image model not loaded")
    
    # 提取参数
    prompt = params.get('prompt', '')
    negative_prompt = params.get('negativePrompt', '')
    width = params.get('width', 512)
    height = params.get('height', 512)
    steps = params.get('steps', 20)
    cfg_scale = params.get('cfgScale', 7.0)
    seed = params.get('seed', -1)
    num_images = params.get('numImages', 1)
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
    
    # FLUX模型原生支持长提示词，不需要复杂的embedding处理
    generation_kwargs = {
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "generator": None,  # 稍后设置
    }

    # Generate embeds using the pipeline's own encoder for robustness
    print("🧬 Generating prompt embeddings using pipeline.encode_prompt()...")
    try:
        device = get_device()

        # Encode positive prompt
        prompt_embeds_obj = txt2img_pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1, # Encode for a single image initially
            do_classifier_free_guidance=cfg_scale > 1.0, # Still relevant for how embeds might be used/conditioned
        )

        # Encode negative prompt
        # Note: Some encode_prompt versions might not need do_classifier_free_guidance for negative prompts
        # or expect it to be False. For safety, keeping it consistent or specific to positive.
        negative_prompt_embeds_obj = txt2img_pipe.encode_prompt(
            prompt=negative_prompt if negative_prompt else "", # Pass empty string if no negative prompt
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=cfg_scale > 1.0, # Or False, depending on pipeline requirements
        )

        # Unpack positive embeddings
        if hasattr(prompt_embeds_obj, 'prompt_embeds'):
            generation_kwargs["prompt_embeds"] = prompt_embeds_obj.prompt_embeds
            if hasattr(prompt_embeds_obj, 'pooled_prompt_embeds'): # FLUX uses pooled
                 generation_kwargs["pooled_prompt_embeds"] = prompt_embeds_obj.pooled_prompt_embeds
        elif isinstance(prompt_embeds_obj, tuple) and len(prompt_embeds_obj) >= 1: # Basic embed, possibly pooled as second element
            generation_kwargs["prompt_embeds"] = prompt_embeds_obj[0]
            if len(prompt_embeds_obj) > 1 and prompt_embeds_obj[1] is not None: # Check for pooled
                generation_kwargs["pooled_prompt_embeds"] = prompt_embeds_obj[1]
        elif isinstance(prompt_embeds_obj, dict):
            generation_kwargs["prompt_embeds"] = prompt_embeds_obj.get("prompt_embeds")
            if "pooled_prompt_embeds" in prompt_embeds_obj:
                generation_kwargs["pooled_prompt_embeds"] = prompt_embeds_obj.get("pooled_prompt_embeds")
        else:
            raise ValueError("Unsupported output format from pipeline.encode_prompt() for positive prompt")

        # Unpack negative embeddings
        if hasattr(negative_prompt_embeds_obj, 'prompt_embeds'):
            generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds_obj.prompt_embeds
            if hasattr(negative_prompt_embeds_obj, 'pooled_prompt_embeds'): # FLUX uses pooled
                 generation_kwargs["negative_pooled_prompt_embeds"] = negative_prompt_embeds_obj.pooled_prompt_embeds
        elif isinstance(negative_prompt_embeds_obj, tuple) and len(negative_prompt_embeds_obj) >=1:
            generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds_obj[0]
            if len(negative_prompt_embeds_obj) > 1 and negative_prompt_embeds_obj[1] is not None:
                generation_kwargs["negative_pooled_prompt_embeds"] = negative_prompt_embeds_obj[1]
        elif isinstance(negative_prompt_embeds_obj, dict):
            generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds_obj.get("prompt_embeds")
            if "pooled_prompt_embeds" in negative_prompt_embeds_obj: # Note: key might be just 'pooled_prompt_embeds' from encode
                generation_kwargs["negative_pooled_prompt_embeds"] = negative_prompt_embeds_obj.get("pooled_prompt_embeds")
        else:
            raise ValueError("Unsupported output format from pipeline.encode_prompt() for negative prompt")

        print("✅ Embeddings successfully generated and assigned.")

    except Exception as e:
        print(f"⚠️ pipeline.encode_prompt() failed: {e}. Traceback follows.")
        traceback.print_exc()
        print("Falling back to using raw prompts (this will likely cause the original error with FluxPipeline).")
        generation_kwargs["prompt"] = prompt
        generation_kwargs["negative_prompt"] = negative_prompt

    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=txt2img_pipe.device).manual_seed(seed)
    generation_kwargs["generator"] = generator

    results = []
    
    # 优化：批量生成时一次性生成所有图片，而不是循环
    if num_images > 1 and num_images <= 4:  # 限制批量大小避免内存问题
        try:
            print(f"Batch generating {num_images} images...")
            # 生成图像
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                batch_kwargs = generation_kwargs.copy()
                batch_kwargs["num_images_per_prompt"] = num_images
                result = txt2img_pipe(**batch_kwargs)
            
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
                        'seed': seed + i,  # 每张图片不同的种子
                        'width': width,
                        'height': height,
                        'steps': steps,
                        'cfgScale': cfg_scale,
                        'baseModel': base_model,
                        'createdAt': datetime.utcnow().isoformat(),
                        'type': 'text-to-image'
                    }
                    
                    results.append(image_data)
                    
                except Exception as e:
                    print(f"Error processing batch image {i+1}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Batch generation failed, falling back to individual generation: {str(e)}")
            # 如果批量生成失败，回退到单张生成
            num_images = min(num_images, 1)
    
    # 单张生成或批量生成失败的回退
    if len(results) == 0:
        for i in range(num_images):
            try:
                print(f"Generating image {i+1}/{num_images}...")
                # 生成图像
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    # 优化：清理GPU缓存
                    if torch.cuda.is_available() and i > 0:
                        torch.cuda.empty_cache()
                        
                    single_kwargs = generation_kwargs.copy()
                    single_kwargs["num_images_per_prompt"] = 1
                    result = txt2img_pipe(**single_kwargs)
                
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
                    'type': 'text-to-image'
                }
                
                results.append(image_data)
                
                # 为下一张图片更新种子
                if i < num_images - 1:
                    seed += 1
                    generator = torch.Generator(device=txt2img_pipe.device).manual_seed(seed)
                    generation_kwargs["generator"] = generator
                    
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                continue
    
    # 优化：最终清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

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

        # Encode positive prompt
        prompt_embeds_obj = img2img_pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=cfg_scale > 1.0,
        )

        # Encode negative prompt
        negative_prompt_embeds_obj = img2img_pipe.encode_prompt(
            prompt=negative_prompt if negative_prompt else "",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=cfg_scale > 1.0, # Or False
        )

        # Unpack positive embeddings
        if hasattr(prompt_embeds_obj, 'prompt_embeds'):
            generation_kwargs["prompt_embeds"] = prompt_embeds_obj.prompt_embeds
            if hasattr(prompt_embeds_obj, 'pooled_prompt_embeds'):
                 generation_kwargs["pooled_prompt_embeds"] = prompt_embeds_obj.pooled_prompt_embeds
        elif isinstance(prompt_embeds_obj, tuple) and len(prompt_embeds_obj) >= 1:
            generation_kwargs["prompt_embeds"] = prompt_embeds_obj[0]
            if len(prompt_embeds_obj) > 1 and prompt_embeds_obj[1] is not None:
                generation_kwargs["pooled_prompt_embeds"] = prompt_embeds_obj[1]
        elif isinstance(prompt_embeds_obj, dict):
            generation_kwargs["prompt_embeds"] = prompt_embeds_obj.get("prompt_embeds")
            if "pooled_prompt_embeds" in prompt_embeds_obj:
                generation_kwargs["pooled_prompt_embeds"] = prompt_embeds_obj.get("pooled_prompt_embeds")
        else:
            raise ValueError("Unsupported output format from Img2Img pipeline.encode_prompt() for positive prompt")

        # Unpack negative embeddings
        if hasattr(negative_prompt_embeds_obj, 'prompt_embeds'):
            generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds_obj.prompt_embeds
            if hasattr(negative_prompt_embeds_obj, 'pooled_prompt_embeds'):
                 generation_kwargs["negative_pooled_prompt_embeds"] = negative_prompt_embeds_obj.pooled_prompt_embeds
        elif isinstance(negative_prompt_embeds_obj, tuple) and len(negative_prompt_embeds_obj) >= 1:
            generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds_obj[0]
            if len(negative_prompt_embeds_obj) > 1 and negative_prompt_embeds_obj[1] is not None:
                generation_kwargs["negative_pooled_prompt_embeds"] = negative_prompt_embeds_obj[1]
        elif isinstance(negative_prompt_embeds_obj, dict):
            generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds_obj.get("prompt_embeds")
            if "pooled_prompt_embeds" in negative_prompt_embeds_obj:
                generation_kwargs["negative_pooled_prompt_embeds"] = negative_prompt_embeds_obj.get("pooled_prompt_embeds")
        else:
            raise ValueError("Unsupported output format from Img2Img pipeline.encode_prompt() for negative prompt")

        print("✅ Img2Img Embeddings successfully generated and assigned.")

    except Exception as e:
        print(f"⚠️ Img2Img pipeline.encode_prompt() failed: {e}. Traceback follows.")
        traceback.print_exc()
        print("Falling back to using raw prompts for Img2Img (this will likely cause the original error).")
        generation_kwargs["prompt"] = prompt
        generation_kwargs["negative_prompt"] = negative_prompt

    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=img2img_pipe.device).manual_seed(seed)
    generation_kwargs["generator"] = generator

    results = []
    
    # 优化：批量生成时一次性生成所有图片
    if num_images > 1 and num_images <= 4:  # 限制批量大小避免内存问题
        try:
            print(f"Batch generating {num_images} images for img2img...")
            # 生成图像
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
                # 生成图像
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
    """获取可用的LoRA模型列表"""
    available = {}
    for lora_id, lora_info in AVAILABLE_LORAS.items():
        if os.path.exists(lora_info["path"]):
            available[lora_id] = {
                "name": lora_info["name"],
                "description": lora_info["description"],
                "default_weight": lora_info["default_weight"],
                "current_weight": current_lora_config.get(lora_id, 0.0)
            }
    return available

def load_multiple_loras(lora_config: dict) -> bool:
    """加载多个LoRA模型，每个都有自己的权重"""
    global txt2img_pipe, current_lora_config
    
    if not lora_config:
        print("No LoRA configuration provided")
        return False
    
    try:
        print(f"Loading multiple LoRAs with config: {lora_config}")
        
        # 先卸载所有现有的LoRA
        txt2img_pipe.unload_lora_weights()
        
        # 准备LoRA权重和适配器名称
        adapter_names = []
        adapter_weights = []
        
        for lora_id, weight in lora_config.items():
            if weight > 0 and lora_id in AVAILABLE_LORAS:
                lora_path = AVAILABLE_LORAS[lora_id]["path"]
                if os.path.exists(lora_path):
                    # 加载LoRA适配器
                    adapter_name = f"lora_{lora_id}"
                    txt2img_pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                    adapter_names.append(adapter_name)
                    adapter_weights.append(weight)
                    print(f"✅ Loaded LoRA {AVAILABLE_LORAS[lora_id]['name']} with weight {weight}")
                else:
                    print(f"⚠️ LoRA path not found: {lora_path}")
        
        if adapter_names:
            # 设置混合权重
            txt2img_pipe.set_adapters(adapter_names, adapter_weights)
            current_lora_config = lora_config.copy()
            print(f"✅ Successfully loaded {len(adapter_names)} LoRA adapters")
            return True
        else:
            print("❌ No valid LoRA adapters could be loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error loading multiple LoRAs: {str(e)}")
        # 尝试恢复到默认配置
        try:
            txt2img_pipe.unload_lora_weights()
            default_lora_path = AVAILABLE_LORAS["flux_nsfw"]["path"]
            txt2img_pipe.load_lora_weights(default_lora_path)
            current_lora_config = {"flux_nsfw": 1.0}
            print("Recovered to default LoRA configuration")
        except Exception as recovery_error:
            print(f"Failed to recover to default LoRA: {recovery_error}")
        return False

def switch_lora(lora_id: str) -> bool:
    """切换LoRA模型 - 优化版本"""
    global txt2img_pipe, img2img_pipe, current_lora_config
    
    if lora_id not in AVAILABLE_LORAS:
        raise ValueError(f"Unknown LoRA model: {lora_id}")
    
    lora_info = AVAILABLE_LORAS[lora_id]
    lora_path = lora_info["path"]
    
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA model not found: {lora_info['name']} at {lora_path}")
    
    # 优化：如果已经是当前LoRA，直接返回，避免不必要的重新加载
    if lora_id == current_lora_config["flux_nsfw"]:
        print(f"LoRA {lora_info['name']} is already loaded - skipping switch")
        return True
    
    try:
        print(f"Switching LoRA from {AVAILABLE_LORAS[current_lora_config['flux_nsfw']]['name']} to {lora_info['name']}")
        
        # 卸载当前LoRA
        txt2img_pipe.unload_lora_weights()
        
        # 加载新的LoRA
        txt2img_pipe.load_lora_weights(lora_path)
        
        # 更新当前LoRA
        current_lora_config["flux_nsfw"] = lora_id
        
        print(f"Successfully switched to LoRA: {lora_info['name']}")
        return True
        
    except Exception as e:
        print(f"Failed to switch LoRA: {str(e)}")
        # 尝试恢复到之前的LoRA
        try:
            previous_lora_path = AVAILABLE_LORAS[current_lora_config["flux_nsfw"]]["path"]
            txt2img_pipe.unload_lora_weights()
            txt2img_pipe.load_lora_weights(previous_lora_path)
            print(f"Recovered to previous LoRA: {AVAILABLE_LORAS[current_lora_config['flux_nsfw']]['name']}")
        except Exception as recovery_error:
            print(f"Failed to recover LoRA: {recovery_error}")
        raise RuntimeError(f"Failed to switch LoRA model: {str(e)}")

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
            # 获取可用LoRA列表
            available_loras = get_available_loras()
            return {
                'success': True,
                'data': {
                    'loras': available_loras,
                    'current_config': current_lora_config
                }
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
            # 优化：支持多LoRA配置
            params = job_input.get('params', {})
            requested_lora_config = params.get('lora_config', current_lora_config)
            
            # 检查是否需要更新LoRA配置
            if requested_lora_config != current_lora_config:
                print(f"Auto-loading LoRA config for generation: {requested_lora_config}")
                load_multiple_loras(requested_lora_config)
            
            results = text_to_image(params)
            return {
                'success': True,
                'data': results
            }
            
        elif task_type == 'image-to-image':
            # 优化：支持多LoRA配置
            params = job_input.get('params', {})
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