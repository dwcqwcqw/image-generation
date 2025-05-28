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

# 环境变量
CLOUDFLARE_R2_ACCESS_KEY = os.getenv("CLOUDFLARE_R2_ACCESS_KEY")
CLOUDFLARE_R2_SECRET_KEY = os.getenv("CLOUDFLARE_R2_SECRET_KEY") 
CLOUDFLARE_R2_BUCKET = os.getenv("CLOUDFLARE_R2_BUCKET")
CLOUDFLARE_R2_ENDPOINT = os.getenv("CLOUDFLARE_R2_ENDPOINT")

# 模型路径
FLUX_BASE_PATH = "/runpod-volume/flux_base"
FLUX_LORA_BASE_PATH = "/runpod-volume"

# 支持的LoRA模型列表
AVAILABLE_LORAS = {
    "flux-uncensored-v2": {
        "name": "FLUX Uncensored V2",
        "path": "/runpod-volume/Flux-Uncensored-V2",
        "description": "Enhanced uncensored model for creative freedom"
    },
    "flux-realism": {
        "name": "FLUX Realism",
        "path": "/runpod-volume/Flux-Realism",
        "description": "Photorealistic image generation"
    },
    "flux-anime": {
        "name": "FLUX Anime",
        "path": "/runpod-volume/Flux-Anime", 
        "description": "Anime and manga style generation"
    },
    "flux-portrait": {
        "name": "FLUX Portrait",
        "path": "/runpod-volume/Flux-Portrait",
        "description": "Professional portrait generation"
    }
}

# 默认LoRA
DEFAULT_LORA = "flux-uncensored-v2"

# 初始化 Cloudflare R2 客户端
r2_client = boto3.client(
    's3',
    endpoint_url=CLOUDFLARE_R2_ENDPOINT,
    aws_access_key_id=CLOUDFLARE_R2_ACCESS_KEY,
    aws_secret_access_key=CLOUDFLARE_R2_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

# 全局变量存储模型
txt2img_pipe = None
img2img_pipe = None
current_lora = DEFAULT_LORA

def get_device():
    """获取设备，兼容不同PyTorch版本"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_models():
    """加载 FLUX 模型"""
    global txt2img_pipe, img2img_pipe
    
    print("Loading FLUX models...")
    
    # 检查 CUDA 可用性
    device = get_device()
    print(f"Using device: {device}")
    
    try:
        # 加载文生图模型，增加错误处理
        print("Loading text-to-image pipeline...")
        txt2img_pipe = FluxPipeline.from_pretrained(
            FLUX_BASE_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            # Removed variant parameter to avoid fp16 variant error
            # variant="fp16" if device == "cuda" else None
        )
        
        # 加载默认 LoRA 权重 (必选)
        default_lora_path = AVAILABLE_LORAS[DEFAULT_LORA]["path"]
        if os.path.exists(default_lora_path):
            print(f"Loading default LoRA weights: {AVAILABLE_LORAS[DEFAULT_LORA]['name']} from {default_lora_path}")
            try:
                txt2img_pipe.load_lora_weights(default_lora_path)
                print(f"Successfully loaded LoRA: {AVAILABLE_LORAS[DEFAULT_LORA]['name']}")
            except ValueError as e:
                if "PEFT backend is required" in str(e):
                    print("ERROR: PEFT backend is required for LoRA support")
                    print("Please install: pip install peft>=0.8.0")
                    raise RuntimeError("PEFT library is required but not installed")
                else:
                    print(f"ERROR: Failed to load LoRA weights: {e}")
                    raise RuntimeError(f"Failed to load required LoRA model: {e}")
            except Exception as e:
                print(f"ERROR: Failed to load LoRA weights: {e}")
                raise RuntimeError(f"Failed to load required LoRA model: {e}")
        else:
            print(f"ERROR: Default LoRA weights not found at {default_lora_path}")
            raise RuntimeError(f"Required LoRA model not found: {AVAILABLE_LORAS[DEFAULT_LORA]['name']}")
        
        # 验证其他可用的LoRA模型
        available_loras = []
        for lora_id, lora_info in AVAILABLE_LORAS.items():
            if os.path.exists(lora_info["path"]):
                available_loras.append(lora_id)
                print(f"✓ Available LoRA: {lora_info['name']}")
            else:
                print(f"✗ Missing LoRA: {lora_info['name']} at {lora_info['path']}")
        
        if len(available_loras) == 0:
            raise RuntimeError("No LoRA models found. LoRA models are required for this service.")
        
        print(f"Total available LoRA models: {len(available_loras)}")
        
        print("Moving pipeline to device...")
        txt2img_pipe = txt2img_pipe.to(device)
        
        # 加载图生图模型 (共享组件)
        print("Creating image-to-image pipeline...")
        img2img_pipe = FluxImg2ImgPipeline(
            vae=txt2img_pipe.vae,
            text_encoder=txt2img_pipe.text_encoder,
            text_encoder_2=txt2img_pipe.text_encoder_2,
            tokenizer=txt2img_pipe.tokenizer,
            tokenizer_2=txt2img_pipe.tokenizer_2,
            transformer=txt2img_pipe.transformer,
            scheduler=txt2img_pipe.scheduler,
        )
        img2img_pipe = img2img_pipe.to(device)
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def upload_to_r2(image_data: bytes, filename: str) -> str:
    """上传图片到 Cloudflare R2"""
    try:
        r2_client.put_object(
            Bucket=CLOUDFLARE_R2_BUCKET,
            Key=filename,
            Body=image_data,
            ContentType='image/png',
            ACL='public-read'
        )
        
        # 构建公共 URL
        public_url = f"{CLOUDFLARE_R2_ENDPOINT}/{CLOUDFLARE_R2_BUCKET}/{filename}"
        return public_url
        
    except Exception as e:
        print(f"Error uploading to R2: {str(e)}")
        raise e

def image_to_bytes(image: Image.Image) -> bytes:
    """将 PIL Image 转换为字节"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', quality=95)
    return buffer.getvalue()

def base64_to_image(base64_str: str) -> Image.Image:
    """将 base64 字符串转换为 PIL Image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image.convert('RGB')

def text_to_image(params: dict) -> list:
    """文生图生成"""
    global txt2img_pipe
    
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
    
    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=txt2img_pipe.device).manual_seed(seed)
    
    results = []
    
    for i in range(num_images):
        try:
            # 生成图像
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                result = txt2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
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
                'createdAt': datetime.utcnow().isoformat(),
                'type': 'text-to-image'
            }
            
            results.append(image_data)
            
            # 为下一张图片更新种子
            if i < num_images - 1:
                seed += 1
                generator = torch.Generator(device=txt2img_pipe.device).manual_seed(seed)
                
        except Exception as e:
            print(f"Error generating image {i+1}: {str(e)}")
            continue
    
    return results

def image_to_image(params: dict) -> list:
    """图生图生成"""
    global img2img_pipe
    
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
    
    # 处理输入图像
    if isinstance(image_data, str):
        source_image = base64_to_image(image_data)
    else:
        raise ValueError("Invalid image data format")
    
    # 调整图像尺寸
    source_image = source_image.resize((width, height), Image.Resampling.LANCZOS)
    
    # 设置随机种子
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device=img2img_pipe.device).manual_seed(seed)
    
    results = []
    
    for i in range(num_images):
        try:
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
            print(f"Error generating image {i+1}: {str(e)}")
            continue
    
    return results

def get_available_loras() -> dict:
    """获取可用的LoRA模型列表"""
    available = {}
    for lora_id, lora_info in AVAILABLE_LORAS.items():
        if os.path.exists(lora_info["path"]):
            available[lora_id] = {
                "name": lora_info["name"],
                "description": lora_info["description"],
                "is_current": lora_id == current_lora
            }
    return available

def switch_lora(lora_id: str) -> bool:
    """切换LoRA模型"""
    global txt2img_pipe, img2img_pipe, current_lora
    
    if lora_id not in AVAILABLE_LORAS:
        raise ValueError(f"Unknown LoRA model: {lora_id}")
    
    lora_info = AVAILABLE_LORAS[lora_id]
    lora_path = lora_info["path"]
    
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA model not found: {lora_info['name']} at {lora_path}")
    
    if lora_id == current_lora:
        print(f"LoRA {lora_info['name']} is already loaded")
        return True
    
    try:
        print(f"Switching to LoRA: {lora_info['name']}")
        
        # 卸载当前LoRA
        txt2img_pipe.unload_lora_weights()
        
        # 加载新的LoRA
        txt2img_pipe.load_lora_weights(lora_path)
        
        # 更新当前LoRA
        current_lora = lora_id
        
        print(f"Successfully switched to LoRA: {lora_info['name']}")
        return True
        
    except Exception as e:
        print(f"Failed to switch LoRA: {str(e)}")
        # 尝试恢复到之前的LoRA
        try:
            previous_lora_path = AVAILABLE_LORAS[current_lora]["path"]
            txt2img_pipe.unload_lora_weights()
            txt2img_pipe.load_lora_weights(previous_lora_path)
        except:
            pass
        raise RuntimeError(f"Failed to switch LoRA model: {str(e)}")

def handler(job):
    """RunPod 处理函数"""
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
                    'current': current_lora
                }
            }
            
        elif task_type == 'switch-lora':
            # 切换LoRA模型
            lora_id = job_input.get('lora_id')
            if not lora_id:
                return {
                    'success': False,
                    'error': 'lora_id is required'
                }
            
            switch_lora(lora_id)
            return {
                'success': True,
                'data': {
                    'current_lora': current_lora,
                    'message': f'Switched to {AVAILABLE_LORAS[current_lora]["name"]}'
                }
            }
        
        elif task_type == 'text-to-image':
            # 检查是否需要切换LoRA
            params = job_input.get('params', {})
            requested_lora = params.get('lora_model', current_lora)
            
            if requested_lora != current_lora:
                print(f"Switching LoRA from {current_lora} to {requested_lora}")
                switch_lora(requested_lora)
            
            results = text_to_image(params)
            return {
                'success': True,
                'data': results
            }
            
        elif task_type == 'image-to-image':
            # 检查是否需要切换LoRA
            params = job_input.get('params', {})
            requested_lora = params.get('lora_model', current_lora)
            
            if requested_lora != current_lora:
                print(f"Switching LoRA from {current_lora} to {requested_lora}")
                switch_lora(requested_lora)
            
            results = image_to_image(params)
            return {
                'success': True,
                'data': results
            }
            
        else:
            return {
                'success': False,
                'error': f'Unknown task type: {task_type}'
            }
            
    except Exception as e:
        print(f"Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # 在启动时加载模型
    try:
        load_models()
        print("Starting RunPod serverless...")
        
        # 启动 RunPod serverless
        runpod.serverless.start({
            "handler": handler
        })
    except Exception as e:
        print(f"Failed to start serverless: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e 