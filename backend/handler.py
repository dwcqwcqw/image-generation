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
FLUX_LORA_PATH = "/runpod-volume/Flux-Uncensored-V2"

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
        
        # 加载 LoRA 权重
        if os.path.exists(FLUX_LORA_PATH):
            print(f"Loading LoRA weights from {FLUX_LORA_PATH}")
            try:
                txt2img_pipe.load_lora_weights(FLUX_LORA_PATH)
                print("Loaded LoRA weights successfully")
            except ValueError as e:
                if "PEFT backend is required" in str(e):
                    print("Warning: PEFT backend not available, skipping LoRA weights loading")
                    print("Install peft library for LoRA support: pip install peft")
                else:
                    print(f"Warning: Failed to load LoRA weights: {e}")
            except Exception as e:
                print(f"Warning: Failed to load LoRA weights: {e}")
                print("Continuing without LoRA weights...")
        else:
            print(f"Warning: LoRA weights not found at {FLUX_LORA_PATH}")
            print("Model will work without LoRA weights")
        
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

def handler(job):
    """RunPod 处理函数"""
    try:
        job_input = job['input']
        task_type = job_input.get('task_type')
        
        if task_type == 'text-to-image':
            results = text_to_image(job_input.get('params', {}))
            return {
                'success': True,
                'data': results
            }
            
        elif task_type == 'image-to-image':
            results = image_to_image(job_input.get('params', {}))
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