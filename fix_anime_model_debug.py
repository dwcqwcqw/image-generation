#!/usr/bin/env python3
"""
修复动漫模型NoneType错误的快速脚本
"""

import re

def fix_handler_file():
    """修复handler.py中的问题"""
    
    # 读取文件内容
    with open('backend/handler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: 在generate_images_common函数开头添加None检查
    pattern1 = r'(def generate_images_common\(.*?\) -> list:\s*"""通用图像生成逻辑"""\s*global txt2img_pipe, current_base_model\s*)'
    replacement1 = r'''\1
    # 🚨 修复：确保所有参数都不为None，避免NoneType错误
    if prompt is None or prompt == "":
        prompt = "masterpiece, best quality, 1boy"
        print(f"⚠️  空prompt，使用默认: {prompt}")
    if negative_prompt is None:
        negative_prompt = ""
        print(f"⚠️  negative_prompt为None，使用空字符串")
    
    print(f"🔍 Debug - prompt: {repr(prompt)}, negative_prompt: {repr(negative_prompt)}")
    '''
    
    if re.search(pattern1, content, re.DOTALL):
        content = re.sub(pattern1, replacement1, content, count=1, flags=re.DOTALL)
        print("✅ 添加了generate_images_common的None检查")
    
    # 修复2: 在diffusers管道中强制禁用安全检查
    pattern2 = r'(# 创建图像到图像管道（共享组件）\s*img2img_pipeline = StableDiffusionImg2ImgPipeline\(.*?\)\.to\(device\))'
    replacement2 = r'''\1
        
        # 🚨 额外确保安全检查器被禁用
        txt2img_pipeline.safety_checker = None
        txt2img_pipeline.requires_safety_checker = False
        img2img_pipeline.safety_checker = None
        img2img_pipeline.requires_safety_checker = False'''
    
    if re.search(pattern2, content, re.DOTALL):
        content = re.sub(pattern2, replacement2, content, count=1, flags=re.DOTALL)
        print("✅ 强化了安全检查器禁用")
    
    # 修复3: 添加更多LoRA选项到动漫模型配置
    # 首先检查现有的LoRA配置
    anime_lora_pattern = r'("anime": \{[^}]+?"lora_path": "[^"]+?",\s*"lora_id": "[^"]+?",)'
    
    if re.search(anime_lora_pattern, content):
        # 添加新的LoRA选项到可用LoRA列表
        content += '''
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
'''
        print("✅ 添加了新的动漫LoRA配置")
    
    # 写回文件
    with open('backend/handler.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 所有修复已应用到 backend/handler.py")

if __name__ == "__main__":
    fix_handler_file() 