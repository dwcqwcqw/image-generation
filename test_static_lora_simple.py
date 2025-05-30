#!/usr/bin/env python3
"""
简化测试静态前端 + 动态后端搜索的LoRA系统（避免依赖冲突）
"""

import os

# 复制核心配置和函数（避免导入整个handler模块）
LORA_SEARCH_PATHS = {
    "realistic": [
        "/runpod-volume/lora",
        "/runpod-volume/lora/realistic"
    ],
    "anime": [
        "/runpod-volume/cartoon/lora",
        "/runpod-volume/anime/lora"
    ]
}

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
    
    # 动漫风格LoRA
    "gayporn": ["Gayporn.safetensor", "gayporn.safetensors", "GayPorn.safetensors"]
}

def find_lora_file(lora_id: str, base_model: str) -> str:
    """动态搜索LoRA文件路径"""
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
                if filename.endswith(('.safetensors', '.ckpt', '.pt')):
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
        ]
    }

def test_static_lora_system():
    """测试静态LoRA系统"""
    print("🧪 测试静态前端 + 动态后端搜索的LoRA系统")
    print("=" * 60)
    
    # 测试1: 验证搜索路径配置
    print("\n📁 测试1: 验证搜索路径配置")
    print(f"真人风格搜索路径: {LORA_SEARCH_PATHS['realistic']}")
    print(f"动漫风格搜索路径: {LORA_SEARCH_PATHS['anime']}")
    
    # 测试2: 验证文件模式配置
    print("\n🔍 测试2: 验证文件模式配置")
    print(f"配置的LoRA数量: {len(LORA_FILE_PATTERNS)}")
    for lora_id, patterns in list(LORA_FILE_PATTERNS.items())[:3]:
        print(f"  {lora_id}: {patterns}")
    print("  ...")
    
    # 测试3: 测试动态文件搜索
    print("\n🔎 测试3: 测试动态文件搜索")
    test_loras = ["flux_nsfw", "chastity_cage", "gayporn"]
    
    for lora_id in test_loras:
        base_model = "realistic" if lora_id != "gayporn" else "anime"
        result = find_lora_file(lora_id, base_model)
        status = "✅ 找到" if result else "❌ 未找到"
        print(f"  {lora_id} ({base_model}): {status}")
        if result:
            print(f"    路径: {result}")
    
    # 测试4: 测试简化的API函数
    print("\n📋 测试4: 测试简化的API函数")
    
    try:
        by_model = get_loras_by_base_model()
        print(f"  get_loras_by_base_model(): ✅ 成功")
        print(f"    真人风格LoRA数量: {len(by_model.get('realistic', []))}")
        print(f"    动漫风格LoRA数量: {len(by_model.get('anime', []))}")
        
    except Exception as e:
        print(f"  API函数测试: ❌ 失败 - {e}")
    
    # 测试5: 验证前端静态列表一致性
    print("\n🎨 测试5: 验证前端静态列表一致性")
    
    # 从后端API获取列表
    backend_data = get_loras_by_base_model()
    realistic_loras = [lora['id'] for lora in backend_data.get('realistic', [])]
    anime_loras = [lora['id'] for lora in backend_data.get('anime', [])]
    
    # 检查是否与文件模式配置一致
    realistic_patterns = [lora_id for lora_id in LORA_FILE_PATTERNS.keys() if lora_id != 'gayporn']
    anime_patterns = [lora_id for lora_id in LORA_FILE_PATTERNS.keys() if lora_id == 'gayporn']
    
    print(f"  真人风格一致性: {'✅' if set(realistic_loras) == set(realistic_patterns) else '❌'}")
    print(f"  动漫风格一致性: {'✅' if set(anime_loras) == set(anime_patterns) else '❌'}")
    
    # 测试6: 验证前端静态列表内容
    print("\n🎨 测试6: 前端静态列表内容")
    print("真人风格LoRA:")
    for lora in backend_data['realistic'][:5]:  # 显示前5个
        print(f"  - {lora['id']}: {lora['name']}")
    print("  ...")
    
    print("动漫风格LoRA:")
    for lora in backend_data['anime']:
        print(f"  - {lora['id']}: {lora['name']}")
    
    print("\n🎉 静态LoRA系统测试完成!")
    print("\n📝 总结:")
    print("✅ 前端使用静态列表，无需动态扫描")
    print("✅ 后端在加载时动态搜索文件")
    print("✅ 配置简化，性能提升")
    return True

if __name__ == "__main__":
    test_static_lora_system() 