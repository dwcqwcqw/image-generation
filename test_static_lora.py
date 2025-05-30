#!/usr/bin/env python3
"""
测试静态前端 + 动态后端搜索的LoRA系统
"""

import sys
import os
sys.path.append('backend')

from handler import (
    find_lora_file, 
    get_available_loras, 
    get_loras_by_base_model,
    LORA_SEARCH_PATHS,
    LORA_FILE_PATTERNS
)

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
        available = get_available_loras()
        print(f"  get_available_loras(): ✅ 成功")
        print(f"    消息: {available.get('message', 'N/A')}")
        
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
    
    print("\n🎉 静态LoRA系统测试完成!")
    return True

if __name__ == "__main__":
    test_static_lora_system() 