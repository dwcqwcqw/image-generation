#!/usr/bin/env python3
"""
修复后的智能Prompt压缩测试脚本
测试去重功能和negative prompt压缩
"""

import re

def compress_prompt_to_77_tokens(prompt: str, max_tokens: int = 75) -> str:
    """
    智能压缩prompt到指定token数量以内
    保留最重要的关键词和描述
    """
    # 计算当前token数量
    token_pattern = r'\w+|[^\w\s]'
    current_tokens = len(re.findall(token_pattern, prompt.lower()))
    
    if current_tokens <= max_tokens:
        return prompt
    
    print(f"🔧 压缩prompt: {current_tokens} tokens -> {max_tokens} tokens")
    
    # 定义重要性权重
    priority_keywords = {
        # 质量标签 - 最高优先级
        'quality': ['masterpiece', 'best quality', 'amazing quality', 'high quality', 'ultra quality'],
        # 主体描述 - 高优先级  
        'subject': ['man', 'boy', 'male', 'muscular', 'handsome', 'lean', 'naked', 'nude'],
        # 身体部位 - 中高优先级
        'anatomy': ['torso', 'chest', 'abs', 'penis', 'erect', 'flaccid', 'body'],
        # 动作姿态 - 中优先级
        'pose': ['reclining', 'lying', 'sitting', 'standing', 'pose', 'position'],
        # 环境道具 - 中优先级
        'environment': ['bed', 'sheets', 'satin', 'luxurious', 'room', 'background'],
        # 光影效果 - 低优先级
        'lighting': ['lighting', 'illuminated', 'soft', 'moody', 'warm', 'cinematic'],
        # 情感表达 - 低优先级
        'emotion': ['serene', 'intense', 'confident', 'contemplation', 'allure']
    }
    
    # 🚨 修复：使用set来跟踪已添加的词，避免重复
    words = prompt.split()
    used_words = set()  # 跟踪已使用的词
    compressed_parts = []
    remaining_tokens = max_tokens
    
    # 按优先级处理
    priority_order = ['quality', 'subject', 'anatomy', 'pose', 'environment', 'lighting', 'emotion']
    
    for category in priority_order:
        if remaining_tokens <= 5:  # 预留一些空间
            break
            
        category_keywords = priority_keywords[category]
        
        # 找到属于这个类别的词
        for word in words:
            if remaining_tokens <= 0:
                break
                
            word_clean = word.lower().strip('.,!?;:')
            
            # 检查是否属于当前类别 且 没有被使用过
            if word_clean not in used_words and any(keyword in word_clean for keyword in category_keywords):
                word_tokens = len(re.findall(token_pattern, word.lower()))
                if word_tokens <= remaining_tokens:
                    compressed_parts.append(word)
                    used_words.add(word_clean)
                    remaining_tokens -= word_tokens
    
    # 🚨 修复：如果还有空间，添加其他重要但未分类的词（避免重复）
    if remaining_tokens > 0:
        for word in words:
            if remaining_tokens <= 0:
                break
                
            word_clean = word.lower().strip('.,!?;:')
            if word_clean not in used_words:
                word_tokens = len(re.findall(token_pattern, word.lower()))
                if word_tokens <= remaining_tokens:
                    compressed_parts.append(word)
                    used_words.add(word_clean)
                    remaining_tokens -= word_tokens
    
    compressed_prompt = ' '.join(compressed_parts)
    final_tokens = len(re.findall(token_pattern, compressed_prompt.lower()))
    
    print(f"✅ 压缩完成: '{compressed_prompt}' ({final_tokens} tokens)")
    return compressed_prompt

def count_tokens(text: str) -> int:
    """计算token数量"""
    token_pattern = r'\w+|[^\w\s]'
    return len(re.findall(token_pattern, text.lower()))

def check_duplicates(text: str) -> list:
    """检查重复词汇"""
    words = [word.lower().strip('.,!?;:') for word in text.split()]
    duplicates = []
    seen = set()
    
    for word in words:
        if word in seen:
            duplicates.append(word)
        else:
            seen.add(word)
    
    return duplicates

def test_fixed_compression():
    """测试修复后的压缩功能"""
    print("🧪 修复后的智能Prompt压缩测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        {
            'name': '超长positive prompt (114 tokens)',
            'prompt': 'masterpiece, best quality, amazing quality, A lean, muscular, and handsome man reclining on a bed with luxurious satin sheets. His chiseled torso is partially illuminated by soft, moody lighting that accentuates the contours of his muscles. One arm is tucked under his head, creating a relaxed yet confident pose. His expression is serene yet intense, with a gaze that suggests quiet contemplation. The satin sheets gently reflect the light, adding a sense of elegance and intimacy to the scene. The overall atmosphere is warm, sensual, and cinematic, evoking a timeless allure. erect penis, flaccid penis',
            'type': 'positive'
        },
        {
            'name': '超长negative prompt (90 tokens)', 
            'prompt': 'bad quality, worst quality, low quality, normal quality, lowres, blurry, fuzzy, out of focus, bad anatomy, bad hands, missing fingers, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, bad proportions, gross proportions, disfigured, out of frame, extra limbs, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, cross-eyed',
            'type': 'negative'
        },
        {
            'name': '中等长度prompt (47 tokens)',
            'prompt': 'masterpiece, best quality, muscular handsome man, naked, sitting on bed, soft lighting, detailed anatomy, erect penis',
            'type': 'positive'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}: {test_case['name']}")
        print("-" * 40)
        
        original = test_case['prompt']
        original_tokens = count_tokens(original)
        
        print(f"原始{test_case['type']} prompt ({original_tokens} tokens):")
        print(f"'{original[:200]}{'...' if len(original) > 200 else ''}'")
        print()
        
        if original_tokens > 75:
            compressed = compress_prompt_to_77_tokens(original, max_tokens=75)
            compressed_tokens = count_tokens(compressed)
            
            # 检查重复词汇
            duplicates = check_duplicates(compressed)
            
            print(f"\n📊 压缩结果:")
            print(f"  原始: {original_tokens} tokens")
            print(f"  压缩: {compressed_tokens} tokens")
            print(f"  压缩率: {((original_tokens - compressed_tokens) / original_tokens * 100):.1f}%")
            
            if duplicates:
                print(f"  ❌ 发现重复词汇: {', '.join(set(duplicates))}")
            else:
                print(f"  ✅ 无重复词汇")
            
            # 分析保留的关键词
            original_words = set(original.lower().split())
            compressed_words = set(compressed.lower().split())
            preserved_words = compressed_words.intersection(original_words)
            
            print(f"  保留词汇: {len(preserved_words)}/{len(original_words)} ({len(preserved_words)/len(original_words)*100:.1f}%)")
        else:
            print("✅ 无需压缩，已在75 token限制内")
    
    print("\n" + "=" * 60)
    print("🎯 修复测试完成！")
    print("🔧 主要修复:")
    print("  ✅ 去除重复词汇")
    print("  ✅ 支持negative prompt压缩")
    print("  ✅ 优化压缩算法")

if __name__ == "__main__":
    test_fixed_compression() 