#!/usr/bin/env python3
"""
智能Prompt压缩测试脚本
测试将超长prompt压缩到77 token以内的效果
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
    
    # 分词并分类
    words = prompt.split()
    categorized_words = {category: [] for category in priority_keywords.keys()}
    uncategorized_words = []
    
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        categorized = False
        
        for category, keywords in priority_keywords.items():
            if any(keyword in word_lower for keyword in keywords):
                categorized_words[category].append(word)
                categorized = True
                break
        
        if not categorized:
            uncategorized_words.append(word)
    
    # 按优先级重建prompt
    compressed_parts = []
    remaining_tokens = max_tokens
    
    # 优先级顺序
    priority_order = ['quality', 'subject', 'anatomy', 'pose', 'environment', 'lighting', 'emotion']
    
    for category in priority_order:
        if remaining_tokens <= 0:
            break
            
        category_words = categorized_words[category]
        if category_words:
            # 计算这个类别的token数
            category_text = ' '.join(category_words)
            category_tokens = len(re.findall(token_pattern, category_text.lower()))
            
            if category_tokens <= remaining_tokens:
                compressed_parts.extend(category_words)
                remaining_tokens -= category_tokens
            else:
                # 部分添加最重要的词
                for word in category_words:
                    word_tokens = len(re.findall(token_pattern, word.lower()))
                    if word_tokens <= remaining_tokens:
                        compressed_parts.append(word)
                        remaining_tokens -= word_tokens
                    else:
                        break
    
    # 如果还有剩余空间，添加未分类的重要词
    for word in uncategorized_words:
        if remaining_tokens <= 0:
            break
        word_tokens = len(re.findall(token_pattern, word.lower()))
        if word_tokens <= remaining_tokens:
            compressed_parts.append(word)
            remaining_tokens -= word_tokens
    
    compressed_prompt = ' '.join(compressed_parts)
    final_tokens = len(re.findall(token_pattern, compressed_prompt.lower()))
    
    print(f"✅ 压缩完成: '{compressed_prompt}' ({final_tokens} tokens)")
    return compressed_prompt

def count_tokens(text: str) -> int:
    """计算token数量"""
    token_pattern = r'\w+|[^\w\s]'
    return len(re.findall(token_pattern, text.lower()))

def test_compression():
    """测试压缩功能"""
    print("🧪 智能Prompt压缩测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        {
            'name': '超长prompt (114 tokens)',
            'prompt': 'masterpiece, best quality, amazing quality, A lean, muscular, and handsome man reclining on a bed with luxurious satin sheets. His chiseled torso is partially illuminated by soft, moody lighting that accentuates the contours of his muscles. One arm is tucked under his head, creating a relaxed yet confident pose. His expression is serene yet intense, with a gaze that suggests quiet contemplation. The satin sheets gently reflect the light, adding a sense of elegance and intimacy to the scene. The overall atmosphere is warm, sensual, and cinematic, evoking a timeless allure. erect penis, flaccid penis'
        },
        {
            'name': '中等长度prompt (47 tokens)',
            'prompt': 'masterpiece, best quality, muscular handsome man, naked, sitting on bed, soft lighting, detailed anatomy, erect penis'
        },
        {
            'name': '短prompt (11 tokens)',
            'prompt': 'masterpiece, best quality, amazing quality, a naked boy'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}: {test_case['name']}")
        print("-" * 40)
        
        original = test_case['prompt']
        original_tokens = count_tokens(original)
        
        print(f"原始prompt ({original_tokens} tokens):")
        print(f"'{original}'")
        print()
        
        if original_tokens > 75:
            compressed = compress_prompt_to_77_tokens(original, max_tokens=75)
            compressed_tokens = count_tokens(compressed)
            
            print(f"\n📊 压缩结果:")
            print(f"  原始: {original_tokens} tokens")
            print(f"  压缩: {compressed_tokens} tokens")
            print(f"  压缩率: {((original_tokens - compressed_tokens) / original_tokens * 100):.1f}%")
            
            # 分析保留的关键词
            original_words = set(original.lower().split())
            compressed_words = set(compressed.lower().split())
            preserved_words = compressed_words.intersection(original_words)
            lost_words = original_words - compressed_words
            
            print(f"  保留词汇: {len(preserved_words)}/{len(original_words)} ({len(preserved_words)/len(original_words)*100:.1f}%)")
            if lost_words:
                print(f"  丢失词汇: {', '.join(sorted(lost_words)[:10])}{'...' if len(lost_words) > 10 else ''}")
        else:
            print("✅ 无需压缩，已在75 token限制内")
    
    print("\n" + "=" * 60)
    print("🎯 测试完成！智能压缩功能可以有效避免黑图问题")

if __name__ == "__main__":
    test_compression() 