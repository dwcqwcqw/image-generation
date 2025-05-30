#!/usr/bin/env python3
"""
测试长提示词处理功能（改进版本）
"""
import re

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

def test_long_prompt_processing():
    """测试长提示词处理功能"""
    print("🧪 Testing improved long prompt processing functionality...\n")
    
    # 测试1：短prompt（应该两个编码器都使用完整prompt）
    short_prompt = "A beautiful landscape with mountains and rivers"
    print(f"📝 Test 1 - Short prompt:")
    print(f"Input: {short_prompt}")
    clip_short, t5_short = process_long_prompt(short_prompt)
    print(f"CLIP output: {clip_short}")
    print(f"T5 output: {t5_short}")
    assert clip_short == short_prompt
    assert t5_short == short_prompt
    print("✅ Short prompt test passed\n")
    
    # 测试2：实际导致截断的prompt（从日志中提取）
    real_long_prompt = "A young, handsome, muscular man with defined abs and pecs stands confidently in a luxurious bedroom. His partner, equally attractive, kneels before him with an expression of desire and anticipation. The man's arousal is evident, and his partner leans in closer, ready to pleasure him. The scene is intimate and passionate as the partner gives him a blowjob,"
    print(f"📝 Test 2 - Real problematic prompt (557 chars):")
    print(f"Input: {real_long_prompt}")
    clip_real, t5_real = process_long_prompt(real_long_prompt)
    print(f"CLIP output ({len(clip_real)} chars): {clip_real}")
    print(f"T5 output ({len(t5_real)} chars): {t5_real}")
    
    # 验证CLIP部分token数不超过75
    clip_tokens = len(re.findall(r'\w+|[^\w\s]', clip_real.lower()))
    print(f"🔍 CLIP token count: {clip_tokens} (should be ≤ 75)")
    assert clip_tokens <= 75, f"CLIP token count {clip_tokens} exceeds limit of 75"
    assert t5_real == real_long_prompt, "T5 should have full prompt"
    print("✅ Real prompt test passed\n")
    
    # 测试3：超长prompt（需要两个编码器都截断）
    ultra_long_prompt = " ".join([
        "A very detailed and extremely long prompt that describes",
        "countless elements and scenarios that would definitely",
        "exceed both the CLIP and T5 token limits,",
        "including multiple characters, complex backgrounds,",
        "detailed lighting conditions, specific art styles,",
        "numerous objects and props, detailed clothing descriptions,",
        "facial expressions, body positions, environmental details,",
        "atmospheric conditions, color palettes, composition rules,",
        "camera angles, depth of field settings, and many other",
        "technical and artistic specifications that continue on",
        "and on with even more descriptive elements and requirements"
    ] * 10)  # 重复10次让它变得超长
    
    print(f"📝 Test 3 - Ultra-long prompt ({len(ultra_long_prompt)} chars):")
    clip_ultra, t5_ultra = process_long_prompt(ultra_long_prompt)
    
    clip_ultra_tokens = len(re.findall(r'\w+|[^\w\s]', clip_ultra.lower()))
    t5_ultra_tokens = len(re.findall(r'\w+|[^\w\s]', t5_ultra.lower()))
    
    print(f"🔍 CLIP result: {clip_ultra_tokens} tokens (should be ≤ 75)")
    print(f"🔍 T5 result: {t5_ultra_tokens} tokens (should be ≤ 500)")
    
    assert clip_ultra_tokens <= 75, f"CLIP token count {clip_ultra_tokens} exceeds limit"
    assert t5_ultra_tokens <= 500, f"T5 token count {t5_ultra_tokens} exceeds limit"
    print("✅ Ultra-long prompt test passed\n")
    
    print("🎉 All improved prompt processing tests passed!")

if __name__ == "__main__":
    test_long_prompt_processing() 