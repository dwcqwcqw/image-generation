#!/usr/bin/env python3
"""
测试长提示词处理功能（简化版本）
"""

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
    
    # 简单的token估算（按空格和逗号分割）
    words = prompt.replace(',', ' , ').split()
    estimated_tokens = len(words)
    
    print(f"📏 Prompt analysis: {len(prompt)} chars, ~{estimated_tokens} tokens")
    
    if estimated_tokens <= max_clip_tokens:
        # 短prompt：两个编码器都使用完整prompt
        print("✅ Short prompt: using full prompt for both CLIP and T5")
        return prompt, prompt
    else:
        # 长prompt：CLIP使用截断版本，T5使用完整版本（如果不超过512token）
        if estimated_tokens <= max_t5_tokens:
            # 为CLIP创建截断版本，保持语义完整性
            clip_words = words[:max_clip_tokens]
            # 尝试在句号或逗号处截断以保持语义
            for i in range(len(clip_words) - 1, max(0, len(clip_words) - 10), -1):
                if clip_words[i].endswith(('.', ',', ';')):
                    clip_words = clip_words[:i+1]
                    break
            
            clip_prompt = ' '.join(clip_words).replace(' , ', ', ')
            print(f"📝 Long prompt optimization:")
            print(f"   CLIP prompt: ~{len(clip_words)} tokens (truncated)")
            print(f"   T5 prompt: ~{estimated_tokens} tokens (full)")
            return clip_prompt, prompt
        else:
            # 超长prompt：两个编码器都需要截断
            clip_words = words[:max_clip_tokens]
            t5_words = words[:max_t5_tokens]
            
            # 尝试在合适位置截断
            for i in range(len(clip_words) - 1, max(0, len(clip_words) - 10), -1):
                if clip_words[i].endswith(('.', ',', ';')):
                    clip_words = clip_words[:i+1]
                    break
                    
            for i in range(len(t5_words) - 1, max(0, len(t5_words) - 20), -1):
                if t5_words[i].endswith(('.', ',', ';')):
                    t5_words = t5_words[:i+1]
                    break
            
            clip_prompt = ' '.join(clip_words).replace(' , ', ', ')
            t5_prompt = ' '.join(t5_words).replace(' , ', ', ')
            
            print(f"⚠️  Ultra-long prompt: both encoders truncated")
            print(f"   CLIP prompt: ~{len(clip_words)} tokens")
            print(f"   T5 prompt: ~{len(t5_words)} tokens")
            return clip_prompt, t5_prompt

def test_long_prompt_processing():
    """测试长提示词处理功能"""
    print("🧪 Testing long prompt processing functionality...")
    
    # 测试1：短prompt（应该两个编码器都使用完整prompt）
    short_prompt = "A beautiful landscape with mountains and rivers"
    clip_short, t5_short = process_long_prompt(short_prompt)
    print(f"\n📝 Test 1 - Short prompt:")
    print(f"Input: {short_prompt}")
    print(f"CLIP output: {clip_short}")
    print(f"T5 output: {t5_short}")
    assert clip_short == short_prompt
    assert t5_short == short_prompt
    print("✅ Short prompt test passed")
    
    # 测试2：中等长度prompt（CLIP截断，T5完整）
    medium_prompt = "A highly detailed, photorealistic digital artwork depicting a majestic mountain landscape at golden hour, with snow-capped peaks towering above a serene alpine lake that perfectly reflects the warm orange and pink hues of the sunset sky, surrounded by dense coniferous forests of pine and fir trees, with a small wooden cabin nestled among the trees, smoke gently rising from its chimney, creating a peaceful and idyllic scene that captures the essence of natural beauty and tranquility"
    clip_medium, t5_medium = process_long_prompt(medium_prompt)
    print(f"\n📝 Test 2 - Medium prompt:")
    print(f"Input length: {len(medium_prompt)} chars")
    print(f"CLIP output: {clip_medium}")
    print(f"T5 output: {t5_medium}")
    assert len(clip_medium) < len(medium_prompt)  # CLIP应该被截断
    assert t5_medium == medium_prompt  # T5应该是完整的
    print("✅ Medium prompt test passed")
    
    # 测试3：超长prompt（两个编码器都截断）
    very_long_prompt = " ".join([
        "A highly detailed, photorealistic digital artwork depicting",
        "a majestic mountain landscape at golden hour with snow-capped peaks",
        "towering above a serene alpine lake that perfectly reflects",
        "the warm orange and pink hues of the sunset sky surrounded by",
        "dense coniferous forests of pine and fir trees with a small wooden cabin",
        "nestled among the trees smoke gently rising from its chimney creating",
        "a peaceful and idyllic scene that captures the essence of natural beauty",
        "and tranquility featuring intricate details like individual leaves on trees",
        "ripples on the water surface reflections of clouds wildlife such as deer",
        "and birds atmospheric perspective with misty valleys distant mountain ranges",
        "dramatic lighting effects volumetric rays of sunlight filtering through",
        "the forest canopy creating god rays and lens flares professional photography",
        "style with shallow depth of field bokeh effects cinematic composition",
        "rule of thirds leading lines dynamic range HDR processing post-processing",
        "color grading warm color palette earth tones natural saturation",
        "high contrast sharp focus crystal clear details 8K resolution",
        "ultra-wide aspect ratio panoramic view establishing shot environmental",
        "storytelling mood and atmosphere emotional impact artistic vision",
        "masterpiece quality award-winning photography nature documentary style"
    ] * 5)  # 重复5次创建超长prompt
    
    clip_long, t5_long = process_long_prompt(very_long_prompt)
    print(f"\n📝 Test 3 - Very long prompt:")
    print(f"Input length: {len(very_long_prompt)} chars")
    print(f"CLIP output length: {len(clip_long)} chars")
    print(f"T5 output length: {len(t5_long)} chars")
    assert len(clip_long) < len(very_long_prompt)  # CLIP应该被截断
    assert len(t5_long) < len(very_long_prompt)    # T5也应该被截断
    print("✅ Very long prompt test passed")
    
    # 测试4：空prompt
    empty_clip, empty_t5 = process_long_prompt("")
    assert empty_clip == ""
    assert empty_t5 == ""
    print("✅ Empty prompt test passed")
    
    print("\n🎉 All long prompt processing tests passed!")

if __name__ == "__main__":
    test_long_prompt_processing() 