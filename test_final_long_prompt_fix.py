#!/usr/bin/env python3
"""
最终长prompt修复验证脚本
测试真正绕过CLIP 77 token限制的实现
"""

import re
import time

def analyze_logs_for_truncation(log_file="logs"):
    """分析日志中的截断问题"""
    print("🔍 分析日志中的长prompt处理...")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找截断警告
        truncation_warnings = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if "Token indices sequence length is longer than" in line:
                truncation_warnings.append({
                    'line': i+1,
                    'content': line.strip(),
                    'context': lines[max(0, i-2):i+3]
                })
            elif "The following part of your input was truncated" in line:
                truncation_warnings.append({
                    'line': i+1, 
                    'content': line.strip(),
                    'type': 'truncation'
                })
        
        # 查找长prompt处理日志
        long_prompt_logs = []
        for i, line in enumerate(lines):
            if "长prompt分为" in line and "段处理" in line:
                long_prompt_logs.append({
                    'line': i+1,
                    'content': line.strip(),
                    'context': lines[max(0, i-1):i+4]
                })
            elif "真正的分段长prompt处理完成" in line:
                long_prompt_logs.append({
                    'line': i+1,
                    'content': line.strip(),
                    'type': 'success'
                })
        
        print(f"\n📊 分析结果:")
        print(f"   截断警告: {len([w for w in truncation_warnings if 'truncation' in w.get('type', '')])} 条")
        print(f"   Token长度警告: {len([w for w in truncation_warnings if 'truncation' not in w.get('type', '')])} 条")
        print(f"   长prompt处理日志: {len(long_prompt_logs)} 条")
        
        if truncation_warnings:
            print(f"\n⚠️  仍然发现截断问题:")
            for i, warning in enumerate(truncation_warnings[:5]):  # 只显示前5条
                print(f"   {i+1}. 第{warning['line']}行: {warning['content'][:100]}...")
        
        if long_prompt_logs:
            print(f"\n✅ 长prompt处理日志:")
            for log in long_prompt_logs[-3:]:  # 显示最近3条
                print(f"   - {log['content']}")
                
        # 分析最新的处理是否成功
        latest_success = None
        for log in reversed(long_prompt_logs):
            if log.get('type') == 'success':
                latest_success = log
                break
        
        if latest_success:
            print(f"\n🎉 最新的长prompt处理成功: 第{latest_success['line']}行")
            return True
        else:
            print(f"\n❌ 未发现最新的长prompt处理成功日志")
            return False
            
    except FileNotFoundError:
        print(f"❌ 日志文件不存在: {log_file}")
        return False
    except Exception as e:
        print(f"❌ 分析日志时出错: {e}")
        return False

def create_test_prompts():
    """创建测试用的长prompt"""
    
    prompts = {
        "medium": {
            "text": "masterpiece, best quality, 1boy, muscular, handsome, detailed eyes, perfect anatomy, high resolution, cinematic lighting, detailed background",
            "expected_tokens": 25
        },
        "long": {
            "text": "masterpiece, best quality, amazing quality, 4k, ultra detailed, 1boy, solo, male focus, bodybuilder, muscular, big pecs, bara, thick thighs, huge thighs, handsome face, detailed eyes, perfect anatomy, dramatic lighting, cinematic composition, high resolution, photorealistic, detailed background, intricate details, professional photography, studio lighting, depth of field, bokeh effect, vibrant colors, sharp focus, perfect skin texture, detailed facial features, expressive eyes, confident pose, dynamic angle, artistic composition",
            "expected_tokens": 85
        },
        "super_long": {
            "text": "masterpiece, best quality, amazing quality, 4k, ultra detailed, ultra high resolution, photorealistic, professional photography, 1boy, solo, male focus, bodybuilder, muscular male, big pecs, bara, thick thighs, huge thighs, broad shoulders, defined abs, handsome face, chiseled jawline, detailed eyes, expressive eyes, perfect anatomy, flawless skin, detailed skin texture, dramatic lighting, cinematic lighting, studio lighting, professional lighting setup, volumetric lighting, rim lighting, soft shadows, high contrast, cinematic composition, dynamic angle, low angle shot, heroic pose, confident expression, serene expression, detailed background, intricate details, architectural background, modern interior, luxurious setting, elegant furniture, glass windows, natural light, depth of field, bokeh effect, shallow depth of field, vibrant colors, rich colors, color grading, sharp focus, crystal clear, hyper detailed, ultra sharp, 8k resolution, award winning photography, magazine quality, commercial photography style, fashion photography, editorial style, fine art photography",
            "expected_tokens": 180
        }
    }
    
    return prompts

def estimate_tokens(text):
    """估算token数量"""
    # 简单的token估算
    words = text.split()
    token_pattern = r'\w+|[^\w\s]'
    total_tokens = 0
    for word in words:
        tokens = len(re.findall(token_pattern, word.lower()))
        total_tokens += tokens
    return total_tokens

def main():
    """主函数"""
    print("🚀 最终长prompt修复验证")
    print("=" * 50)
    
    # 1. 分析当前日志状态
    print("\n1️⃣  分析当前日志状态")
    current_success = analyze_logs_for_truncation()
    
    # 2. 创建测试prompts
    print("\n2️⃣  测试prompt分析")
    test_prompts = create_test_prompts()
    
    for name, prompt_info in test_prompts.items():
        actual_tokens = estimate_tokens(prompt_info["text"])
        expected_tokens = prompt_info["expected_tokens"]
        
        print(f"\n📝 {name.upper()} Prompt:")
        print(f"   长度: {len(prompt_info['text'])} 字符")
        print(f"   预期tokens: {expected_tokens}")
        print(f"   实际估算: {actual_tokens} tokens")
        print(f"   需要分段: {'是' if actual_tokens > 75 else '否'}")
        
        if actual_tokens > 75:
            segments_needed = (actual_tokens + 74) // 75
            print(f"   预期分段数: {segments_needed}")
    
    # 3. 检查修复状态
    print(f"\n3️⃣  修复状态检查")
    
    if current_success:
        print("✅ 当前日志显示长prompt处理成功")
        print("📈 建议测试以下场景:")
        print("   - 动漫模型 + LoRA + 超长prompt (180+ tokens)")
        print("   - 真人模型 + LoRA + 长prompt (100+ tokens)")
        print("   - 验证不再出现截断警告")
    else:
        print("❌ 当前日志仍显示截断问题")
        print("🔧 需要验证修复是否正确部署")
    
    # 4. 生成测试建议
    print(f"\n4️⃣  测试建议")
    print("📋 推荐测试步骤:")
    print("1. 重启服务确保新代码生效")
    print("2. 使用super_long prompt测试动漫模型+LoRA")
    print("3. 检查日志是否出现 '真正的分段长prompt处理完成'")
    print("4. 验证不再有 'truncated because CLIP' 警告")
    print("5. 确认生成的图像质量正常")
    
    print(f"\n🎯 关键成功指标:")
    print("- 无截断警告消息")
    print("- 显示 '绕过77 token限制' 日志")
    print("- 图像文件大小正常 (>1MB)")
    print("- 生成速度正常")

if __name__ == "__main__":
    main() 