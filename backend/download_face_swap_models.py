#!/usr/bin/env python3
"""
换脸模型下载脚本
自动下载和配置换脸所需的模型文件
"""

import os
import urllib.request
import shutil
from pathlib import Path

def create_model_directory(base_path="/runpod-volume/faceswap"):
    """创建模型目录结构"""
    print(f"📁 创建模型目录: {base_path}")
    
    # 尝试多个可能的路径
    possible_paths = [
        "/runpod-volume/faceswap",
        "/workspace/faceswap", 
        "/app/faceswap",
        "./faceswap"
    ]
    
    for path in possible_paths:
        try:
            os.makedirs(path, exist_ok=True)
            print(f"✅ 目录创建成功: {path}")
            return path
        except PermissionError:
            print(f"⚠️ 权限不足，跳过: {path}")
        except Exception as e:
            print(f"❌ 创建失败: {path} - {e}")
    
    # 如果都失败，使用当前目录
    fallback_path = "./models/faceswap"
    os.makedirs(fallback_path, exist_ok=True)
    print(f"✅ 使用备用目录: {fallback_path}")
    return fallback_path

def download_file(url, filename, target_dir):
    """下载文件到指定目录"""
    target_path = os.path.join(target_dir, filename)
    
    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"✅ 文件已存在: {filename} ({size_mb:.1f}MB)")
        return True
    
    print(f"📥 下载 {filename}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"✅ 下载完成: {filename} ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {filename} - {e}")
        return False

def download_insightface_models():
    """下载InsightFace模型"""
    print("\n🔄 检查InsightFace模型...")
    
    try:
        import insightface
        from insightface import app
        
        # 这会自动下载buffalo_l模型到 ~/.insightface/models/
        print("📦 初始化InsightFace模型...")
        face_app = app.FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace模型准备完成")
        return True
        
    except Exception as e:
        print(f"❌ InsightFace模型下载失败: {e}")
        return False

def download_face_swap_models(model_dir):
    """下载换脸模型"""
    print(f"\n🔄 下载换脸模型到: {model_dir}")
    
    models = {
        "inswapper_128_fp16.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128_fp16.onnx"
    }
    
    success_count = 0
    for filename, url in models.items():
        if download_file(url, filename, model_dir):
            success_count += 1
    
    return success_count == len(models)

def create_model_config(model_dir):
    """创建模型配置文件"""
    config_content = f"""# 换脸模型配置
# 生成时间: {os.popen('date').read().strip()}

[paths]
base_dir = {model_dir}
face_swap_model = {os.path.join(model_dir, 'inswapper_128_fp16.onnx')}
face_analysis_model = ~/.insightface/models/buffalo_l

[settings]
face_enhancement = false  # GFPGAN不可用时禁用
detection_size = 640
"""
    
    config_path = os.path.join(model_dir, "config.ini")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ 配置文件已创建: {config_path}")

def verify_models(model_dir):
    """验证模型文件"""
    print(f"\n🔍 验证模型文件...")
    
    required_files = [
        "inswapper_128_fp16.onnx"
    ]
    
    all_good = True
    for filename in required_files:
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {filename}: {size_mb:.1f}MB")
        else:
            print(f"❌ 缺失: {filename}")
            all_good = False
    
    # 检查InsightFace模型
    insightface_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
    if os.path.exists(insightface_dir):
        files = os.listdir(insightface_dir)
        print(f"✅ InsightFace模型: {len(files)} 个文件")
    else:
        print("❌ InsightFace模型目录不存在")
        all_good = False
    
    return all_good

def main():
    """主函数"""
    print("🚀 开始下载换脸模型...")
    
    # 创建模型目录
    model_dir = create_model_directory()
    
    # 下载InsightFace模型
    insightface_ok = download_insightface_models()
    
    # 下载换脸模型
    faceswap_ok = download_face_swap_models(model_dir)
    
    # 创建配置文件
    create_model_config(model_dir)
    
    # 验证所有模型
    all_ok = verify_models(model_dir)
    
    print(f"\n📊 下载结果:")
    print(f"   InsightFace: {'✅' if insightface_ok else '❌'}")
    print(f"   换脸模型: {'✅' if faceswap_ok else '❌'}")
    print(f"   整体验证: {'✅' if all_ok else '❌'}")
    
    if all_ok:
        print(f"\n🎉 所有模型下载完成！")
        print(f"   模型目录: {model_dir}")
        print(f"   请确保在handler.py中正确配置路径")
    else:
        print(f"\n⚠️ 部分模型下载失败，但系统可以在无换脸模式下运行")
    
    return all_ok

if __name__ == "__main__":
    main() 