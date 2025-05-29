#!/usr/bin/env python3
"""
Debug startup script for RunPod container
Helps identify issues with container startup and model loading
"""

import os
import sys
import subprocess
import traceback
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def run_command(cmd, description):
    """Run a command and print the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(f"{description}:")
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(f"STDERR: {result.stderr.strip()}")
        print(f"Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"{description}: Command timed out")
    except Exception as e:
        print(f"{description}: Error - {e}")

def check_environment():
    """Check system environment and dependencies"""
    print_section("SYSTEM ENVIRONMENT CHECK")
    
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check system resources
    run_command("free -h", "Memory usage")
    run_command("df -h", "Disk usage")
    run_command("nvidia-smi", "GPU status")
    
    # Check environment variables
    print("\nEnvironment variables:")
    for key in sorted(os.environ.keys()):
        if any(keyword in key.upper() for keyword in ['CUDA', 'TORCH', 'CLOUDFLARE', 'RUNPOD']):
            print(f"  {key}={os.environ[key]}")

def check_python_packages():
    """Check Python package installations"""
    print_section("PYTHON PACKAGES CHECK")
    
    critical_packages = [
        'torch', 'torchvision', 'numpy', 'diffusers', 
        'transformers', 'PIL', 'boto3', 'runpod'
    ]
    
    for package in critical_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError as e:
            print(f"✗ {package}: Import failed - {e}")
        except Exception as e:
            print(f"? {package}: Error - {e}")

def check_torch_cuda():
    """Check PyTorch CUDA functionality"""
    print_section("PYTORCH CUDA CHECK")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                
            # Test CUDA functionality
            try:
                test_tensor = torch.tensor([1.0, 2.0]).cuda()
                print(f"✓ CUDA tensor test passed: {test_tensor}")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ CUDA tensor test failed: {e}")
        else:
            print("CUDA not available - will run on CPU")
            
    except Exception as e:
        print(f"PyTorch check failed: {e}")
        traceback.print_exc()

def check_model_paths():
    """Check if model directories exist"""
    print_section("MODEL PATHS CHECK")
    
    expected_paths = [
        "/runpod-volume",
        "/runpod-volume/flux_base",
        "/runpod-volume/lora",
        "/runpod-volume/lora/flux_nsfw",
        "/runpod-volume/cartoon",
        "/runpod-volume/cartoon/lora"
    ]
    
    for path in expected_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                files = os.listdir(path)
                file_count = len(files)
                print(f"✓ {path} (directory, {file_count} items)")
                if file_count > 0 and file_count < 10:  # Show files if not too many
                    for file in files[:5]:  # Show first 5 files
                        print(f"    - {file}")
                    if file_count > 5:
                        print(f"    ... and {file_count - 5} more")
            else:
                file_size = os.path.getsize(path) / 1024**2  # MB
                print(f"✓ {path} (file, {file_size:.1f}MB)")
        else:
            print(f"✗ {path} (not found)")

def test_imports():
    """Test critical imports for the handler"""
    print_section("CRITICAL IMPORTS TEST")
    
    try:
        print("Testing diffusers...")
        from diffusers import FluxPipeline, FluxImg2ImgPipeline
        print("✓ Diffusers imports successful")
        
        print("Testing PIL...")
        from PIL import Image
        print("✓ PIL import successful")
        
        print("Testing boto3...")
        import boto3
        print("✓ boto3 import successful")
        
        print("Testing runpod...")
        import runpod
        print("✓ runpod import successful")
        
        print("Testing compel...")
        try:
            from compel import Compel
            print("✓ compel import successful")
        except ImportError:
            print("⚠ compel import failed (optional)")
            
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        traceback.print_exc()

def main():
    """Main debug function"""
    print_section("RUNPOD CONTAINER DEBUG STARTUP")
    
    try:
        check_environment()
        check_python_packages()
        check_torch_cuda()
        check_model_paths()
        test_imports()
        
        print_section("STARTING MAIN HANDLER")
        print("All checks completed. Starting handler.py...")
        
        # Import and run the main handler
        import handler
        print("✓ Handler imported successfully")
        
    except Exception as e:
        print_section("STARTUP ERROR")
        print(f"Fatal error during startup: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 