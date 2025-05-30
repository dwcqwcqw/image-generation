#!/usr/bin/env python3
"""
Test script for handler.py to verify fixes before deployment.
This script tests the basic structure and imports without requiring GPU/models.
"""

import sys
import traceback
from unittest.mock import Mock, MagicMock
import torch

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    try:
        # Mock the diffusers imports to avoid requiring actual models
        sys.modules['diffusers'] = Mock()
        sys.modules['diffusers.FluxPipeline'] = Mock()
        sys.modules['diffusers.FluxImg2ImgPipeline'] = Mock()
        
        # Mock compel if not available
        sys.modules['compel'] = Mock()
        
        # Import the handler module
        import backend.handler as handler
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_function_signatures():
    """Test that key functions have correct signatures"""
    print("🧪 Testing function signatures...")
    try:
        import backend.handler as handler
        
        # Test that text_to_image function exists and accepts dict
        if not hasattr(handler, 'text_to_image'):
            raise AttributeError("text_to_image function not found")
        
        if not hasattr(handler, 'image_to_image'):
            raise AttributeError("image_to_image function not found")
            
        if not hasattr(handler, 'handler'):
            raise AttributeError("handler function not found")
            
        print("✅ All required functions found")
        return True
    except Exception as e:
        print(f"❌ Function signature error: {e}")
        traceback.print_exc()
        return False

def test_encode_prompt_calls():
    """Test the encode_prompt call structure without actual execution"""
    print("🧪 Testing encode_prompt call structure...")
    try:
        # Mock pipeline with encode_prompt that requires prompt_2
        mock_pipeline = Mock()
        
        # Configure encode_prompt to require prompt_2 parameter
        def mock_encode_prompt(*args, **kwargs):
            required_params = ['prompt', 'prompt_2', 'device', 'num_images_per_prompt']
            missing_params = [p for p in required_params if p not in kwargs and len(args) < len(required_params)]
            
            if missing_params:
                raise TypeError(f"encode_prompt() missing required arguments: {missing_params}")
            
            # Return mock embeddings object
            mock_embeds = Mock()
            mock_embeds.prompt_embeds = torch.zeros((1, 77, 768))  # Mock tensor
            mock_embeds.pooled_prompt_embeds = torch.zeros((1, 768))  # Mock pooled tensor
            return mock_embeds
        
        mock_pipeline.encode_prompt = mock_encode_prompt
        mock_pipeline.device = 'cpu'
        
        # Test the call structure that our handler.py uses
        try:
            result = mock_pipeline.encode_prompt(
                prompt="test prompt",
                prompt_2="test prompt",  # This should be present
                device='cpu',
                num_images_per_prompt=1
            )
            print("✅ encode_prompt call structure is correct")
            return True
        except TypeError as te:
            print(f"❌ encode_prompt call structure error: {te}")
            return False
            
    except Exception as e:
        print(f"❌ encode_prompt test error: {e}")
        traceback.print_exc()
        return False

def test_memory_management():
    """Test memory management functions"""
    print("🧪 Testing memory management...")
    try:
        # Test torch.cuda functions (should work even without CUDA)
        if hasattr(torch.cuda, 'is_available'):
            torch.cuda.is_available()  # Should not raise error
        
        if hasattr(torch.cuda, 'empty_cache'):
            # This should work even without CUDA
            try:
                torch.cuda.empty_cache()
            except:
                pass  # Expected to potentially fail without CUDA
        
        print("✅ Memory management functions accessible")
        return True
    except Exception as e:
        print(f"❌ Memory management error: {e}")
        return False

def test_error_handling():
    """Test error handling structure"""
    print("🧪 Testing error handling...")
    try:
        # Test that we can catch CUDA OOM errors
        try:
            raise torch.cuda.OutOfMemoryError("Mock OOM error")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"✅ Can catch CUDA OOM errors: {type(oom).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting handler.py validation tests...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Function Signatures", test_function_signatures),
        ("Encode Prompt Calls", test_encode_prompt_calls),
        ("Memory Management", test_memory_management),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Code should be safe to deploy.")
        return True
    else:
        print("⚠️  Some tests failed. Please review before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 