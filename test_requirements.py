#!/usr/bin/env python3
"""
Test script to validate that requirements.txt dependencies work correctly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import triton
        print(f"✅ Triton {triton.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Triton import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True


def test_cuda_availability():
    """Test CUDA availability for GPU operations."""
    import torch
    
    print(f"\nTesting CUDA availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        print(f"✅ Current CUDA device: {torch.cuda.current_device()}")
        return True
    else:
        print("⚠️  CUDA is not available (CPU-only mode)")
        return False


def test_basic_operations():
    """Test basic tensor operations."""
    import torch
    
    print(f"\nTesting basic tensor operations...")
    try:
        # Test CPU operations
        x = torch.randn(10, 10)
        y = x.sum()
        print(f"✅ CPU tensor operations work")
        
        # Test GPU operations if available
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            y_cuda = x_cuda.sum()
            print(f"✅ CUDA tensor operations work")
        
        return True
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Requirements Validation Test ===\n")
    
    success = True
    success &= test_imports()
    success &= test_cuda_availability()  # Note: CUDA might not be available in all environments
    success &= test_basic_operations()
    
    print(f"\n=== Test Results ===")
    if success:
        print("🎉 All tests passed! The environment is ready for Triton development.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")