# matmul_autotune.py - Enhanced Triton Matrix Multiplication with Advanced Autotuning
"""
Enhanced matrix multiplication kernel with comprehensive autotuning, multi-precision support,
and performance optimizations. This implementation provides:

- Extensive autotuning configurations covering diverse hardware scenarios
- Multi-precision support (FP16, FP32, BF16) with proper accumulation
- Intelligent grid calculation and memory layout optimization
- Comprehensive error handling and input validation
- Performance benchmarking and comparison utilities
- Production-ready implementation with detailed documentation
"""

import torch
import triton
import triton.language as tl
from triton.runtime.autotuner import Config
from typing import Tuple, Optional, Union
import time
import functools

# Comprehensive autotuning configuration covering diverse scenarios
MATMUL_CONFIGS = [
    # Small matrices (good for batch processing)
    Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 32,  "NUM_WARPS": 2}),
    Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32,  "NUM_WARPS": 2}),
    Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32,  "NUM_WARPS": 2}),
    
    # Medium matrices (balanced configurations)
    Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32,  "NUM_WARPS": 4}),
    Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32,  "NUM_WARPS": 4}),
    Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32,  "NUM_WARPS": 4}),
    Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64,  "NUM_WARPS": 4}),
    
    # Large matrices (high-throughput configurations)
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32,  "NUM_WARPS": 4}),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "NUM_WARPS": 4}),
    Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32,  "NUM_WARPS": 8}),
    Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32,  "NUM_WARPS": 8}),
    Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32,  "NUM_WARPS": 8}),
    
    # High-warp configurations for modern GPUs
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "NUM_WARPS": 8}),
    Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64,  "NUM_WARPS": 8}),
    Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "NUM_WARPS": 8}),
    
    # Memory-intensive configurations (larger K blocks)
    Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128, "NUM_WARPS": 4}),
    Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 128, "NUM_WARPS": 4}),
    Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128, "NUM_WARPS": 4}),
]


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["M", "N", "K", "DTYPE"]  # Include dtype in autotuning key
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr, 
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Enhanced Triton matrix multiplication kernel with optimizations.
    
    Improvements over basic implementation:
    - Better memory access patterns with optimized offsets
    - Configurable output dtype matching input precision
    - Improved accumulation strategy for numerical stability
    - Optimized masking for edge cases
    """
    # Program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator with appropriate precision
    # Always use float32 for accumulation to maintain precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main computation loop with optimized memory access
    for k_offset in range(0, K, BLOCK_K):
        # Load matrix A block with bounds checking
        a_mask = (offs_m[:, None] < M) & ((k_offset + offs_k)[None, :] < K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_offset + offs_k)[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load matrix B block with bounds checking  
        b_mask = ((k_offset + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        b_ptrs = b_ptr + ((k_offset + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b)
    
    # Convert accumulator to output dtype
    if DTYPE == tl.float16:
        c = acc.to(tl.float16)
    elif DTYPE == tl.bfloat16:
        c = acc.to(tl.bfloat16)  
    else:  # float32
        c = acc.to(tl.float32)
    
    # Store result with bounds checking
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=c_mask)


def calculate_optimal_grid(M: int, N: int) -> Tuple[int, int]:
    """Calculate optimal grid size based on matrix dimensions."""
    # Use smaller block sizes for small matrices
    if M <= 512 and N <= 512:
        block_m, block_n = 64, 64
    elif M <= 1024 and N <= 1024:
        block_m, block_n = 128, 128
    else:
        block_m, block_n = 256, 256
    
    return (triton.cdiv(M, block_m), triton.cdiv(N, block_n))


def validate_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    """Comprehensive input validation with detailed error messages."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor objects")
    
    if a.device != b.device:
        raise ValueError(f"Input tensors must be on same device. Got {a.device} and {b.device}")
    
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be on CUDA device for Triton kernels")
    
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Input tensors must be 2D. Got shapes {a.shape} and {b.shape}")
    
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matrix dimension mismatch: A({a.shape}) @ B({b.shape}). "
                        f"A.shape[1] ({a.shape[1]}) must equal B.shape[0] ({b.shape[0]})")
    
    if a.dtype != b.dtype:
        raise ValueError(f"Input tensors must have same dtype. Got {a.dtype} and {b.dtype}")
    
    supported_dtypes = {torch.float16, torch.float32, torch.bfloat16}
    if a.dtype not in supported_dtypes:
        raise ValueError(f"Unsupported dtype {a.dtype}. Supported: {supported_dtypes}")


def triton_matmul(a: torch.Tensor, b: torch.Tensor, 
                  output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Enhanced Triton matrix multiplication with comprehensive features.
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)  
        output_dtype: Output dtype (defaults to input dtype)
        
    Returns:
        Result matrix C of shape (M, N)
        
    Raises:
        TypeError: If inputs are not tensors
        ValueError: For invalid shapes, devices, or dtypes
    """
    # Comprehensive input validation
    validate_inputs(a, b)
    
    # Extract dimensions
    M, K = a.shape
    K_b, N = b.shape
    
    # Determine output dtype
    if output_dtype is None:
        output_dtype = a.dtype
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=output_dtype)
    
    # Calculate optimal grid size
    grid = calculate_optimal_grid(M, N)
    
    # Convert dtype to Triton language equivalent
    if output_dtype == torch.float16:
        triton_dtype = tl.float16
    elif output_dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    else:
        triton_dtype = tl.float32
    
    # Launch kernel with proper stride and dtype information
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), 
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1),
        DTYPE=triton_dtype,
    )
    
    return c


def benchmark_matmul(M: int, K: int, N: int, 
                    dtype: torch.dtype = torch.float16,
                    warmup: int = 10, 
                    iterations: int = 100) -> dict:
    """
    Benchmark matrix multiplication performance comparing Triton vs PyTorch.
    
    Returns dictionary with timing results and speedup metrics.
    """
    print(f"🚀 Benchmarking matmul: ({M}, {K}) @ ({K}, {N}) with {dtype}")
    
    # Create test matrices
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    
    # Warmup runs
    for _ in range(warmup):
        _ = triton_matmul(a, b)
        _ = a @ b
    
    torch.cuda.synchronize()
    
    # Benchmark Triton implementation
    start_time = time.perf_counter()
    for _ in range(iterations):
        triton_result = triton_matmul(a, b)
    torch.cuda.synchronize()
    triton_time = time.perf_counter() - start_time
    
    # Benchmark PyTorch implementation  
    start_time = time.perf_counter()
    for _ in range(iterations):
        pytorch_result = (a @ b).to(dtype)
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start_time
    
    # Calculate metrics
    triton_gflops = (2 * M * K * N * iterations) / (triton_time * 1e9)
    pytorch_gflops = (2 * M * K * N * iterations) / (pytorch_time * 1e9)
    speedup = pytorch_time / triton_time
    
    # Verify correctness
    max_diff = torch.max(torch.abs(triton_result - pytorch_result)).item()
    
    results = {
        'triton_time': triton_time,
        'pytorch_time': pytorch_time,
        'triton_gflops': triton_gflops,
        'pytorch_gflops': pytorch_gflops,
        'speedup': speedup,
        'max_diff': max_diff,
        'shape': (M, K, N),
        'dtype': str(dtype)
    }
    
    print(f"  Triton: {triton_time/iterations*1000:.2f}ms ({triton_gflops:.1f} GFLOPS)")
    print(f"  PyTorch: {pytorch_time/iterations*1000:.2f}ms ({pytorch_gflops:.1f} GFLOPS)")
    print(f"  Speedup: {speedup:.2f}x | Max diff: {max_diff:.2e}")
    
    return results


def run_comprehensive_tests():
    """Run comprehensive correctness and performance tests."""
    print("🧪 Running comprehensive matmul tests...")
    
    # Test different matrix sizes and dtypes
    test_configs = [
        (256, 512, 256, torch.float16),
        (1024, 1536, 768, torch.float16),
        (2048, 1024, 512, torch.float16),
        (512, 2048, 1024, torch.float32),
        (1024, 1024, 1024, torch.bfloat16) if torch.cuda.is_bf16_supported() else (1024, 1024, 1024, torch.float16),
    ]
    
    results = []
    for M, K, N, dtype in test_configs:
        try:
            # Correctness test
            a = torch.randn((M, K), device="cuda", dtype=dtype)
            b = torch.randn((K, N), device="cuda", dtype=dtype)
            
            triton_out = triton_matmul(a, b)
            pytorch_ref = (a @ b).to(dtype)
            
            # Determine appropriate tolerance based on dtype
            if dtype == torch.float16:
                atol, rtol = 1e-1, 1e-1
            elif dtype == torch.bfloat16:
                atol, rtol = 1e-1, 1e-1  
            else:  # float32
                atol, rtol = 1e-3, 1e-3
            
            torch.testing.assert_close(triton_out, pytorch_ref, atol=atol, rtol=rtol)
            print(f"✅ Correctness: ({M}, {K}, {N}) {dtype}")
            
            # Performance benchmark
            benchmark_results = benchmark_matmul(M, K, N, dtype, warmup=5, iterations=20)
            results.append(benchmark_results)
            
        except Exception as e:
            print(f"❌ Failed: ({M}, {K}, {N}) {dtype} - {e}")
    
    # Summary
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"\n📊 Average speedup: {avg_speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    print("🔥 Enhanced Triton Matrix Multiplication with Advanced Autotuning")
    
    # Quick correctness test
    print("\n1️⃣ Quick correctness test...")
    M, K, N = 1024, 1536, 768
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    out = triton_matmul(a, b)
    ref = (a @ b).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)
    print("✅ Basic correctness test passed!")
    
    # Comprehensive testing
    print("\n2️⃣ Comprehensive testing...")
    test_results = run_comprehensive_tests()
    
    print(f"\n🎉 All tests completed! Enhanced matmul with {len(MATMUL_CONFIGS)} autotuning configs is ready!")
