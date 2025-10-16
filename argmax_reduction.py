# argmax_reduction.py - Enhanced Triton argmax kernel with improved robustness
"""
Row-wise argmax reduction using Triton with the following improvements:
- Increased BLOCK size to 2048 for better performance on wide matrices
- Enhanced input validation and error handling
- Simplified negative sentinel creation
- Better type hints and documentation
- Edge case handling for empty tensors
"""
import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def argmax_kernel(
    X,
    IDX,
    VAL,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Triton kernel for computing row-wise argmax.
    
    Args:
        X: Input tensor pointer (2D, row-major)
        IDX: Output indices pointer (1D, int64)
        VAL: Output values pointer (1D, float32)
        n_cols: Number of columns per row
        BLOCK: Block size for vectorization
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)  # [0..BLOCK)
    mask = offs < n_cols

    # Load one row slice (pad with -inf outside valid range)
    x = tl.load(X + row * n_cols + offs, mask=mask, other=-float("inf"))

    # 1) Max value across the block
    maxv = tl.max(x, axis=0)

    # 2) Index of the max (largest index on ties)
    offs_i64 = offs.to(tl.int64)
    # Simplified negative sentinel creation
    neg1_i64 = tl.full((BLOCK,), -1, dtype=tl.int64)
    cand = tl.where(x == maxv, offs_i64, neg1_i64)
    idx64 = tl.max(cand, axis=0)

    # Write out results
    tl.store(VAL + row, maxv)
    tl.store(IDX + row, idx64)


def rowwise_argmax(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute row-wise argmax using Triton kernel.
    
    Args:
        x: Input tensor of shape (M, N), must be on CUDA device and float32
        
    Returns:
        Tuple of (indices, values):
        - indices: int64 tensor of shape (M,) with argmax indices for each row
        - values: float32 tensor of shape (M,) with max values for each row
        
    Raises:
        AssertionError: If tensor is not on CUDA, not 2D, or not float32
        ValueError: If tensor has zero columns
    """
    # Enhanced input validation
    assert x.is_cuda, f"Input tensor must be on CUDA device, got device: {x.device}"
    assert x.dim() == 2, f"Input tensor must be 2D, got {x.dim()}D"
    assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
    
    m, n = x.shape
    
    # Handle edge case: empty tensor
    if n == 0:
        raise ValueError("Cannot compute argmax on tensor with 0 columns")
    if m == 0:
        return (
            torch.empty(0, device=x.device, dtype=torch.int64),
            torch.empty(0, device=x.device, dtype=torch.float32)
        )
    
    # Increased BLOCK size for better performance on wide matrices
    BLOCK = min(triton.next_power_of_2(n), 2048)
    
    # Allocate output tensors
    idx = torch.empty(m, device=x.device, dtype=torch.int64)
    val = torch.empty(m, device=x.device, dtype=torch.float32)
    
    # Launch kernel
    argmax_kernel[(m,)](x, idx, val, n, BLOCK=BLOCK)
    
    return idx, val

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, 1000, device="cuda", dtype=torch.float32)
    idx, val = rowwise_argmax(x)
    ref_idx = x.argmax(dim=1)
    ref_val = x.max(dim=1).values
    torch.testing.assert_close(idx.cpu(), ref_idx.cpu())  # dtypes now match
    torch.testing.assert_close(val, ref_val)
    print("OK ✅ argmax reduction correct")
