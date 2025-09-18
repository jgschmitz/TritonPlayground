# vec_add.py
import torch, triton, triton.language as tl

@triton.jit
def vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def vec_add(x, y):
    assert x.is_cuda and y.is_cuda and x.numel() == y.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (triton.cdiv(x.numel(), BLOCK),)
    vec_add_kernel[grid](x, y, out, x.numel(), BLOCK=BLOCK)
    return out

if __name__ == "__main__":
    n = 1_000_000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = vec_add(x, y)
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    print("OK ✅ vec_add correct")
