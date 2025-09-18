# softmax_triton.py
import torch, triton, triton.language as tl

@triton.jit
def softmax_kernel(X, Y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * n_cols + offs
    mask = offs < n_cols
    x = tl.load(ptr, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)              # numerical stability
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(Y + row * n_cols + offs, num / den, mask=mask)

def triton_softmax(x):
    assert x.is_cuda and x.dim() == 2
    m, n = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n)
    grid = (m,)
    softmax_kernel[grid](x, y, n, BLOCK=BLOCK)
    return y

if __name__ == "__main__":
    x = torch.randn(1024, 1536, device="cuda", dtype=torch.float32)
    y = triton_softmax(x)
    torch.testing.assert_close(y, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)
    print("OK ✅ softmax correct")
