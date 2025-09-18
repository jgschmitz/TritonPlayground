# layernorm_gelu_fused.py
import torch, triton, triton.language as tl

@triton.jit
def ln_gelu_kernel(X, Y, gamma, beta, n_cols, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * n_cols + offs, mask=mask, other=0.0)
    mu = tl.sum(x, axis=0) / n_cols
    diff = x - mu
    var = tl.sum(diff * diff, axis=0) / n_cols
    x_hat = diff / tl.sqrt(var + EPS)
    g = tl.load(gamma + offs, mask=mask, other=1.0)
    b = tl.load(beta + offs,  mask=mask, other=0.0)
    y = x_hat * g + b
    # GELU approx
    y = 0.5 * y * (1.0 + tl.tanh(0.79788456 * (y + 0.044715 * y * y * y)))
    tl.store(Y + row * n_cols + offs, y, mask=mask)

def fused_ln_gelu(x, gamma, beta, eps=1e-5):
    m, n = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n)
    ln_gelu_kernel[(m,)](x, y, gamma, beta, n, EPS=eps, BLOCK=BLOCK)
    return y

if __name__ == "__main__":
    m, n = 512, 2048
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    gamma = torch.randn(n, device="cuda", dtype=torch.float32)
    beta  = torch.randn(n, device="cuda", dtype=torch.float32)
    y = fused_ln_gelu(x, gamma, beta)
    # Reference
    x_ref = torch.nn.functional.layer_norm(x, (n,), gamma, beta, 1e-5)
    y_ref = torch.nn.functional.gelu(x_ref)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    print("OK ✅ fused layernorm+gelu correct")
