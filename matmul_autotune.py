# matmul_autotune.py
import torch, triton, triton.language as tl
from triton.runtime.autotuner import Config

@triton.autotune(
    configs=[
        Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_WARPS": 4}),
        Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32, "NUM_WARPS": 8}),
        Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "NUM_WARPS": 4}),
    ],
    key=["M","N","K"]
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak),
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn),
                    mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
             c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_matmul(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c

if __name__ == "__main__":
    M, K, N = 1024, 1536, 768
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    out = triton_matmul(a, b)
    ref = (a @ b).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)  # fp16 tolerance
    print("OK ✅ matmul correct (autotuned)")
