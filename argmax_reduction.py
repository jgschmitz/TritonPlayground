# argmax.py  (int64 indices)
import torch, triton, triton.language as tl

@triton.jit
def argmax_kernel(X, IDX, VAL, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)                      # [0..BLOCK)
    mask = offs < n_cols

    # Load one row slice (pad with -inf outside valid range)
    x = tl.load(X + row * n_cols + offs, mask=mask, other=-float("inf"))

    # 1) Max value across the block
    maxv = tl.max(x, axis=0)

    # 2) Index of the max (largest index on ties)
    offs_i64 = offs.to(tl.int64)
    neg1_i64 = tl.zeros((BLOCK,), dtype=tl.int64) - 1
    cand = tl.where(x == maxv, offs_i64, neg1_i64)
    idx64 = tl.max(cand, axis=0)

    # Write out
    tl.store(VAL + row, maxv)
    tl.store(IDX + row, idx64)

def rowwise_argmax(x: torch.Tensor):
    assert x.is_cuda and x.dim() == 2
    m, n = x.shape
    BLOCK = min(triton.next_power_of_2(n), 1024)
    idx = torch.empty(m, device=x.device, dtype=torch.int64)   # <-- int64 to match torch.argmax
    val = torch.empty(m, device=x.device, dtype=torch.float32)
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
