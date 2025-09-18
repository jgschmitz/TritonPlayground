# argmax_reduction.py
import torch, triton, triton.language as tl

@triton.jit
def argmax_kernel(X, IDX, VAL, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = X + row * n_cols
    mask = offs < n_cols
    vals = tl.load(base + offs, mask=mask, other=-float("inf"))
    idxs = offs
    # tree reduce
    for stride in [512,256,128,64,32,16,8,4,2,1]:
        other_vals = tl.swizzle2d(vals, [stride])
        other_idxs = tl.swizzle2d(idxs, [stride])
        take_other = other_vals > vals
        vals = tl.where(take_other, other_vals, vals)
        idxs = tl.where(take_other, other_idxs, idxs)
    tl.store(VAL + row, vals[0])
    tl.store(IDX + row, idxs[0])

def rowwise_argmax(x):
    m, n = x.shape
    BLOCK = triton.next_power_of_2(n)
    idx = torch.empty(m, device="cuda", dtype=torch.int32)
    val = torch.empty(m, device="cuda", dtype=torch.float32)
    argmax_kernel[(m,)](x, idx, val, n, BLOCK=BLOCK)
    return idx, val

if __name__ == "__main__":
    x = torch.randn(1024, 1000, device="cuda")
    idx, val = rowwise_argmax(x)
    ref_idx = x.argmax(dim=1)
    ref_val = x.max(dim=1).values
    torch.testing.assert_close(idx.cpu(), ref_idx.cpu())
    torch.testing.assert_close(val, ref_val)
    print("OK ✅ argmax reduction correct")
