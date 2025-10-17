#!/usr/bin/env python3
# layernorm_gelu_bench.py
# Fancy benchmark + correctness for fused Triton LayerNorm+GELU vs PyTorch LN->GELU

import argparse
import math
import time
import torch
import triton
import triton.language as tl

# -----------------------------
# Pretty console (optional)
# -----------------------------
USE_RICH = True
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
except Exception:
    USE_RICH = False

def log(*args, **kwargs):
    if USE_RICH:
        rprint(*args, **kwargs)
    else:
        print(*args, **kwargs)

# -----------------------------
# Triton kernel (tiled, 2-pass)
# -----------------------------
@triton.jit
def ln_gelu_kernel_tiled(X, Y, gamma, beta, n_cols,
                         EPS: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_base = row * n_cols

    # ---- Pass 1: mean / var across tiles ----
    total_sum = tl.zeros([1], dtype=tl.float32)
    total_sumsq = tl.zeros([1], dtype=tl.float32)

    col = 0
    while col < n_cols:
        offs = col + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(X + row_base + offs, mask=mask, other=0.0)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)
        col += BLOCK

    n_cols_f32 = tl.full([1], n_cols, tl.float32)
    inv_n = 1.0 / n_cols_f32
    mu = total_sum * inv_n
    var = total_sumsq * inv_n - mu * mu
    denom = tl.sqrt(var + 1e-5)

    # ---- Pass 2: normalize + affine + GELU + store ----
    col = 0
    while col < n_cols:
        offs = col + tl.arange(0, BLOCK)
        mask = offs < n_cols

        x = tl.load(X + row_base + offs, mask=mask, other=0.0)
        g = tl.load(gamma + offs, mask=mask, other=1.0)
        b = tl.load(beta  + offs, mask=mask, other=0.0)

        y = (x - mu) / denom
        y = y * g + b

        # GELU (erf form; portable)
        y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

        tl.store(Y + row_base + offs, y, mask=mask)
        col += BLOCK

def fused_ln_gelu(x: torch.Tensor,
                  gamma: torch.Tensor,
                  beta: torch.Tensor,
                  block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Use CUDA tensors"
    x, gamma, beta = x.contiguous(), gamma.contiguous(), beta.contiguous()
    m, n = x.shape
    y = torch.empty_like(x)
    BLOCK = block_size
    num_warps = 4 if BLOCK <= 128 else 8
    ln_gelu_kernel_tiled[(m,)](
        x, y, gamma, beta, n,
        EPS=1e-5, BLOCK=BLOCK,
        num_warps=num_warps, num_stages=2
    )
    return y

# -----------------------------
# Benchmark utilities
# -----------------------------
def cuda_ms(fn, iters=10, warmup=3):
    # warmup
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        out = fn()
    t1.record(); torch.cuda.synchronize()
    return (t0.elapsed_time(t1) / iters), out

def to_dtype(name: str):
    name = name.lower()
    if name in ("fp32", "float32", "f32"):
        return torch.float32
    if name in ("fp16", "half", "f16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")

def parse_sizes(size_list):
    pairs = []
    for s in size_list:
        if "x" not in s:
            raise ValueError(f"Size '{s}' must be in MxN form, e.g., 512x2048")
        m, n = s.lower().split("x")
        pairs.append((int(m), int(n)))
    return pairs

# -----------------------------
# Optional plotting
# -----------------------------
def maybe_plot(results, out_png="ln_gelu_bench.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        log("[yellow]matplotlib not found; skipping chart[/yellow]")
        return
    labels = [f"{m}x{n}" for (m, n) in results["sizes"]]
    triton_ms = results["triton_ms"]
    torch_ms  = results["torch_ms"]
    x = range(len(labels))

    plt.figure(figsize=(10, 4))
    plt.bar(x, torch_ms, label="PyTorch LN→GELU")
    plt.bar(x, triton_ms, bottom=None, label="Triton fused LN+GELU")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Time (ms, avg)")
    plt.title("LayerNorm+GELU Benchmark (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    log(f"[green]Saved chart to[/green] {out_png}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fused LayerNorm+GELU Triton vs PyTorch benchmark")
    parser.add_argument("--sizes", nargs="+", default=["512x2048", "1024x4096"],
                        help="List of MxN, e.g., 512x2048 1024x4096")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32|fp16|bf16")
    parser.add_argument("--block", type=int, default=1024, help="Triton BLOCK size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--plot", action="store_true", help="save a bar chart PNG")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = to_dtype(args.dtype)
    sizes = parse_sizes(args.sizes)
    device = "cuda"

    # Prepare table
    console = Console() if USE_RICH else None
    if USE_RICH:
        table = Table(title="Fused LayerNorm + GELU Benchmark", expand=True)
        table.add_column("Size (MxN)", justify="center")
        table.add_column("DType", justify="center")
        table.add_column("PyTorch (ms)", justify="right")
        table.add_column("Triton (ms)", justify="right")
        table.add_column("Speedup", justify="right")
        table.add_column("Max |Δ|", justify="right")
        table.add_column("Max rel Δ", justify="right")
    else:
        log("Size\tDType\tPyTorch(ms)\tTriton(ms)\tSpeedup\tMaxAbsErr\tMaxRelErr")

    triton_ms_list, torch_ms_list, sizes_list = [], [], []

    # Loop sizes
    iterator = sizes
    if USE_RICH:
        iterator = track(sizes, description="Benchmarking...")

    for (m, n) in iterator:
        # Data
        x = torch.randn(m, n, device=device, dtype=dtype)
        gamma = torch.randn(n, device=device, dtype=dtype)
        beta  = torch.randn(n, device=device, dtype=dtype)

        # Warmups before timed runs
        _ = fused_ln_gelu(x, gamma, beta, block_size=args.block)
        _ = torch.nn.functional.gelu(torch.nn.functional.layer_norm(x, (n,), gamma, beta, 1e-5))
        torch.cuda.synchronize()

        # Timed PyTorch
        pt_ms, y_ref = cuda_ms(
            lambda: torch.nn.functional.gelu(
                torch.nn.functional.layer_norm(x, (n,), gamma, beta, 1e-5)
            ),
            iters=args.iters, warmup=args.warmup
        )

        # Timed Triton
        tr_ms, y = cuda_ms(
            lambda: fused_ln_gelu(x, gamma, beta, block_size=args.block),
            iters=args.iters, warmup=args.warmup
        )

        # Correctness (compute in fp32 for error metrics)
        y32    = y.float()
        yref32 = y_ref.float()
        abs_err = (y32 - yref32).abs()
        max_abs = abs_err.max().item()
        rel_err = abs_err / (yref32.abs() + 1e-12)
        max_rel = rel_err.max().item()

        # Assert tightness in fp32 path
        if dtype == torch.float32:
            torch.testing.assert_close(y32, yref32, atol=1e-5, rtol=1e-5)

        speedup = pt_ms / tr_ms if tr_ms > 0 else float("inf")

        if USE_RICH:
            table.add_row(
                f"{m}x{n}", str(dtype).replace("torch.", ""),
                f"{pt_ms:,.3f}", f"{tr_ms:,.3f}", f"{speedup:,.2f}×",
                f"{max_abs:.2e}", f"{max_rel:.2e}"
            )
        else:
            log(f"{m}x{n}\t{dtype}\t{pt_ms:.3f}\t{tr_ms:.3f}\t{speedup:.2f}x\t{max_abs:.2e}\t{max_rel:.2e}")

        triton_ms_list.append(tr_ms)
        torch_ms_list.append(pt_ms)
        sizes_list.append((m, n))

    if USE_RICH:
        console.print(table)

    results = {
        "sizes": sizes_list,
        "triton_ms": triton_ms_list,
        "torch_ms": torch_ms_list
    }
    if args.plot:
        maybe_plot(results)

    # Pretty footer
    log("\n[green]Done![/green] Compare times above; Triton should be faster as sizes grow. 🚀")

if __name__ == "__main__":
    main()
