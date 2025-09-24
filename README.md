# Triton Kernel Playgrounds 🧪⚡

Tiny, focused Python scripts to **smoke‑test your [OpenAI Triton](https://github.com/openai/triton) install** and learn the core GPU‑kernel patterns: tiling, masking, reductions, fusion, and autotuning. Each script validates correctness against PyTorch and includes a tiny benchmarking helper. I recommend a GCP image with Cuda and NVidia Drivers pre-installed makes it lots easier - basic knowledge of triton is important if you are using unsloth -> https://unsloth.ai

> **Requires:** NVIDIA GPU + CUDA, Python 3.9–3.12, recent **PyTorch** and **Triton**.

---

## TL;DR

```bash
# 1) Create & activate an env (pick one):
conda create -n triton-lab python=3.10 -y && conda activate triton-lab
#   or using venv
# python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip
pip install torch triton

# 3) Run a quick check
python scripts/sanity_check.py

# 4) Try a kernel
python scripts/vec_add.py
```

---

## What’s inside

| Script                    | Concept           | Highlights                                                    |
| ------------------------- | ----------------- | ------------------------------------------------------------- |
| `sanity_check.py`         | Environment       | Print Triton version, CUDA device, quick readiness check.     |
| `vec_add.py`              | Basics            | Program IDs, block offsets, masked loads/stores.              |
| `softmax_triton.py`       | Reductions        | Per‑row softmax with `tl.max`, `tl.sum`, numerical stability. |
| `matmul_autotune.py`      | Tiling + Autotune | Blocked matmul kernel with autotuning configs.                |
| `layernorm_gelu_fused.py` | Fusion            | Fused LayerNorm + GELU forward kernel.                        |
| `argmax_reduction.py`     | Reductions        | Row‑wise argmax via custom reduction pattern.                 |

---

## Repo layout

```text
.
├── scripts/
│   ├── sanity_check.py
│   ├── vec_add.py
│   ├── softmax_triton.py
│   ├── matmul_autotune.py
│   ├── layernorm_gelu_fused.py
│   └── argmax_reduction.py
├── .gitignore
└── README.md
```

> Place the sample scripts in `scripts/` (names above). Each script prints an `OK ✅` on success.

---

## Usage

Run any script directly:

```bash
python scripts/softmax_triton.py
```

Most scripts compare against a PyTorch reference using `torch.testing.assert_close`. If you see an assertion error, check dtypes, shapes, and GPU capability.

---

## Benchmark helper

Drop this into any script to time kernels:

```python
def bench(fn, *args, warmup=10, iters=100):
    import torch
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record(); torch.cuda.synchronize()
    print(f"{fn.__name__}: {start.elapsed_time(end)/iters:.3f} ms/run")
```

Example usage inside a script:

```python
# after defining `triton_softmax(x)`
bench(triton_softmax, x)
```

---

## Troubleshooting

* **`torch.cuda.is_available() == False`** → install a CUDA‑enabled PyTorch build and NVIDIA drivers.
* **`ValueError: Out of shared memory / registers`** → try smaller `BLOCK_*` sizes or fewer warps.
* **Wrong results at row edges** → verify masks: `mask = offsets < n_elements` and pass `mask=mask, other=0` to `tl.load`/`tl.store`.
* **Autotune very slow first run** → first compile can take time; subsequent runs are cached.

* ### Compatibility
> Triton intrinsics vary by version (e.g., some builds lack `tl.tanh`). This repo uses QuickGELU in the fused LN+activation to avoid that mismatch. If you pin different Torch/Triton combos, adjust as needed.

| PyTorch | Triton | Notes                |
|--------:|:------:|----------------------|
| 2.3.x   | 2.1.x  | OK (default tested)  |

### Notes
- **Argmax ties:** returns the **largest** index. Swap to “first” by reducing with `min` (sentinel trick), as shown in the script comments.
- **Kernel cache:** set a custom cache dir with `export TRITON_CACHE_DIR=./.triton`.

---

## Contributing

PRs welcome! Ideas: backward‑pass kernels, more fusion patterns, dynamic shapes, FP8 paths, multi‑GPU.

---

## License

MIT
