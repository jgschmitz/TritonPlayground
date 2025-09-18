#simple drop in for any script to get timings 
def bench(fn, *args, warmup=10, iters=100):
    torch.cuda.synchronize()
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(*args)
    end.record(); torch.cuda.synchronize()
    print(f"{fn.__name__}: {start.elapsed_time(end)/iters:.3f} ms/run")
