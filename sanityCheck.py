# sanity_check.py
import torch, triton, triton.language as tl

print("Triton version:", triton.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
print("OK ✅ sanity check passed")
