
import time
import torch

if not torch.backends.mps.is_available():
    print("Apple GPU acceleration is unavailable.")
    print("Check the PyTorch installation and macOS version.")
    raise SystemExit(1)

device = torch.device("mps")
print("Using device:", device)

a = torch.randn(3000, 3000, device=device)
b = torch.randn(3000, 3000, device=device)

start = time.time()
c = torch.matmul(a, b)

# Synchronize so the timing includes GPU work.
torch.mps.synchronize()

print("Matrix result shape:", tuple(c.shape))
print("GPU computation completed.")
print("Elapsed seconds:", round(time.time() - start, 3))
print("Result device:", c.device)
