import torch
from batch_invariant_ops import set_batch_invariant_mode
device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

with set_batch_invariant_mode(True):
    pass

def test_batch_invariance_mean(dtype=torch.float32):
    B, D, K = 2048, 4096,64
    a = torch.linspace(-100, 100,B*D*K, dtype=dtype).reshape(B, D, K)

    out1 = torch.mean(a[:1], dim=1)
    out2 = torch.mean(a, dim=1)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    return diff.item == 0, diff

def run_iters(iters=10):
    for dtype in [torch.float32, torch.float16]:
        is_deterministic = True
        difflist = []
    for i in range(iters):
        isd, df = test_batch_invariance_mean(dtype)
        is_deterministic = is_deterministic and isd
        difflist.append(df)
    print(f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {dtype} in {iters} iterations")

# Test with standard PyTorch
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    run_iters()

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    run_iters()