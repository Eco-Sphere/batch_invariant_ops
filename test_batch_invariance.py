import torch
import torch_npu
from batch_invariant_ops import set_batch_invariant_mode
device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)
import sys
"""
optype: 1: mean; 2: log_softmax; 3.matmul
"""
optype = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# Just to get the logging out of the way
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance_matmul(dtype=torch.float32):
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D, dtype=dtype).reshape(B, D)
    b = torch.linspace(-100, 100, D*D, dtype=dtype).reshape(D, D)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    return diff.item() == 0, diff

def test_batch_invariance_mean(dtype=torch.float32):
    B, D, X = 2048, 4096, 16
    a = torch.linspace(-100, 100, B*D*X, dtype=dtype).reshape(B, D, X)

    # Method 1: 
    out1 = torch.mean(a[:1], dim=1, keepdim=False)

    # Method 2:
    out2 = torch.mean(a, dim=1, keepdim=False)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()

    return diff.item() == 0, diff

def test_batch_invariance_log_softmax(dtype=torch.float32):
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D, dtype=dtype).reshape(B, D)

    # Method 1: 
    out1 = torch._log_softmax(a[:1], dim=-1, half_to_float=False)

    # Method 2:
    out2 = torch._log_softmax(a, dim=-1, half_to_float=False)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    return diff.item() == 0, diff

def run_iters(optype=3, iters=10):
    for dtype in [ torch.float32 , torch.bfloat16 ]:
        is_deterministic = True
        difflist = []
        for i in range (iters):
            if optype == 1:
                isd, df = test_batch_invariance_mean(dtype)

            if optype == 2:
                isd, df = test_batch_invariance_log_softmax(dtype)

            if optype == 3:
                isd, df = test_batch_invariance_matmul(dtype)

            is_deterministic = is_deterministic and isd
            difflist.append(df)
        print( f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {dtype} in {iters} iterations")

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    run_iters()

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    run_iters()
