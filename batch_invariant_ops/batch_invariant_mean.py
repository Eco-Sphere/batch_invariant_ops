import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, K, N) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices of the first sub-block
    m_idx = pid * BLOCK_SIZE // K
    k_idx = pid * BLOCK_SIZE % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    for i in range(0, BLOCK_SIZE):
        if m_idx >= M:
            pass
        acc = 0.0
        for n_start in range(0, N, SUB_BLOCK_SIZE):
            n_offsets = n_start + tl.arange(0, SUB_BLOCK_SIZE)
            mask = n_offsets < N

            # Calculate input indices
            input_idx = (
                m_idx * input_stride0 + k_idx * input_stride1 + n_offsets * input_stride2
            )

            # Load and accumulate
            vals = tl.load(input_ptr + input_idx, mask = mask, other = 0.0)
            acc += tl.sum(vals)

        # Compute mean and store
        mean_val = acc / N
        output_idx = m_idx * output_stride0 + k_idx * output_stride1
        tl.store(output_ptr + output_idx, mean_val)

        # Update indices for next iteration
        k_idx += 1
        if k_idx >= K:
            k_idx = 0
            m_idx += 1
            
def mean_dim(
    input:torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert "npu" in str(input.device).lower(), "Input must be a npu tensor"
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, K, N)
    input_3d = input.reshape(M, N, K)
    input_3d = input_3d.transpose(1, 2).contiguous()

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    num_core = get_npu_properties()["num_vectorcore"]
    grid = (num_core,)
    BLOCK_SIZE = triton.cdiv(M * K,  num_core)
    SUB_BLOCK_SIZE = 4096

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
        SUB_BLOCK_SIZE,
    )

    return output

def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems
