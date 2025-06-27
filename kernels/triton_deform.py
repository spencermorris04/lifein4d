# kernels/triton_deform.py

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def deform_update_kernel(
    mu_ptr,          # [N, 3] float32
    W_ptr,           # [N, 3, r] float16  
    b_ptr,           # [r] float16
    mu_out_ptr,      # [N, 3] float32
    visibility_mask_ptr,  # [N] bool - optional visibility mask
    N: tl.constexpr,
    r: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused matrix-vector: μ_out = μ + W @ b(t)
    Uses only masking and tl.where for all conditionals.
    """
    pid = tl.program_id(axis=0)
    
    # Load time coefficients (all threads load the same data)
    b_offsets = tl.arange(0, r)
    b = tl.load(b_ptr + b_offsets, mask=b_offsets < r, other=0.0).to(tl.float32)
    
    # Process each splat in the block using static range
    for block_offset in tl.static_range(BLOCK_SIZE):
        splat_idx = pid * BLOCK_SIZE + block_offset
        
        # Create mask for valid splat
        valid_splat = splat_idx < N
        
        # Load original centroid
        mu_offsets = tl.arange(0, 3)
        mu = tl.load(mu_ptr + splat_idx * 3 + mu_offsets, mask=valid_splat, other=0.0)
        
        # Determine if we should apply deformation
        should_deform = valid_splat
        if visibility_mask_ptr is not None:
            visibility = tl.load(visibility_mask_ptr + splat_idx, mask=valid_splat, other=False)
            should_deform = should_deform & visibility
        
        # Compute deformation delta
        delta = tl.zeros([3], dtype=tl.float32)
        
        # Only compute deformation for valid, visible splats
        W_base = W_ptr + splat_idx * 3 * r
        for j in tl.static_range(r):
            W_col_offsets = tl.arange(0, 3) * r + j
            W_col = tl.load(W_base + W_col_offsets, mask=valid_splat, other=0.0).to(tl.float32)
            delta += W_col * b[j]
        
        # Apply deformation conditionally
        # Use scalar multiplication for conditional application
        deform_factor = tl.where(should_deform, 1.0, 0.0)
        result = mu + delta * deform_factor
        
        # Store result (only for valid splats)
        tl.store(mu_out_ptr + splat_idx * 3 + mu_offsets, result, mask=valid_splat)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['H', 'W'],
)
@triton.jit
def depth_residual_kernel(
    id_buffer_ptr,   # [H, W] int32 - splat indices from rasterizer
    mu_z_ptr,        # [N] float32 - z-coordinates of splats
    depth_ptr,       # [H, W] float32 - ground truth depth
    confidence_ptr,  # [H, W] float32 - depth confidence weights
    grad_mu_z_ptr,   # [N] float32 - output gradients (atomic add)
    loss_accumulator_ptr,  # [num_blocks] float32 - per-block loss accumulation
    valid_pixel_count_ptr, # [num_blocks] int32 - per-block valid pixel count
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HUBER_DELTA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized depth residual computation using only masking.
    """
    pid = tl.program_id(axis=0)
    
    # Per-block accumulators
    block_loss = tl.float32(0.0)
    block_valid_count = 0
    
    # Process each pixel in the block using static range
    for i in tl.static_range(BLOCK_SIZE):
        pixel_idx = pid * BLOCK_SIZE + i
        
        # Create masks for valid conditions
        pixel_valid = pixel_idx < (H * W)
        
        # Load data with masking
        splat_id = tl.load(id_buffer_ptr + pixel_idx, mask=pixel_valid, other=-1).to(tl.int32)
        gt_depth = tl.load(depth_ptr + pixel_idx, mask=pixel_valid, other=0.0)
        confidence = tl.load(confidence_ptr + pixel_idx, mask=pixel_valid, other=0.0)
        
        # Create validity conditions
        splat_valid = (splat_id >= 0) & (splat_id < N)
        depth_valid = (gt_depth > 0.0) & (confidence > 0.0)
        all_valid = pixel_valid & splat_valid & depth_valid
        
        # Load predicted depth
        pred_depth = tl.load(mu_z_ptr + splat_id, mask=all_valid, other=0.0)
        
        # Compute residual and loss (only when valid)
        residual = pred_depth - gt_depth
        abs_residual = tl.abs(residual)
        
        # Smooth L1 (Huber) loss
        loss_value = tl.where(
            abs_residual <= HUBER_DELTA,
            0.5 * residual * residual / HUBER_DELTA,  # Quadratic region
            abs_residual - 0.5 * HUBER_DELTA         # Linear region
        )
        
        grad_value = tl.where(
            abs_residual <= HUBER_DELTA,
            residual / HUBER_DELTA,                   # Quadratic gradient
            tl.where(residual > 0, 1.0, -1.0)        # Linear gradient
        )
        
        # Apply confidence weighting and validity mask
        weighted_loss = tl.where(all_valid, confidence * loss_value, 0.0)
        weighted_grad = tl.where(all_valid, confidence * grad_value, 0.0)
        valid_count = tl.where(all_valid, 1, 0)
        
        # Accumulate in block-local variables
        block_loss += weighted_loss
        block_valid_count += valid_count
        
        # Atomic gradient accumulation
        # Always do the atomic add, but weighted_grad will be 0 for invalid pixels
        tl.atomic_add(grad_mu_z_ptr + tl.where(all_valid, splat_id, 0), weighted_grad)
    
    # Store block-level results  
    tl.store(loss_accumulator_ptr + pid, block_loss)
    tl.store(valid_pixel_count_ptr + pid, tl.int32(block_valid_count))


class DeformUpdateFunction(Function):
    @staticmethod
    def forward(ctx, mu, W, b, visibility_mask=None):
        """
        Forward pass of deformation update.
        
        Args:
            mu: [N, 3] splat centroids (fp32)
            W: [N, 3, r] deformation basis (fp16)
            b: [r] time coefficients (fp16)
            visibility_mask: [N] optional visibility mask (bool)
            
        Returns:
            mu_deformed: [N, 3] deformed centroids (fp32)
        """
        N, _ = mu.shape
        r = b.shape[0]
        
        # Allocate output
        mu_out = torch.empty_like(mu)
        
        # Launch configuration - use blocks for better performance
        BLOCK_SIZE = 256
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        
        # Prepare visibility mask pointer
        visibility_ptr = visibility_mask if visibility_mask is not None else None
        
        deform_update_kernel[grid](
            mu, W, b, mu_out, visibility_ptr,
            N=N, r=r, BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Save for backward - don't mark dirty since we return new tensor
        ctx.save_for_backward(W, b, visibility_mask)
        ctx.r = r
        
        return mu_out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with correct dtype handling.
        """
        W, b, visibility_mask = ctx.saved_tensors
        r = ctx.r
        
        # Gradient w.r.t. original centroids (pass-through)
        grad_mu = grad_output
        
        # Apply visibility mask to gradients if provided
        if visibility_mask is not None:
            grad_mu = grad_mu * visibility_mask.unsqueeze(-1).float()
        
        # Gradient w.r.t. W: grad_W[n, i, j] = grad_output[n, i] * b[j]
        grad_W = torch.mul(grad_output.unsqueeze(-1), b.float())  # [N, 3, r]
        
        # Gradient w.r.t. b: sum over visible splats
        if visibility_mask is not None:
            masked_grad = grad_output * visibility_mask.unsqueeze(-1).float()
            grad_b = torch.sum(W.float() * masked_grad.unsqueeze(-1), dim=(0, 1))
        else:
            grad_b = torch.sum(W.float() * grad_output.unsqueeze(-1), dim=(0, 1))
        
        # Return correct dtypes: fp32 for mu, fp16 for W, fp32 for b (optimizer expects fp32)
        return grad_mu, grad_W.half(), grad_b.float(), None


class DepthResidualFunction(Function):
    @staticmethod
    def forward(ctx, mu_z, id_buffer, depth, confidence, huber_delta=0.01):
        """
        Forward pass with corrected scaling.
        """
        H, W = depth.shape
        N = mu_z.shape[0]
        
        # Calculate number of blocks
        BLOCK_SIZE = 256  # Will be auto-tuned
        num_blocks = triton.cdiv(H * W, BLOCK_SIZE)
        
        # Allocate per-block accumulators
        loss_accumulator = torch.zeros(num_blocks, device=mu_z.device, dtype=torch.float32)
        valid_pixel_count = torch.zeros(num_blocks, device=mu_z.device, dtype=torch.int32)
        grad_mu_z = torch.zeros_like(mu_z)
        
        # Launch kernel with block-level accumulation
        grid = (num_blocks,)
        
        depth_residual_kernel[grid](
            id_buffer, mu_z, depth, confidence,
            grad_mu_z, loss_accumulator, valid_pixel_count,
            N=N, H=H, W=W, HUBER_DELTA=huber_delta
        )
        
        # Reduce block-level results on host (correct scaling)
        total_loss = loss_accumulator.sum()
        total_valid_pixels = valid_pixel_count.sum().float()
        
        # Normalize by valid pixel count (do this once, correctly)
        if total_valid_pixels > 0:
            normalized_loss = total_loss / total_valid_pixels
            normalized_grad = grad_mu_z / total_valid_pixels
        else:
            normalized_loss = torch.tensor(0.0, device=mu_z.device)
            normalized_grad = torch.zeros_like(grad_mu_z)
        
        # Save for backward
        ctx.save_for_backward(normalized_grad)
        
        return normalized_loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass returns pre-computed gradients.
        """
        grad_mu_z, = ctx.saved_tensors
        
        # Scale by upstream gradient
        grad_mu_z = grad_mu_z * grad_output
        
        return grad_mu_z, None, None, None, None


# Public API functions
def apply_deformation(mu, W, b, visibility_mask=None):
    """
    Apply low-rank deformation to splat centroids.
    
    Args:
        mu: [N, 3] splat centroids
        W: [N, 3, r] deformation basis matrices
        b: [r] time-dependent coefficients
        visibility_mask: [N] optional boolean mask for visible splats
        
    Returns:
        mu_deformed: [N, 3] deformed centroids
    """
    return DeformUpdateFunction.apply(mu, W, b, visibility_mask)


def compute_depth_loss(mu_z, id_buffer, depth, confidence, huber_delta=0.01):
    """
    Compute depth reprojection loss using smooth L1 (Huber).
    
    Args:
        mu_z: [N] z-coordinates of splats
        id_buffer: [H, W] splat ID buffer from rasterizer
        depth: [H, W] ground truth depth map
        confidence: [H, W] depth confidence weights
        huber_delta: Huber loss threshold for smooth L1
        
    Returns:
        loss: scalar depth loss
    """
    return DepthResidualFunction.apply(mu_z, id_buffer, depth, confidence, huber_delta)


def compile_kernels():
    """Compile Triton kernels for faster startup."""
    print("Compiling optimized Triton deformation kernels...")
    
    # Dummy inputs for compilation
    N, r = 1000, 4
    H, W = 256, 256
    
    mu = torch.randn(N, 3, device='cuda', dtype=torch.float32)
    W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16)
    b = torch.randn(r, device='cuda', dtype=torch.float16)
    visibility_mask = torch.randint(0, 2, (N,), device='cuda', dtype=torch.bool)
    
    # Compile deformation kernel with and without visibility mask
    _ = apply_deformation(mu, W, b)
    _ = apply_deformation(mu, W, b, visibility_mask)
    
    # Compile depth kernel with different sizes for autotuning
    for h, w in [(256, 256), (512, 512), (1080, 1920)]:
        mu_z = torch.randn(N, device='cuda', dtype=torch.float32)
        id_buffer = torch.randint(-1, N, (h, w), device='cuda', dtype=torch.int32)
        depth = torch.randn(h, w, device='cuda', dtype=torch.float32)
        confidence = torch.ones(h, w, device='cuda', dtype=torch.float32)
        
        try:
            _ = compute_depth_loss(mu_z, id_buffer, depth, confidence)
        except Exception as e:
            print(f"Compilation warning for {h}x{w}: {e}")
    
    print("Kernel compilation complete.")


# Helper functions for debugging and validation
def validate_tensors(mu, W, b, visibility_mask=None):
    """Validate tensor shapes and dtypes for deformation kernels."""
    assert mu.dim() == 2 and mu.shape[1] == 3, f"mu must be [N, 3], got {mu.shape}"
    assert mu.dtype == torch.float32, f"mu must be float32, got {mu.dtype}"
    assert mu.device.type == 'cuda', f"mu must be on CUDA, got {mu.device}"
    
    N = mu.shape[0]
    r = b.shape[0]
    
    assert W.shape == (N, 3, r), f"W must be [N, 3, r], got {W.shape}"
    assert W.dtype == torch.float16, f"W must be float16, got {W.dtype}"
    assert W.device.type == 'cuda', f"W must be on CUDA, got {W.device}"
    
    assert b.dim() == 1, f"b must be 1D, got {b.dim()}D"
    assert b.dtype == torch.float16, f"b must be float16, got {b.dtype}"
    assert b.device.type == 'cuda', f"b must be on CUDA, got {b.device}"
    
    if visibility_mask is not None:
        assert visibility_mask.shape == (N,), f"visibility_mask must be [N], got {visibility_mask.shape}"
        assert visibility_mask.dtype == torch.bool, f"visibility_mask must be bool, got {visibility_mask.dtype}"
        assert visibility_mask.device.type == 'cuda', f"visibility_mask must be on CUDA, got {visibility_mask.device}"


def validate_depth_tensors(mu_z, id_buffer, depth, confidence):
    """Validate tensor shapes and dtypes for depth loss kernels."""
    N = mu_z.shape[0]
    H, W = depth.shape
    
    assert mu_z.dim() == 1, f"mu_z must be 1D, got {mu_z.dim()}D"
    assert mu_z.dtype == torch.float32, f"mu_z must be float32, got {mu_z.dtype}"
    assert mu_z.device.type == 'cuda', f"mu_z must be on CUDA, got {mu_z.device}"
    
    assert id_buffer.shape == (H, W), f"id_buffer must be [H, W], got {id_buffer.shape}"
    assert id_buffer.dtype == torch.int32, f"id_buffer must be int32, got {id_buffer.dtype}"
    assert id_buffer.device.type == 'cuda', f"id_buffer must be on CUDA, got {id_buffer.device}"
    
    assert depth.shape == (H, W), f"depth must be [H, W], got {depth.shape}"
    assert depth.dtype == torch.float32, f"depth must be float32, got {depth.dtype}"
    assert depth.device.type == 'cuda', f"depth must be on CUDA, got {depth.device}"
    
    assert confidence.shape == (H, W), f"confidence must be [H, W], got {confidence.shape}"
    assert confidence.dtype == torch.float32, f"confidence must be float32, got {confidence.dtype}"
    assert confidence.device.type == 'cuda', f"confidence must be on CUDA, got {confidence.device}"


# Performance benchmarking utilities
def benchmark_deformation_kernel(N=10000, r=4, num_runs=100, warmup=10):
    """Benchmark the deformation kernel performance."""
    mu = torch.randn(N, 3, device='cuda', dtype=torch.float32)
    W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16)
    b = torch.randn(r, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        _ = apply_deformation(mu, W, b)
    
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start_time = time.time()
    
    for _ in range(num_runs):
        result = apply_deformation(mu, W, b)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    
    print(f"Deformation kernel benchmark:")
    print(f"  N={N}, r={r}")
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {N / avg_time_ms * 1000:.0f} splats/sec")
    
    return avg_time_ms


def benchmark_depth_kernel(N=10000, H=512, W=512, num_runs=100, warmup=10):
    """Benchmark the depth loss kernel performance."""
    mu_z = torch.randn(N, device='cuda', dtype=torch.float32) + 5.0
    id_buffer = torch.randint(-1, N, (H, W), device='cuda', dtype=torch.int32)
    depth = torch.randn(H, W, device='cuda', dtype=torch.float32).abs() + 1.0
    confidence = torch.ones(H, W, device='cuda', dtype=torch.float32) * 0.8
    
    # Warmup
    for _ in range(warmup):
        _ = compute_depth_loss(mu_z, id_buffer, depth, confidence)
    
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start_time = time.time()
    
    for _ in range(num_runs):
        loss = compute_depth_loss(mu_z, id_buffer, depth, confidence)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    
    print(f"Depth loss kernel benchmark:")
    print(f"  Resolution: {H}x{W}, N={N}")
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {H * W / avg_time_ms * 1000:.0f} pixels/sec")
    
    return avg_time_ms


if __name__ == "__main__":
    """Test script when run directly."""
    print("Testing Triton deformation kernels...")
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    try:
        # Test compilation
        compile_kernels()
        
        # Basic functionality test
        N, r = 1000, 4
        mu = torch.randn(N, 3, device='cuda', dtype=torch.float32)
        W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16)
        b = torch.randn(r, device='cuda', dtype=torch.float16)
        
        result = apply_deformation(mu, W, b)
        print(f"✓ Deformation test passed: {result.shape}")
        
        # Test depth loss
        H, W = 256, 256
        mu_z = torch.randn(N, device='cuda', dtype=torch.float32) + 5.0
        id_buffer = torch.randint(-1, N, (H, W), device='cuda', dtype=torch.int32)
        depth = torch.randn(H, W, device='cuda', dtype=torch.float32).abs() + 1.0
        confidence = torch.ones(H, W, device='cuda', dtype=torch.float32) * 0.5
        
        loss = compute_depth_loss(mu_z, id_buffer, depth, confidence)
        print(f"✓ Depth loss test passed: {loss.item():.6f}")
        
        # Run benchmarks
        print("\nRunning performance benchmarks...")
        benchmark_deformation_kernel()
        benchmark_depth_kernel()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)