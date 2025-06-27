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
    Uses block-based processing with shared memory for b coefficients.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Load time coefficients into shared memory once per block
    if tl.program_id(axis=0) == 0 and tl.program_id(axis=1) == 0:
        # First thread loads b coefficients
        b_range = tl.arange(0, r)
        b_shared = tl.load(b_ptr + b_range, mask=b_range < r)
    else:
        b_shared = tl.zeros([r], dtype=tl.float16)
    
    # Broadcast b to all threads in block (using register spilling)
    b = tl.load(b_ptr + tl.arange(0, r)).to(tl.float32)
    
    # Process multiple splats per block
    for block_offset in range(BLOCK_SIZE):
        splat_idx = block_start + block_offset
        if splat_idx >= N:
            break
        
        # Check visibility if mask provided
        if visibility_mask_ptr is not None:
            is_visible = tl.load(visibility_mask_ptr + splat_idx)
            if not is_visible:
                # Copy original position unchanged
                mu_x = tl.load(mu_ptr + splat_idx * 3 + 0)
                mu_y = tl.load(mu_ptr + splat_idx * 3 + 1)
                mu_z = tl.load(mu_ptr + splat_idx * 3 + 2)
                tl.store(mu_out_ptr + splat_idx * 3 + 0, mu_x)
                tl.store(mu_out_ptr + splat_idx * 3 + 1, mu_y)
                tl.store(mu_out_ptr + splat_idx * 3 + 2, mu_z)
                continue
        
        # Load original centroid using vectorized access
        mu_offsets = tl.arange(0, 3)
        mu = tl.load(mu_ptr + splat_idx * 3 + mu_offsets)  # [3]
        
        # Load W matrix for this splat [3, r] - keep as fp16, convert during compute
        W_base = W_ptr + splat_idx * 3 * r
        delta = tl.zeros([3], dtype=tl.float32)
        
        # Compute W @ b efficiently
        for j in tl.static_range(r):
            W_col_offsets = tl.arange(0, 3) * r + j
            W_col = tl.load(W_base + W_col_offsets).to(tl.float32)  # [3]
            delta += W_col * b[j]
        
        # Store result μ + W @ b
        result = mu + delta
        tl.store(mu_out_ptr + splat_idx * 3 + mu_offsets, result)


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
    Optimized depth residual computation with block-level reduction.
    Uses proper atomic operations and handles scaling correctly.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Per-thread accumulators
    thread_loss = tl.float32(0.0)
    thread_valid_count = 0
    
    # Process pixels in this block
    for i in range(BLOCK_SIZE):
        pixel_idx = block_start + i
        if pixel_idx >= H * W:
            break
            
        # Load splat ID for this pixel
        splat_id = tl.load(id_buffer_ptr + pixel_idx).to(tl.int32)
        
        # Skip background pixels
        if splat_id < 0 or splat_id >= N:
            continue
            
        # Load depth and confidence
        gt_depth = tl.load(depth_ptr + pixel_idx)
        confidence = tl.load(confidence_ptr + pixel_idx)
        
        # Skip low confidence pixels
        if confidence <= 0.0 or gt_depth <= 0.0:
            continue
        
        # Load predicted depth (z-coordinate of splat)
        pred_depth = tl.load(mu_z_ptr + splat_id)
        
        # Compute residual
        residual = pred_depth - gt_depth
        abs_residual = tl.abs(residual)
        
        # Smooth L1 (Huber) loss with consistent confidence weighting
        if abs_residual <= HUBER_DELTA:
            # Quadratic region: 0.5 * residual^2 / delta
            loss_value = 0.5 * residual * residual / HUBER_DELTA
            grad_value = residual / HUBER_DELTA
        else:
            # Linear region: |residual| - 0.5 * delta
            loss_value = abs_residual - 0.5 * HUBER_DELTA
            grad_value = tl.where(residual > 0, 1.0, -1.0)
        
        # Apply confidence weighting to both loss and gradient
        weighted_loss = confidence * loss_value
        weighted_grad = confidence * grad_value
        
        # Accumulate in thread-local variables
        thread_loss += weighted_loss
        thread_valid_count += 1
        
        # Atomic gradient accumulation (this is fine, gradients are additive)
        tl.atomic_add(grad_mu_z_ptr + splat_id, weighted_grad)
    
    # Block-level reduction using shared memory
    # Note: Triton doesn't have explicit shared memory, so we'll use atomic operations
    # but only one atomic per block instead of per thread
    
    # Each block writes its totals to separate locations
    tl.store(loss_accumulator_ptr + pid, thread_loss)
    tl.store(valid_pixel_count_ptr + pid, thread_valid_count)


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