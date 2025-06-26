# kernels/triton_deform.py
import torch
import triton
import triton.language as tl
from torch.autograd import Function


# ----------------------------------------------------------------------------- #
# Deformation kernel  (μ_out = μ + W @ b(t))
# ----------------------------------------------------------------------------- #
@triton.jit
def deform_update_kernel(
    mu_ptr,                # [N, 3]  fp32
    W_ptr,                 # [N, 3, r]  fp16
    b_ptr,                 # [r]  fp16
    mu_out_ptr,            # [N, 3]  fp32
    visibility_mask_ptr,   # [N]  bool  (0 ⇒ no mask)
    N: tl.constexpr,
    r: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused μ_out = μ + W @ b with optional visibility masking."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load time coefficients once per program – stays in registers.
    b = tl.load(b_ptr + tl.arange(0, r)).to(tl.float32)     # [r]

    # Iterate over a fixed-size tile; mask out lanes that run past N.
    for offset in tl.static_range(BLOCK_SIZE):
        splat_idx = block_start + offset
        valid_lane = splat_idx < N
        if not valid_lane:
            continue

        # Visibility masking (if pointer ≠ 0)
        if visibility_mask_ptr != 0:
            is_visible = tl.load(visibility_mask_ptr + splat_idx)
            if not is_visible:
                # copy original μ straight through
                mu_val = tl.load(
                    mu_ptr + splat_idx * 3 + tl.arange(0, 3)
                )                                      # [3]
                tl.store(
                    mu_out_ptr + splat_idx * 3 + tl.arange(0, 3),
                    mu_val,
                )
                continue

        # ---- load μ ---------------------------------------------------------- #
        mu = tl.load(
            mu_ptr + splat_idx * 3 + tl.arange(0, 3)
        ).to(tl.float32)                                 # [3]

        # ---- compute Δ = W @ b ------------------------------------------------ #
        W_base = W_ptr + splat_idx * 3 * r
        delta = tl.zeros([3], dtype=tl.float32)

        # Unroll over “r” (compile-time constant)
        for j in tl.static_range(r):
            W_col = tl.load(W_base + tl.arange(0, 3) * r + j).to(tl.float32)
            delta += W_col * b[j]                        # fused-mul-add

        # ---- write result ---------------------------------------------------- #
        tl.store(
            mu_out_ptr + splat_idx * 3 + tl.arange(0, 3),
            mu + delta,
        )


# ----------------------------------------------------------------------------- #
# Depth residual kernel  (smooth-L1 + confidence weights)
# ----------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['H', 'W'],
)
@triton.jit
def depth_residual_kernel(
    id_buffer_ptr,          # [H, W]  int32 (splat IDs, -1 = bg)
    mu_z_ptr,               # [N]     fp32  (z of splats)
    depth_ptr,              # [H, W]  fp32  (GT depth)
    confidence_ptr,         # [H, W]  fp32  (weights)
    grad_mu_z_ptr,          # [N]     fp32  (∂L/∂μ_z)  – atomic add
    loss_acc_ptr,           # [num_blocks] fp32
    valid_cnt_ptr,          # [num_blocks] int32
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HUBER_DELTA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-pixel smooth-L1 residual against depth map."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    thread_loss = tl.float32(0.0)
    thread_valid = 0

    for i in tl.static_range(BLOCK_SIZE):
        pixel_idx = block_start + i
        if pixel_idx >= H * W:
            continue

        splat_id = tl.load(id_buffer_ptr + pixel_idx).to(tl.int32)
        if (splat_id < 0) or (splat_id >= N):
            continue

        gt_depth   = tl.load(depth_ptr + pixel_idx)
        confidence = tl.load(confidence_ptr + pixel_idx)
        if (confidence <= 0.0) or (gt_depth <= 0.0):
            continue

        pred_depth = tl.load(mu_z_ptr + splat_id)
        residual   = pred_depth - gt_depth
        abs_res    = tl.abs(residual)

        # smooth-L1 (Huber) loss
        is_quad   = abs_res <= HUBER_DELTA
        loss_val  = tl.where(
            is_quad,
            0.5 * residual * residual / HUBER_DELTA,
            abs_res - 0.5 * HUBER_DELTA,
        )
        grad_val = tl.where(
            is_quad,
            residual / HUBER_DELTA,
            tl.where(residual > 0, 1.0, -1.0),
        )

        weighted_loss = confidence * loss_val
        weighted_grad = confidence * grad_val

        thread_loss  += weighted_loss
        thread_valid += 1
        tl.atomic_add(grad_mu_z_ptr + splat_id, weighted_grad)

    # block-level accumulation (one atomic write per block)
    tl.store(loss_acc_ptr  + pid, thread_loss)
    tl.store(valid_cnt_ptr + pid, thread_valid)


# ----------------------------------------------------------------------------- #
# PyTorch autograd wrappers
# ----------------------------------------------------------------------------- #
class DeformUpdateFunction(Function):
    @staticmethod
    def forward(ctx, mu, W, b, visibility_mask=None):
        N, _  = mu.shape
        r     = b.shape[0]
        out   = torch.empty_like(mu)

        BLOCK = 256
        grid  = (triton.cdiv(N, BLOCK),)

        vis_ptr = visibility_mask if visibility_mask is not None else 0
        deform_update_kernel[grid](
            mu, W, b, out, vis_ptr,
            N=N, r=r, BLOCK_SIZE=BLOCK,
        )

        ctx.save_for_backward(W, b, visibility_mask)
        ctx.r = r
        return out

    @staticmethod
    def backward(ctx, grad_out):
        W, b, mask = ctx.saved_tensors
        grad_mu = grad_out
        if mask is not None:
            grad_mu = grad_mu * mask.unsqueeze(-1).float()

        grad_W = grad_out.unsqueeze(-1) * b.float()          # [N,3,r]
        if mask is not None:
            grad_b = torch.sum(W.float() * (grad_out * mask.unsqueeze(-1).float()).unsqueeze(-1), dim=(0, 1))
        else:
            grad_b = torch.sum(W.float() * grad_out.unsqueeze(-1), dim=(0, 1))

        return grad_mu, grad_W.half(), grad_b.float(), None


class DepthResidualFunction(Function):
    @staticmethod
    def forward(ctx, mu_z, id_buf, depth, conf, huber_delta=0.01):
        H, W = depth.shape
        N    = mu_z.shape[0]

        BLOCK = 256
        num_blocks = triton.cdiv(H * W, BLOCK)

        loss_acc = torch.zeros(num_blocks, device=mu_z.device, dtype=torch.float32)
        valid_cnt = torch.zeros(num_blocks, device=mu_z.device, dtype=torch.int32)
        grad_mu_z = torch.zeros_like(mu_z)

        grid = (num_blocks,)
        depth_residual_kernel[grid](
            id_buf, mu_z, depth, conf,
            grad_mu_z, loss_acc, valid_cnt,
            N=N, H=H, W=W, HUBER_DELTA=huber_delta, BLOCK_SIZE=BLOCK,
        )

        tot_loss   = loss_acc.sum()
        tot_pixels = valid_cnt.sum().float()
        if tot_pixels > 0:
            loss = tot_loss / tot_pixels
            grad = grad_mu_z / tot_pixels
        else:
            loss = torch.tensor(0.0, device=mu_z.device)
            grad = torch.zeros_like(mu_z)

        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_mu_z,) = ctx.saved_tensors
        return grad_mu_z * grad_output, None, None, None, None


# ----------------------------------------------------------------------------- #
# Convenience API
# ----------------------------------------------------------------------------- #
def apply_deformation(mu, W, b, visibility_mask=None):
    return DeformUpdateFunction.apply(mu, W, b, visibility_mask)


def compute_depth_loss(mu_z, id_buffer, depth, confidence, huber_delta=0.01):
    return DepthResidualFunction.apply(mu_z, id_buffer, depth, confidence, huber_delta)


def compile_kernels():
    """Dry-run compile to amortise JIT overhead."""
    print("Compiling Triton kernels…")

    N, r = 1000, 4
    mu   = torch.randn(N, 3, device='cuda', dtype=torch.float32)
    W    = torch.randn(N, 3, r, device='cuda', dtype=torch.float16)
    b    = torch.randn(r,     device='cuda', dtype=torch.float16)
    vis  = torch.randint(0, 2, (N,), device='cuda', dtype=torch.bool)

    _ = apply_deformation(mu, W, b)
    _ = apply_deformation(mu, W, b, vis)

    for H, Wimg in [(256, 256), (512, 512), (1080, 1920)]:
        mu_z  = torch.randn(N, device='cuda')
        idbuf = torch.randint(-1, N, (H, Wimg), device='cuda', dtype=torch.int32)
        depth = torch.randn(H, Wimg, device='cuda')
        conf  = torch.ones_like(depth)
        _ = compute_depth_loss(mu_z, idbuf, depth, conf)

    print("Kernel compilation complete.")
