# kernels/triton_deform.py
import torch
import triton
import triton.language as tl
from torch.autograd import Function

# --------------------------------------------------------------------------- #
# Deformation kernel  μ_out = μ + W @ b(t)
# --------------------------------------------------------------------------- #
@triton.jit
def deform_update_kernel(
    mu_ptr,                # [N, 3]  fp32
    W_ptr,                 # [N, 3, r] fp16
    b_ptr,                 # [r]      fp16
    mu_out_ptr,            # [N, 3]  fp32
    visibility_mask_ptr,   # [N] bool    (ignored if USE_MASK = False)
    N: tl.constexpr,
    r: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,              ### NEW
):
    """Fused centroid update without break/continue; optional visibility mask."""
    pid         = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Put time-coefficients in registers
    b = tl.load(b_ptr + tl.arange(0, r)).to(tl.float32)

    for offset in tl.static_range(BLOCK_SIZE):  # unrolled, compile-time
        splat_idx = block_start + offset
        lane_is_valid = splat_idx < N

        if lane_is_valid:
            if USE_MASK:                         ### NEW
                is_visible = tl.load(visibility_mask_ptr + splat_idx)
            else:                                ### NEW
                is_visible = True

            # base μ
            mu = tl.load(mu_ptr + splat_idx * 3 + tl.arange(0, 3)).to(tl.float32)

            delta = tl.zeros([3], dtype=tl.float32)
            if is_visible:
                W_base = W_ptr + splat_idx * 3 * r
                for j in tl.static_range(r):
                    W_col = tl.load(W_base + tl.arange(0, 3) * r + j).to(tl.float32)
                    delta += W_col * b[j]

            tl.store(
                mu_out_ptr + splat_idx * 3 + tl.arange(0, 3),
                mu + delta,
            )


# --------------------------------------------------------------------------- #
# Depth residual kernel (smooth-L1 with confidence) — no break/continue
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["H", "W"],
)
@triton.jit
def depth_residual_kernel(
    id_buffer_ptr,       # [H, W] int32  (splat id, −1 = bg)
    mu_z_ptr,            # [N] fp32
    depth_ptr,           # [H, W] fp32
    conf_ptr,            # [H, W] fp32
    grad_mu_z_ptr,       # [N] fp32   (atomic add)
    loss_acc_ptr,        # [num_blocks] fp32
    valid_cnt_ptr,       # [num_blocks] int32
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HUBER_DELTA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid         = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    thread_loss  = tl.float32(0.0)
    thread_valid = 0

    for i in tl.static_range(BLOCK_SIZE):
        pix = block_start + i
        if pix < H * W:
            sid   = tl.load(id_buffer_ptr + pix).to(tl.int32)
            good  = (sid >= 0) & (sid < N)

            if good:
                d_gt = tl.load(depth_ptr + pix)
                c    = tl.load(conf_ptr + pix)
                good = (c > 0.0) & (d_gt > 0.0)

                if good:
                    d_pred = tl.load(mu_z_ptr + sid)
                    res    = d_pred - d_gt
                    abs_r  = tl.abs(res)

                    quad   = abs_r <= HUBER_DELTA
                    loss   = tl.where(
                        quad,
                        0.5 * res * res / HUBER_DELTA,
                        abs_r - 0.5 * HUBER_DELTA,
                    )
                    grad   = tl.where(
                        quad,
                        res / HUBER_DELTA,
                        tl.where(res > 0, 1.0, -1.0),
                    )

                    thread_loss  += c * loss
                    thread_valid += 1
                    tl.atomic_add(grad_mu_z_ptr + sid, c * grad)

    tl.store(loss_acc_ptr  + pid, thread_loss)
    tl.store(valid_cnt_ptr + pid, thread_valid)


# --------------------------------------------------------------------------- #
# Autograd wrappers
# --------------------------------------------------------------------------- #
class DeformUpdateFunction(Function):
    @staticmethod
    def forward(ctx, mu, W, b, visibility_mask=None):
        N, _ = mu.shape
        r    = b.numel()
        out  = torch.empty_like(mu)

        BLOCK = 256
        grid  = (triton.cdiv(N, BLOCK),)

        use_mask = visibility_mask is not None              ### NEW
        vis_ptr  = visibility_mask if use_mask else mu      ### dummy ptr (never read)

        deform_update_kernel[grid](
            mu, W, b, out, vis_ptr,
            N=N, r=r, BLOCK_SIZE=BLOCK, USE_MASK=use_mask, ### NEW
        )

        ctx.save_for_backward(W, b, visibility_mask)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        W, b, mask = ctx.saved_tensors
        grad_mu = grad_out if mask is None else grad_out * mask.unsqueeze(-1).float()
        grad_W  = grad_out.unsqueeze(-1) * b.float()
        if mask is None:
            grad_b = torch.sum(W.float() * grad_out.unsqueeze(-1), dim=(0, 1))
        else:
            grad_b = torch.sum(
                W.float() * (grad_out * mask.unsqueeze(-1).float()).unsqueeze(-1),
                dim=(0, 1),
            )
        return grad_mu, grad_W.half(), grad_b.float(), None


class DepthResidualFunction(Function):
    @staticmethod
    def forward(ctx, mu_z, id_buf, depth, conf, huber_delta=0.01):
        H, W = depth.shape
        N    = mu_z.shape[0]

        BLOCK = 256
        nb = triton.cdiv(H * W, BLOCK)

        loss_acc  = torch.zeros(nb, device=mu_z.device, dtype=torch.float32)
        valid_cnt = torch.zeros(nb, device=mu_z.device, dtype=torch.int32)
        grad_mu_z = torch.zeros_like(mu_z)

        depth_residual_kernel[(nb,)](
            id_buf, mu_z, depth, conf,
            grad_mu_z, loss_acc, valid_cnt,
            N=N, H=H, W=W,
            HUBER_DELTA=huber_delta,
            BLOCK_SIZE=BLOCK,
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
    def backward(ctx, grad_out):
        (grad_mu_z,) = ctx.saved_tensors
        return grad_mu_z * grad_out, None, None, None, None


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def apply_deformation(mu, W, b, visibility_mask=None):
    return DeformUpdateFunction.apply(mu, W, b, visibility_mask)


def compute_depth_loss(mu_z, id_buffer, depth, confidence, huber_delta=0.01):
    return DepthResidualFunction.apply(
        mu_z, id_buffer, depth, confidence, huber_delta
    )


# --------------------------------------------------------------------------- #
# Ahead-of-time compilation helper
# --------------------------------------------------------------------------- #
def compile_kernels():
    """Compile kernels once so the first real call is instant."""
    print("Compiling Triton kernels …")

    N, r = 1024, 4
    mu   = torch.randn(N, 3, device="cuda", dtype=torch.float32)
    W    = torch.randn(N, 3, r, device="cuda", dtype=torch.float16)
    b    = torch.randn(r,     device="cuda", dtype=torch.float16)
    vis  = torch.randint(0, 2, (N,), device="cuda", dtype=torch.bool)

    apply_deformation(mu, W, b)          # no mask
    apply_deformation(mu, W, b, vis)     # with mask

    for H, Wimg in [(256, 256), (512, 512), (1080, 1920)]:
        mu_z  = torch.randn(N, device="cuda")
        idbuf = torch.randint(-1, N, (H, Wimg), device="cuda", dtype=torch.int32)
        depth = torch.randn(H, Wimg, device="cuda")
        conf  = torch.ones_like(depth)
        compute_depth_loss(mu_z, idbuf, depth, conf)

    print("Kernel compilation complete.")
