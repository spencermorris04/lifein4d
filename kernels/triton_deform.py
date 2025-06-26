# kernels/triton_deform.py
import torch, triton, triton.language as tl
from torch.autograd import Function

# ──────────────────────────────────────────────────────────────────────────
# Deformation kernel  μ_out = μ + W @ b(t)     (optional visibility mask)
# ──────────────────────────────────────────────────────────────────────────
@triton.jit
def deform_update_kernel(
    mu_ptr, W_ptr, b_ptr, mu_out_ptr, visibility_mask_ptr,
    N: tl.constexpr, r: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, USE_MASK: tl.constexpr,
):
    pid         = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # time-coefficients live in registers
    b = tl.load(b_ptr + tl.arange(0, r)).to(tl.float32)

    for off in tl.static_range(BLOCK_SIZE):            # compile-time unrolled
        idx           = block_start + off
        lane_is_valid = idx < N

        if lane_is_valid:
            visible = True
            if USE_MASK:
                visible = tl.load(visibility_mask_ptr + idx)

            base_mu = idx * 3

            # load μ = [x,y,z]
            mu_x = tl.load(mu_ptr + base_mu + 0).to(tl.float32)
            mu_y = tl.load(mu_ptr + base_mu + 1).to(tl.float32)
            mu_z = tl.load(mu_ptr + base_mu + 2).to(tl.float32)

            d0 = tl.float32(0.0)  # delta_x
            d1 = tl.float32(0.0)  # delta_y
            d2 = tl.float32(0.0)  # delta_z

            if visible:
                W_base = W_ptr + idx * 3 * r
                # accumulate W[:, j] * b[j]
                for j in tl.static_range(r):
                    w0 = tl.load(W_base + 0 * r + j).to(tl.float32)
                    w1 = tl.load(W_base + 1 * r + j).to(tl.float32)
                    w2 = tl.load(W_base + 2 * r + j).to(tl.float32)
                    d0 += w0 * b[j]
                    d1 += w1 * b[j]
                    d2 += w2 * b[j]

            tl.store(mu_out_ptr + base_mu + 0, mu_x + d0)
            tl.store(mu_out_ptr + base_mu + 1, mu_y + d1)
            tl.store(mu_out_ptr + base_mu + 2, mu_z + d2)


# ──────────────────────────────────────────────────────────────────────────
# Depth residual kernel (unchanged except for no tl.arange)
# ──────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ], key=['H', 'W'],
)
@triton.jit
def depth_residual_kernel(
    id_buf_ptr, mu_z_ptr, depth_ptr, conf_ptr,
    grad_mu_z_ptr, loss_acc_ptr, valid_cnt_ptr,
    N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HUBER_DELTA: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid         = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    t_loss  = tl.float32(0.0)
    t_valid = 0

    for i in tl.static_range(BLOCK_SIZE):
        pix = block_start + i
        if pix < H * W:
            sid  = tl.load(id_buf_ptr + pix).to(tl.int32)
            good = (sid >= 0) & (sid < N)

            if good:
                d_gt = tl.load(depth_ptr + pix)
                c    = tl.load(conf_ptr + pix)
                good = (c > 0.0) & (d_gt > 0.0)

                if good:
                    d_pred = tl.load(mu_z_ptr + sid)
                    res    = d_pred - d_gt
                    abs_r  = tl.abs(res)

                    quad   = abs_r <= HUBER_DELTA
                    loss   = tl.where(quad,
                                      0.5 * res * res / HUBER_DELTA,
                                      abs_r - 0.5 * HUBER_DELTA)
                    grad   = tl.where(quad,
                                      res / HUBER_DELTA,
                                      tl.where(res > 0, 1.0, -1.0))

                    t_loss  += c * loss
                    t_valid += 1
                    tl.atomic_add(grad_mu_z_ptr + sid, c * grad)

    tl.store(loss_acc_ptr  + pid, t_loss)
    tl.store(valid_cnt_ptr + pid, t_valid)


# ──────────────────────────────────────────────────────────────────────────
# Autograd wrappers
# ──────────────────────────────────────────────────────────────────────────
class DeformUpdateFunction(Function):
    @staticmethod
    def forward(ctx, mu, W, b, vis=None):
        N, _ = mu.shape
        BLOCK = 256
        grid  = (triton.cdiv(N, BLOCK),)

        out = torch.empty_like(mu)
        use_mask = vis is not None
        vis_ptr  = vis if use_mask else mu  # dummy ptr, never deref if !use_mask

        deform_update_kernel[grid](
            mu, W, b, out, vis_ptr,
            N=N, r=b.numel(), BLOCK_SIZE=BLOCK, USE_MASK=use_mask,
        )
        ctx.save_for_backward(W, b, vis)
        return out

    @staticmethod
    def backward(ctx, g_out):
        W, b, vis = ctx.saved_tensors
        g_mu = g_out if vis is None else g_out * vis.unsqueeze(-1).float()
        g_W  = g_out.unsqueeze(-1) * b.float()
        if vis is None:
            g_b = torch.sum(W.float() * g_out.unsqueeze(-1), dim=(0, 1))
        else:
            g_b = torch.sum(
                W.float() * (g_out * vis.unsqueeze(-1).float()).unsqueeze(-1),
                dim=(0, 1),
            )
        return g_mu, g_W.half(), g_b.float(), None


class DepthResidualFunction(Function):
    @staticmethod
    def forward(ctx, mu_z, id_buf, depth, conf, huber_delta=0.01):
        H, W = depth.shape
        BLOCK = 256
        nb = triton.cdiv(H * W, BLOCK)

        loss_acc  = torch.zeros(nb, device=mu_z.device, dtype=torch.float32)
        valid_cnt = torch.zeros(nb, device=mu_z.device, dtype=torch.int32)
        grad_mu_z = torch.zeros_like(mu_z)

        depth_residual_kernel[(nb,)](
            id_buf, mu_z, depth, conf,
            grad_mu_z, loss_acc, valid_cnt,
            N=mu_z.shape[0], H=H, W=W,
            HUBER_DELTA=huber_delta, BLOCK_SIZE=BLOCK,
        )

        tot_loss   = loss_acc.sum()
        tot_pix    = valid_cnt.sum().float()
        if tot_pix > 0:
            loss = tot_loss / tot_pix
            grad = grad_mu_z / tot_pix
        else:
            loss = torch.tensor(0.0, device=mu_z.device)
            grad = torch.zeros_like(mu_z)

        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def backward(ctx, g_out):
        (grad,) = ctx.saved_tensors
        return grad * g_out, None, None, None, None


# ──────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────────────────────────────────
def apply_deformation(mu, W, b, visibility_mask=None):
    return DeformUpdateFunction.apply(mu, W, b, visibility_mask)

def compute_depth_loss(mu_z, id_buffer, depth, confidence, huber_delta=0.01):
    return DepthResidualFunction.apply(mu_z, id_buffer, depth,
                                       confidence, huber_delta)


# ──────────────────────────────────────────────────────────────────────────
# Ahead-of-time compilation helper
# ──────────────────────────────────────────────────────────────────────────
def compile_kernels():
    print("Compiling Triton kernels …")
    N, r = 1024, 4
    mu   = torch.randn(N, 3, device='cuda', dtype=torch.float32)
    W    = torch.randn(N, 3, r, device='cuda', dtype=torch.float16)
    b    = torch.randn(r,     device='cuda', dtype=torch.float16)
    vis  = torch.randint(0, 2, (N,), device='cuda', dtype=torch.bool)

    apply_deformation(mu, W, b)
    apply_deformation(mu, W, b, vis)

    for H, Wimg in [(256, 256), (512, 512), (1080, 1920)]:
        mu_z  = torch.randn(N, device='cuda')
        ids   = torch.randint(-1, N, (H, Wimg), device='cuda', dtype=torch.int32)
        depth = torch.randn(H, Wimg, device='cuda')
        conf  = torch.ones_like(depth)
        compute_depth_loss(mu_z, ids, depth, conf)

    print("Kernel compilation complete.")
