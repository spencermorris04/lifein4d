# kernels/triton_deform.py
#
# Clean Triton-2 rewrite (≈340 LOC)
# Author: OpenAI ChatGPT – 2025-06
#
# ─────────────────────────────────────────────────────────────────────────────
#  Deformation update  μ_out = μ + W @ b(t)   (optionally gated by visibility)
# ─────────────────────────────────────────────────────────────────────────────

import torch, triton, triton.language as tl
from torch.autograd import Function

# --------------------------------------------------------------------------- #
# Low-rank deformation kernel
# --------------------------------------------------------------------------- #
@triton.jit
def deform_update_kernel(
    mu_ptr,              # float32 [N,3]
    W_ptr,               # float16 [N,3,r]
    b_ptr,               # float16 [r]
    mu_out_ptr,          # float32 [N,3]
    vis_ptr,             # bool [N] or dummy
    N: tl.constexpr,     # number of splats
    R: tl.constexpr,     # rank (b-dim)   – must be power-of-2 (4/8/…)
    BLOCK_SIZE: tl.constexpr,       # 64/128/256
    USE_MASK: tl.constexpr,         # 0 / 1  (specialises kernel)
):
    # ---- lane identifiers -------------------------------------------------- #
    pid   = tl.program_id(axis=0)
    lane  = tl.arange(0, BLOCK_SIZE)                # [BLOCK_SIZE]
    idx   = pid * BLOCK_SIZE + lane                 # global splat idx
    active = idx < N                                # bool mask

    # ---- load time coefficients  b  --------------------------------------- #
    b = tl.load(b_ptr + tl.arange(0, R)).to(tl.float32)  # [R]  – in registers

    # ---- visibility (compile-time specialisation) -------------------------- #
    if USE_MASK:
        vis = tl.load(vis_ptr + idx, mask=active, other=0)  # bool vector
        active = active & vis                               # only visible lanes

    # ---- base pointers ----------------------------------------------------- #
    base_mu = idx * 3                          # each splat has 3 coords

    # load μ  (scalar loads avoid 3-element arange)
    mu_x = tl.load(mu_ptr + base_mu + 0, mask=active, other=0.).to(tl.float32)
    mu_y = tl.load(mu_ptr + base_mu + 1, mask=active, other=0.).to(tl.float32)
    mu_z = tl.load(mu_ptr + base_mu + 2, mask=active, other=0.).to(tl.float32)

    # delta accumulators
    d0 = tl.zeros([BLOCK_SIZE], tl.float32)   # x
    d1 = tl.zeros([BLOCK_SIZE], tl.float32)   # y
    d2 = tl.zeros([BLOCK_SIZE], tl.float32)   # z

    # ---- fused mat-vec  W @ b  -------------------------------------------- #
    # Pointer stride: splat_i · (3·R) + {0,1,2}·R + j
    w_row0_base = W_ptr + idx * 3 * R + 0 * R
    w_row1_base = W_ptr + idx * 3 * R + 1 * R
    w_row2_base = W_ptr + idx * 3 * R + 2 * R

    for j in tl.static_range(R):              # compile-time unrolled
        bj = b[j]
        w0 = tl.load(w_row0_base + j, mask=active, other=0.).to(tl.float32)
        w1 = tl.load(w_row1_base + j, mask=active, other=0.).to(tl.float32)
        w2 = tl.load(w_row2_base + j, mask=active, other=0.).to(tl.float32)
        d0 += w0 * bj
        d1 += w1 * bj
        d2 += w2 * bj

    # ---- write-back -------------------------------------------------------- #
    res_x = mu_x + d0
    res_y = mu_y + d1
    res_z = mu_z + d2

    tl.store(mu_out_ptr + base_mu + 0, res_x, mask=idx < N)
    tl.store(mu_out_ptr + base_mu + 1, res_y, mask=idx < N)
    tl.store(mu_out_ptr + base_mu + 2, res_z, mask=idx < N)


# --------------------------------------------------------------------------- #
# Depth residual kernel  (smooth L1 / Huber)
# --------------------------------------------------------------------------- #
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
    id_ptr, z_ptr, d_ptr, c_ptr,
    grad_z_ptr, loss_acc_ptr, cnt_acc_ptr,
    N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HUBER_DELTA: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(axis=0)
    lane  = tl.arange(0, BLOCK_SIZE)
    pix   = pid * BLOCK_SIZE + lane           # flat pixel idx
    valid_p = pix < H * W                     # pixels inside image

    # local accumulators
    t_loss  = tl.zeros([BLOCK_SIZE], tl.float32)
    t_count = tl.zeros([BLOCK_SIZE], tl.int32)

    # gather ids
    sid = tl.load(id_ptr + pix, mask=valid_p, other=-1).to(tl.int32)
    good = valid_p & (sid >= 0) & (sid < N)

    # depth & confidence
    d_gt  = tl.load(d_ptr + pix, mask=good, other=0.)
    conf  = tl.load(c_ptr + pix, mask=good, other=0.)
    good  = good & (conf > 0.) & (d_gt > 0.)

    # pred depth
    d_pred = tl.load(z_ptr + sid, mask=good, other=0.)

    # residual
    res   = d_pred - d_gt
    abs_r = tl.abs(res)

    quad      = abs_r <= HUBER_DELTA
    loss_val  = tl.where(quad,
                         0.5 * res * res / HUBER_DELTA,
                         abs_r - 0.5 * HUBER_DELTA)
    grad_val  = tl.where(quad,
                         res / HUBER_DELTA,
                         tl.where(res > 0, 1.0, -1.0))

    # weight by confidence & accumulate
    w_loss = loss_val * conf * good.to(tl.float32)
    w_grad = grad_val * conf * good.to(tl.float32)

    t_loss  += w_loss
    t_count += good.to(tl.int32)

    tl.atomic_add(grad_z_ptr + sid, w_grad, mask=good)

    # block-level reduction (1 store per program)
    block_loss  = tl.sum(t_loss)
    block_count = tl.sum(t_count)

    tl.store(loss_acc_ptr  + pid, block_loss)
    tl.store(cnt_acc_ptr   + pid, block_count)


# --------------------------------------------------------------------------- #
# Autograd wrappers
# --------------------------------------------------------------------------- #
class _DeformUpdate(Function):
    @staticmethod
    def forward(ctx, mu, W, b, vis=None):
        assert (b.numel() & (b.numel() - 1)) == 0, "R must be power-of-2"
        N, _  = mu.shape
        BSIZE = 128
        grid  = (triton.cdiv(N, BSIZE),)

        out = torch.empty_like(mu)
        use_mask = vis is not None
        vis_ptr  = vis if use_mask else mu  # dummy valid pointer

        deform_update_kernel[grid](
            mu, W, b, out, vis_ptr,
            N=N, R=b.numel(),
            BLOCK_SIZE=BSIZE, USE_MASK=int(use_mask),
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
                W.float() *
                (g_out * vis.unsqueeze(-1).float()).unsqueeze(-1),
                dim=(0, 1))
        return g_mu, g_W.half(), g_b.float(), None


class _DepthResidual(Function):
    @staticmethod
    def forward(ctx, z, ids, depth, conf, delta=0.01):
        H, W = depth.shape
        BS   = 256
        nb   = triton.cdiv(H * W, BS)

        loss_acc  = torch.zeros(nb, device=z.device, dtype=torch.float32)
        cnt_acc   = torch.zeros(nb, device=z.device, dtype=torch.int32)
        grad_z    = torch.zeros_like(z)

        depth_residual_kernel[(nb,)](
            ids, z, depth, conf,
            grad_z, loss_acc, cnt_acc,
            N=z.shape[0], H=H, W=W,
            HUBER_DELTA=delta, BLOCK_SIZE=BS)

        tot_loss = loss_acc.sum()
        tot_cnt  = cnt_acc.sum().float()
        if tot_cnt > 0:
            loss = tot_loss / tot_cnt
            grad = grad_z   / tot_cnt
        else:
            loss = torch.tensor(0., device=z.device)
            grad = torch.zeros_like(z)

        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def backward(ctx, g_out):
        (grad,) = ctx.saved_tensors
        return grad * g_out, None, None, None, None


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def apply_deformation(mu, W, b, visibility_mask=None):
    return _DeformUpdate.apply(mu, W, b, visibility_mask)

def compute_depth_loss(z, id_buf, depth, conf, huber_delta=0.01):
    return _DepthResidual.apply(z, id_buf, depth, conf, huber_delta)


# --------------------------------------------------------------------------- #
# Ahead-of-time compilation helper  (≈1 sec on modern GPUs)
# --------------------------------------------------------------------------- #
def compile_kernels():
    print("Compiling Triton kernels …")

    # dummy tensors
    N, R = 2048, 4
    mu   = torch.randn(N, 3, device='cuda', dtype=torch.float32)
    Wmat = torch.randn(N, 3, R, device='cuda', dtype=torch.float16)
    bvec = torch.randn(R,     device='cuda', dtype=torch.float16)
    mask = torch.randint(0, 2, (N,), device='cuda', dtype=torch.bool)

    apply_deformation(mu, Wmat, bvec)          # no mask
    apply_deformation(mu, Wmat, bvec, mask)    # with mask

    for H, W in [(256,256), (512,512), (1080,1920)]:
        z     = torch.randn(N, device='cuda')
        ids   = torch.randint(-1, N, (H,W), device='cuda', dtype=torch.int32)
        depth = torch.randn(H, W, device='cuda')
        conf  = torch.ones_like(depth)
        compute_depth_loss(z, ids, depth, conf)

    print("Kernel compilation complete.")
