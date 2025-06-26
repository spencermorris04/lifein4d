# gaussian_renderer/__init__.py

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time


def render(
    viewpoint_camera, 
    pc: GaussianModel, 
    pipe, 
    bg_color: torch.Tensor, 
    scaling_modifier=1.0, 
    override_color=None, 
    stage="fine", 
    cam_type=None,
    apply_deform=True,
    time_coefficients=None,
    rigid_transforms=None,
):
    """
    Render the scene with optional monocular deformation.
    
    Background tensor (bg_color) must be on GPU!
    
    Args:
        viewpoint_camera: Camera parameters
        pc: GaussianModel instance
        pipe: Pipeline configuration
        bg_color: Background color tensor
        scaling_modifier: Scaling factor for Gaussians
        override_color: Override colors if provided
        stage: Rendering stage ("coarse" or "fine")
        cam_type: Camera type for special handling
        apply_deform: Whether to apply mono4dgs deformation
        time_coefficients: Time-dependent deformation coefficients [r]
        rigid_transforms: Rigid transformation parameters
        
    Returns:
        Dictionary with rendered image, viewspace points, visibility filter, radii, depth
    """
 
    # Create zero tensor for gradients of 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz
    
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    else:
        raster_settings = viewpoint_camera['camera']
        time = torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0], 1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    deformation_point = pc._deformation_table
    
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        # Apply standard 4DGS deformation if not using mono4dgs
        if not pc.enable_mono4dgs or not apply_deform:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
                means3D, scales, rotations, opacity, shs, time
            )
        else:
            # Apply mono4dgs deformation pipeline
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = apply_mono4dgs_deformation(
                pc, means3D, scales, rotations, opacity, shs, time_coefficients, rigid_transforms
            )
    else:
        raise NotImplementedError

    # Apply activations
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    # Handle color computation
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs_final,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "means3D_final": means3D_final,  # For depth loss computation
    }


def apply_mono4dgs_deformation(
    pc: GaussianModel,
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity: torch.Tensor,
    shs: torch.Tensor,
    time_coefficients: torch.Tensor,
    rigid_transforms=None,
) -> tuple:
    """
    Apply mono4dgs deformation pipeline.
    
    Args:
        pc: GaussianModel with mono4dgs components
        means3D: Original splat positions [N, 3]
        scales: Original scales [N, 3]
        rotations: Original rotations [N, 4]
        opacity: Original opacity [N, 1]
        shs: Original SH features [N, C, (max_sh_degree+1)^2]
        time_coefficients: Time-dependent coefficients [r]
        rigid_transforms: Optional rigid transformation parameters
        
    Returns:
        Tuple of (deformed_means3D, scales, rotations, opacity, shs)
    """
    deformed_means3D = means3D
    
    # Step 1: Apply low-rank deformation W @ b(t)
    if time_coefficients is not None and pc._deformation_manager is not None:
        # Use Triton kernel for efficient deformation
        deformed_means3D = pc.apply_mono_deformation(time_coefficients)
    
    # Step 2: Apply rigid cluster transformations EARLY
    # This ensures the rasterizer receives the transformed positions
    # and gradients flow through the rigid transform parameters
    if rigid_transforms is not None and pc._cluster_ids is not None:
        deformed_means3D = rigid_transforms.transform_points(
            deformed_means3D, pc._cluster_ids
        )
    
    # NOTE: We return the deformed positions early so that:
    # 1. The rasterizer uses the transformed centroids
    # 2. splat_depths slicing uses the same tensor (maintains gradient flow)
    # 3. No .detach() calls break the gradient computation graph
    
    return deformed_means3D, scales, rotations, opacity, shs


def extract_id_buffer(rasterizer_output, H: int, W: int) -> torch.Tensor:
    """
    Extract splat ID buffer from rasterizer output.
    
    NOTE: This requires modification to the C++/CUDA rasterizer to output
    per-pixel splat indices. The modifications needed are:
    
    1. In diff_gaussian_rasterization/cuda_rasterizer/rasterizer_impl.cu:
       - Add uint32_t* id_buffer parameter to CudaRasterizer::Rasterizer::forward
       - In the pixel shader loop, write splat_id to id_buffer[pixf]
       - Initialize background pixels to UINT32_MAX
    
    2. In diff_gaussian_rasterization/diff_gaussian_rasterization/__init__.py:
       - Add id_buffer to GaussianRasterizationFunction.forward
       - Return id_buffer alongside rendered image, radii, depth
    
    3. In the Python binding (rasterize_gaussians):
       - Allocate id_buffer tensor: torch.full((H, W), 2**32-1, dtype=torch.uint32)
       - Pass to CUDA kernel
       - Convert to int32 with background = -1
    
    Args:
        rasterizer_output: Output from the Gaussian rasterizer
        H: Image height
        W: Image width
        
    Returns:
        id_buffer: Splat ID buffer [H, W] with int32 values (-1 for background)
    """
    # This is the real implementation once the rasterizer is modified:
    if 'id_buffer' in rasterizer_output:
        # Convert from uint32 (background=UINT32_MAX) to int32 (background=-1)
        id_buffer_uint32 = rasterizer_output['id_buffer']
        id_buffer = torch.where(
            id_buffer_uint32 == (2**32 - 1),
            torch.tensor(-1, dtype=torch.int32, device=id_buffer_uint32.device),
            id_buffer_uint32.to(torch.int32)
        )
        return id_buffer
    else:
        # Fallback: create approximate ID buffer using depth-based assignment
        # This is less accurate but provides some functionality
        print("Warning: Using approximate ID buffer. Modify rasterizer for best results.")
        
        device = rasterizer_output["render"].device
        depth = rasterizer_output.get("depth")
        
        if depth is not None:
            # Create pseudo-ID buffer based on depth quantization
            # This is a rough approximation - real implementation needs rasterizer changes
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            pseudo_ids = (depth_normalized * 1000).long().clamp(0, 999)
            
            # Set background pixels (very far or very close) to -1
            background_mask = (depth > 100) | (depth < 0.1)
            pseudo_ids[background_mask] = -1
            
            return pseudo_ids.to(torch.int32)
        else:
            # Last resort: return dummy buffer
            return torch.full((H, W), -1, device=device, dtype=torch.int32)


def render_with_depth_loss(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    time_coefficients: torch.Tensor = None,
    rigid_transforms=None,
    **render_kwargs
) -> dict:
    """
    Render with depth loss computation.
    
    This function combines rendering with depth loss computation for efficiency.
    
    Args:
        viewpoint_camera: Camera parameters
        pc: GaussianModel instance
        pipe: Pipeline configuration
        bg_color: Background color tensor
        gt_depth: Ground truth depth map [H, W]
        depth_confidence: Depth confidence map [H, W]
        time_coefficients: Time-dependent deformation coefficients [r]
        rigid_transforms: Rigid transformation parameters
        **render_kwargs: Additional rendering arguments
        
    Returns:
        Dictionary with render results and depth loss information
    """
    # Perform standard rendering
    render_output = render(
        viewpoint_camera, pc, pipe, bg_color,
        time_coefficients=time_coefficients,
        rigid_transforms=rigid_transforms,
        **render_kwargs
    )
    
    # Extract rendered depth and splat positions
    rendered_depth = render_output["depth"]
    means3D_final = render_output["means3D_final"]
    
    # Get image dimensions
    H, W = gt_depth.shape
    
    # Extract ID buffer (this needs to be implemented in the rasterizer)
    id_buffer = extract_id_buffer(render_output, H, W)
    
    # Compute depth loss using Triton kernel
    from kernels.triton_deform import compute_depth_loss
    
    # Extract z-coordinates of final splat positions
    splat_depths = means3D_final[:, 2]  # [N]
    
    # Compute depth loss
    depth_loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, depth_confidence)
    
    # Add depth information to output
    render_output.update({
        "depth_loss": depth_loss,
        "id_buffer": id_buffer,
        "splat_depths": splat_depths,
    })
    
    return render_output


def get_rasterizer_id_buffer():
    """
    Utility function to access the rasterizer's internal ID buffer.
    
    This needs to be implemented based on the specific rasterizer being used.
    The goal is to extract which splat is responsible for each pixel.
    
    Returns:
        Function that can extract ID buffer from rasterizer state
    """
    # TODO: Implement based on actual rasterizer
    # This might require modifications to the rasterizer itself
    # to expose the ID buffer during rendering
    
    def extract_id_buffer_impl(rasterizer_state):
        # Placeholder implementation
        return torch.full((512, 512), -1, dtype=torch.int32, device='cuda')
    
    return extract_id_buffer_impl