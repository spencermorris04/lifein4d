# loss_depth_pose.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from kernels.triton_deform import apply_deformation, compute_depth_loss
from rigid_cluster import RigidTransform


class DepthReprojectionLoss(nn.Module):
    """
    Depth reprojection loss using confidence-weighted metric depth.
    
    Computes L1 loss between predicted splat depths and ground truth
    metric depth from foundation models (e.g., Depth Pro).
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        confidence_threshold: float = 0.1,
        max_depth: float = 100.0,
        normalize_by_depth: bool = True,
    ):
        """
        Initialize depth reprojection loss.
        
        Args:
            loss_type: Loss function type ('l1', 'l2', 'huber')
            confidence_threshold: Minimum confidence to include pixel
            max_depth: Maximum valid depth value
            normalize_by_depth: Whether to normalize loss by depth magnitude
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth
        self.normalize_by_depth = normalize_by_depth
    
    def forward(
        self,
        splat_depths: torch.Tensor,
        id_buffer: torch.Tensor,
        gt_depth: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute depth reprojection loss.
        
        Args:
            splat_depths: Z-coordinates of splats [N]
            id_buffer: Splat ID buffer from rasterizer [H, W]
            gt_depth: Ground truth depth map [H, W]
            confidence: Depth confidence weights [H, W]
            
        Returns:
            loss: Scalar depth loss
            info: Dictionary with loss information
        """
        # Filter by confidence and depth validity
        valid_mask = (
            (confidence > self.confidence_threshold) &
            (gt_depth > 0) &
            (gt_depth < self.max_depth) &
            (id_buffer >= 0)
        )
        
        if not valid_mask.any():
            # No valid pixels
            return torch.tensor(0.0, device=splat_depths.device), {'num_valid': 0}
        
        # Apply confidence and validity filtering
        filtered_confidence = confidence * valid_mask.float()
        
        # Use Triton kernel for efficient computation
        loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, filtered_confidence)
        
        # Collect statistics
        num_valid = valid_mask.sum().item()
        mean_confidence = filtered_confidence[valid_mask].mean().item()
        mean_depth_error = 0.0
        
        if num_valid > 0:
            # Compute mean error for logging
            valid_ids = id_buffer[valid_mask]
            valid_gt = gt_depth[valid_mask]
            valid_pred = splat_depths[valid_ids]
            mean_depth_error = (valid_pred - valid_gt).abs().mean().item()
        
        info = {
            'num_valid': num_valid,
            'mean_confidence': mean_confidence,
            'mean_depth_error': mean_depth_error,
        }
        
        return loss, info


class PoseConsistencyLoss(nn.Module):
    """
    Pose consistency loss using SfM landmarks.
    
    Encourages 3D splat positions to be consistent with camera poses
    and 3D landmarks from structure-from-motion.
    """
    
    def __init__(
        self,
        loss_type: str = 'l2',
        max_distance: float = 1.0,
        landmark_weight: float = 1.0,
    ):
        """
        Initialize pose consistency loss.
        
        Args:
            loss_type: Loss function type ('l1', 'l2', 'huber')
            max_distance: Maximum distance for valid landmark matches
            landmark_weight: Weight for landmark consistency term
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.max_distance = max_distance
        self.landmark_weight = landmark_weight
    
    def forward(
        self,
        splat_positions: torch.Tensor,
        camera_pose: torch.Tensor,
        landmarks_3d: torch.Tensor,
        landmarks_2d: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute pose consistency loss.
        
        Args:
            splat_positions: 3D positions of splats [N, 3]
            camera_pose: Camera pose matrix [4, 4]
            landmarks_3d: 3D landmarks from SfM [M, 3]
            landmarks_2d: Corresponding 2D observations [M, 2]
            camera_intrinsics: Camera intrinsic matrix [3, 3]
            
        Returns:
            loss: Scalar pose consistency loss
            info: Dictionary with loss information
        """
        if landmarks_3d.shape[0] == 0:
            return torch.tensor(0.0, device=splat_positions.device), {'num_matches': 0}
        
        # Transform landmarks to camera frame
        landmarks_3d_hom = torch.cat([
            landmarks_3d,
            torch.ones(landmarks_3d.shape[0], 1, device=landmarks_3d.device)
        ], dim=1)  # [M, 4]
        
        landmarks_cam = torch.matmul(camera_pose, landmarks_3d_hom.T).T[:, :3]  # [M, 3]
        
        # Find nearest splats to each landmark
        distances = torch.cdist(splat_positions, landmarks_cam)  # [N, M]
        nearest_splat_ids = torch.argmin(distances, dim=0)  # [M]
        min_distances = distances[nearest_splat_ids, torch.arange(landmarks_cam.shape[0])]
        
        # Filter by maximum distance threshold
        valid_mask = min_distances < self.max_distance
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=splat_positions.device), {'num_matches': 0}
        
        # Compute loss for valid matches
        valid_landmarks = landmarks_cam[valid_mask]  # [V, 3]
        valid_splat_ids = nearest_splat_ids[valid_mask]  # [V]
        valid_splats = splat_positions[valid_splat_ids]  # [V, 3]
        
        # Position consistency loss
        if self.loss_type == 'l1':
            position_loss = F.l1_loss(valid_splats, valid_landmarks)
        elif self.loss_type == 'l2':
            position_loss = F.mse_loss(valid_splats, valid_landmarks)
        elif self.loss_type == 'huber':
            position_loss = F.huber_loss(valid_splats, valid_landmarks)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Collect statistics
        num_matches = valid_mask.sum().item()
        mean_distance = min_distances[valid_mask].mean().item()
        
        info = {
            'num_matches': num_matches,
            'mean_distance': mean_distance,
        }
        
        total_loss = self.landmark_weight * position_loss
        
        return total_loss, info


class DeformationRegularizer(nn.Module):
    """
    Regularization terms for deformation field and basis matrices.
    
    Encourages smooth deformations and prevents overfitting.
    """
    
    def __init__(
        self,
        basis_weight: float = 1e-4,
        temporal_weight: float = 1e-3,
        spatial_weight: float = 1e-4,
    ):
        """
        Initialize deformation regularizer.
        
        Args:
            basis_weight: Weight for basis matrix L2 regularization
            temporal_weight: Weight for temporal smoothness
            spatial_weight: Weight for spatial smoothness
        """
        super().__init__()
        
        self.basis_weight = basis_weight
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
    
    def forward(
        self,
        basis_matrices: torch.Tensor,
        time_coefficients: torch.Tensor,
        splat_positions: torch.Tensor,
        neighbor_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute deformation regularization loss.
        
        Args:
            basis_matrices: Deformation basis matrices [N, 3, r]
            time_coefficients: Time-dependent coefficients [r] or [T, r]
            splat_positions: 3D positions of splats [N, 3]
            neighbor_ids: Nearest neighbor indices for spatial smoothness [N, K]
            
        Returns:
            loss: Total regularization loss
            info: Dictionary with loss breakdown
        """
        total_loss = 0.0
        info = {}
        
        # Basis matrix L2 regularization
        if self.basis_weight > 0:
            basis_loss = torch.sum(basis_matrices ** 2)
            total_loss += self.basis_weight * basis_loss
            info['basis_loss'] = basis_loss.item()
        
        # Temporal smoothness (if multiple time steps available)
        if self.temporal_weight > 0 and time_coefficients.dim() > 1:
            T = time_coefficients.shape[0]
            if T > 1:
                # Finite difference in time
                temporal_diff = time_coefficients[1:] - time_coefficients[:-1]  # [T-1, r]
                temporal_loss = torch.sum(temporal_diff ** 2)
                total_loss += self.temporal_weight * temporal_loss
                info['temporal_loss'] = temporal_loss.item()
        
        # Spatial smoothness (if neighbor information available)
        if self.spatial_weight > 0 and neighbor_ids is not None:
            N, K = neighbor_ids.shape
            
            # Compute differences between neighboring basis matrices
            neighbor_bases = basis_matrices[neighbor_ids]  # [N, K, 3, r]
            splat_bases = basis_matrices.unsqueeze(1).expand(-1, K, -1, -1)  # [N, K, 3, r]
            
            spatial_diff = neighbor_bases - splat_bases  # [N, K, 3, r]
            spatial_loss = torch.sum(spatial_diff ** 2)
            total_loss += self.spatial_weight * spatial_loss
            info['spatial_loss'] = spatial_loss.item()
        
        return total_loss, info


class RigidRegularizer(nn.Module):
    """
    Regularization for rigid cluster transformations.
    
    Encourages small rotations and translations for stability.
    """
    
    def __init__(
        self,
        rotation_weight: float = 1e-3,
        translation_weight: float = 1e-3,
    ):
        """
        Initialize rigid regularizer.
        
        Args:
            rotation_weight: Weight for rotation regularization
            translation_weight: Weight for translation regularization
        """
        super().__init__()
        
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def forward(self, rigid_transforms: RigidTransform) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rigid transformation regularization loss.
        
        Args:
            rigid_transforms: Rigid transformation parameters
            
        Returns:
            loss: Total regularization loss
            info: Dictionary with loss breakdown
        """
        total_loss = 0.0
        info = {}
        
        # Rotation regularization (L2 on so(3) parameters)
        if self.rotation_weight > 0:
            rotation_loss = torch.sum(rigid_transforms.rotation_params ** 2)
            total_loss += self.rotation_weight * rotation_loss
            info['rotation_loss'] = rotation_loss.item()
        
        # Translation regularization
        if self.translation_weight > 0:
            translation_loss = torch.sum(rigid_transforms.translation_params ** 2)
            total_loss += self.translation_weight * translation_loss
            info['translation_loss'] = translation_loss.item()
        
        return total_loss, info


class Mono4DGSLoss(nn.Module):
    """
    Combined loss function for monocular 4D Gaussian Splatting.
    
    Integrates photometric, depth, pose, and regularization losses.
    """
    
    def __init__(
        self,
        lambda_depth: float = 1.0,
        lambda_pose: float = 0.1,
        lambda_deform: float = 1e-4,
        lambda_rigid: float = 1e-3,
        **loss_kwargs
    ):
        """
        Initialize combined loss function.
        
        Args:
            lambda_depth: Weight for depth reprojection loss
            lambda_pose: Weight for pose consistency loss
            lambda_deform: Weight for deformation regularization
            lambda_rigid: Weight for rigid regularization
            **loss_kwargs: Additional arguments for individual loss components
        """
        super().__init__()
        
        self.lambda_depth = lambda_depth
        self.lambda_pose = lambda_pose
        self.lambda_deform = lambda_deform
        self.lambda_rigid = lambda_rigid
        
        # Initialize loss components
        self.depth_loss = DepthReprojectionLoss(**loss_kwargs.get('depth', {}))
        self.pose_loss = PoseConsistencyLoss(**loss_kwargs.get('pose', {}))
        self.deform_regularizer = DeformationRegularizer(**loss_kwargs.get('deform', {}))
        self.rigid_regularizer = RigidRegularizer(**loss_kwargs.get('rigid', {}))
    
    def forward(
        self,
        # Standard 4DGS outputs
        rendered_image: torch.Tensor,
        gt_image: torch.Tensor,
        splat_positions: torch.Tensor,
        
        # Depth-related inputs
        splat_depths: torch.Tensor,
        id_buffer: torch.Tensor,
        gt_depth: torch.Tensor,
        depth_confidence: torch.Tensor,
        
        # Pose-related inputs
        camera_pose: torch.Tensor,
        landmarks_3d: Optional[torch.Tensor] = None,
        landmarks_2d: Optional[torch.Tensor] = None,
        camera_intrinsics: Optional[torch.Tensor] = None,
        
        # Deformation inputs
        basis_matrices: Optional[torch.Tensor] = None,
        time_coefficients: Optional[torch.Tensor] = None,
        rigid_transforms: Optional[RigidTransform] = None,
        
        # Additional loss weights
        photometric_loss: Optional[torch.Tensor] = None,
        opacity_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute combined loss.
        
        Args:
            rendered_image: Rendered RGB image [3, H, W]
            gt_image: Ground truth RGB image [3, H, W]
            splat_positions: 3D positions of splats [N, 3]
            splat_depths: Z-coordinates of splats [N]
            id_buffer: Splat ID buffer [H, W]
            gt_depth: Ground truth depth [H, W]
            depth_confidence: Depth confidence [H, W]
            camera_pose: Camera pose matrix [4, 4]
            landmarks_3d: 3D landmarks [M, 3]
            landmarks_2d: 2D landmarks [M, 2]
            camera_intrinsics: Camera intrinsics [3, 3]
            basis_matrices: Deformation basis [N, 3, r]
            time_coefficients: Time coefficients [r]
            rigid_transforms: Rigid transformations
            photometric_loss: Pre-computed photometric loss
            opacity_loss: Pre-computed opacity loss
            
        Returns:
            total_loss: Combined loss value
            loss_info: Dictionary with loss breakdown and statistics
        """
        total_loss = 0.0
        loss_info = {}
        
        # Photometric loss (standard 4DGS)
        if photometric_loss is not None:
            total_loss += photometric_loss
            loss_info['photometric_loss'] = photometric_loss.item()
        else:
            # Compute L1 loss if not provided
            photo_loss = F.l1_loss(rendered_image, gt_image)
            total_loss += photo_loss
            loss_info['photometric_loss'] = photo_loss.item()
        
        # Opacity loss (standard 4DGS)
        if opacity_loss is not None:
            total_loss += opacity_loss
            loss_info['opacity_loss'] = opacity_loss.item()
        
        # Depth reprojection loss
        if self.lambda_depth > 0:
            depth_loss_val, depth_info = self.depth_loss(
                splat_depths, id_buffer, gt_depth, depth_confidence
            )
            total_loss += self.lambda_depth * depth_loss_val
            loss_info['depth_loss'] = depth_loss_val.item()
            loss_info.update({f'depth_{k}': v for k, v in depth_info.items()})
        
        # Pose consistency loss
        if self.lambda_pose > 0 and landmarks_3d is not None:
            pose_loss_val, pose_info = self.pose_loss(
                splat_positions, camera_pose, landmarks_3d, landmarks_2d, camera_intrinsics
            )
            total_loss += self.lambda_pose * pose_loss_val
            loss_info['pose_loss'] = pose_loss_val.item()
            loss_info.update({f'pose_{k}': v for k, v in pose_info.items()})
        
        # Deformation regularization
        if self.lambda_deform > 0 and basis_matrices is not None:
            deform_loss_val, deform_info = self.deform_regularizer(
                basis_matrices, time_coefficients, splat_positions
            )
            total_loss += self.lambda_deform * deform_loss_val
            loss_info['deform_reg_loss'] = deform_loss_val.item()
            loss_info.update({f'deform_{k}': v for k, v in deform_info.items()})
        
        # Rigid regularization
        if self.lambda_rigid > 0 and rigid_transforms is not None:
            rigid_loss_val, rigid_info = self.rigid_regularizer(rigid_transforms)
            total_loss += self.lambda_rigid * rigid_loss_val
            loss_info['rigid_reg_loss'] = rigid_loss_val.item()
            loss_info.update({f'rigid_{k}': v for k, v in rigid_info.items()})
        
        # Store total loss
        loss_info['total_loss'] = total_loss.item()
        
        return total_loss, loss_info
    
    def update_weights(
        self,
        iteration: int,
        schedule: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update loss weights based on training schedule.
        
        Args:
            iteration: Current training iteration
            schedule: Dictionary with weight schedules
        """
        if schedule is None:
            return
        
        # Example scheduling (can be customized)
        if 'depth' in schedule:
            self.lambda_depth = schedule['depth'].get(iteration, self.lambda_depth)
        
        if 'pose' in schedule:
            self.lambda_pose = schedule['pose'].get(iteration, self.lambda_pose)
        
        if 'deform' in schedule:
            self.lambda_deform = schedule['deform'].get(iteration, self.lambda_deform)
        
        if 'rigid' in schedule:
            self.lambda_rigid = schedule['rigid'].get(iteration, self.lambda_rigid)