# rigid_cluster.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def so3_exp_map(w: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from so(3) to SO(3) using Rodrigues' formula.
    
    Args:
        w: Rotation vectors [B, 3] in axis-angle representation
        
    Returns:
        R: Rotation matrices [B, 3, 3]
    """
    batch_size = w.shape[0]
    device = w.device
    
    # Compute angle (magnitude of rotation vector)
    theta = torch.norm(w, dim=1, keepdim=True)  # [B, 1]
    
    # Handle small angles to avoid numerical instability
    small_angle = theta < 1e-4
    
    # For small angles, use first-order approximation
    # R ≈ I + [w]_× for small ||w||
    k = w / (theta + 1e-8)  # Normalized axis [B, 3]
    
    # Skew-symmetric matrix [k]_×
    K = torch.zeros(batch_size, 3, 3, device=device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    
    # Rodrigues' formula: R = I + sin(θ)[k]_× + (1-cos(θ))[k]_×²
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    sin_theta = torch.sin(theta).unsqueeze(-1)  # [B, 1, 1]
    cos_theta = torch.cos(theta).unsqueeze(-1)  # [B, 1, 1]
    
    R = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
    
    # For small angles, use linear approximation
    R_small = I + K
    
    # Select based on angle magnitude
    mask = small_angle.unsqueeze(-1).expand_as(R)
    R = torch.where(mask, R_small, R)
    
    return R


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map from SO(3) to so(3).
    
    Args:
        R: Rotation matrices [B, 3, 3]
        
    Returns:
        w: Rotation vectors [B, 3] in axis-angle representation
    """
    batch_size = R.shape[0]
    device = R.device
    
    # Compute rotation angle
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # [B]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)  # Avoid numerical issues
    theta = torch.acos(cos_theta)  # [B]
    
    # Handle small angles
    small_angle = theta < 1e-4
    
    # For general case, extract axis from skew-symmetric part
    sin_theta = torch.sin(theta)
    
    # Axis from skew-symmetric part: (R - R^T) / (2 * sin(θ))
    skew = R - R.transpose(-2, -1)  # [B, 3, 3]
    axis = torch.zeros(batch_size, 3, device=device)
    axis[:, 0] = skew[:, 2, 1]
    axis[:, 1] = skew[:, 0, 2]
    axis[:, 2] = skew[:, 1, 0]
    
    # Normalize axis
    axis_normalized = axis / (2 * sin_theta.unsqueeze(-1) + 1e-8)
    
    # Rotation vector = axis * angle
    w_general = axis_normalized * theta.unsqueeze(-1)
    
    # For small angles, use linear approximation
    w_small = axis / 2
    
    # Select based on angle magnitude
    mask = small_angle.unsqueeze(-1).expand_as(w_general)
    w = torch.where(mask, w_small, w_general)
    
    return w


class RigidTransform(nn.Module):
    """
    SE(3) rigid transformation parameterized in Lie algebra.
    
    Stores rotation as so(3) vector and translation as R³ vector.
    Provides efficient conversion to SE(3) matrices.
    """
    
    def __init__(
        self,
        num_clusters: int,
        device: str = 'cuda',
        init_std: float = 0.01,
    ):
        """
        Initialize rigid transforms for multiple clusters.
        
        Args:
            num_clusters: Number of rigid clusters
            device: Device to store parameters
            init_std: Standard deviation for parameter initialization
        """
        super().__init__()
        
        self.num_clusters = num_clusters
        self.device = device
        
        # Rotation parameters (so(3) vectors)
        self.rotation_params = nn.Parameter(
            torch.randn(num_clusters, 3, device=device) * init_std
        )
        
        # Translation parameters
        self.translation_params = nn.Parameter(
            torch.randn(num_clusters, 3, device=device) * init_std
        )
    
    def get_rotation_matrices(self) -> torch.Tensor:
        """
        Get rotation matrices from so(3) parameters.
        
        Returns:
            R: Rotation matrices [num_clusters, 3, 3]
        """
        return so3_exp_map(self.rotation_params)
    
    def get_translations(self) -> torch.Tensor:
        """
        Get translation vectors.
        
        Returns:
            t: Translation vectors [num_clusters, 3]
        """
        return self.translation_params
    
    def get_se3_matrices(self) -> torch.Tensor:
        """
        Get full SE(3) transformation matrices.
        
        Returns:
            T: SE(3) matrices [num_clusters, 4, 4]
        """
        R = self.get_rotation_matrices()  # [K, 3, 3]
        t = self.get_translations()       # [K, 3]
        
        # Construct SE(3) matrices
        T = torch.zeros(self.num_clusters, 4, 4, device=self.device)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        T[:, 3, 3] = 1.0
        
        return T
    
    def transform_points(self, points: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        """
        Transform points using cluster-specific rigid transforms.
        
        Args:
            points: Input points [N, 3]
            cluster_ids: Cluster assignment for each point [N]
            
        Returns:
            transformed_points: Transformed points [N, 3]
        """
        R = self.get_rotation_matrices()  # [K, 3, 3]
        t = self.get_translations()       # [K, 3]
        
        # Index rotation and translation for each point
        R_points = R[cluster_ids]  # [N, 3, 3]
        t_points = t[cluster_ids]  # [N, 3]
        
        # Apply transformation: R @ p + t
        transformed = torch.bmm(R_points, points.unsqueeze(-1)).squeeze(-1) + t_points
        
        return transformed
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get regularization loss for rigid transform parameters.
        
        The loss encourages small rotations and translations.
        
        Returns:
            loss: Regularization loss
        """
        # L2 penalty on rotation parameters (encourages small rotations)
        rotation_loss = torch.sum(self.rotation_params ** 2)
        
        # L2 penalty on translation parameters
        translation_loss = torch.sum(self.translation_params ** 2)
        
        return rotation_loss + translation_loss
    
    def add_clusters(self, num_new_clusters: int) -> None:
        """
        Add new rigid clusters.
        
        Args:
            num_new_clusters: Number of clusters to add
        """
        # Initialize new parameters
        new_rotations = torch.randn(num_new_clusters, 3, device=self.device) * 0.01
        new_translations = torch.randn(num_new_clusters, 3, device=self.device) * 0.01
        
        # Concatenate with existing parameters
        self.rotation_params = nn.Parameter(
            torch.cat([self.rotation_params.data, new_rotations], dim=0)
        )
        self.translation_params = nn.Parameter(
            torch.cat([self.translation_params.data, new_translations], dim=0)
        )
        
        self.num_clusters += num_new_clusters


class OpticalFlowKMeans:
    """
    K-means clustering on optical flow for rigid cluster initialization.
    
    Uses optical flow magnitude and direction to cluster splats into
    rigid motion groups.
    """
    
    def __init__(
        self,
        num_clusters: int = 8,
        max_iters: int = 100,
        tol: float = 1e-4,
        device: str = 'cuda',
    ):
        """
        Initialize optical flow K-means clustering.
        
        Args:
            num_clusters: Number of rigid clusters
            max_iters: Maximum iterations for K-means
            tol: Convergence tolerance
            device: Device for computations
        """
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.device = device
    
    def extract_flow_features(
        self,
        flow: torch.Tensor,
        positions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Extract flow features for clustering.
        
        Args:
            flow: Optical flow [H, W, 2]
            positions: 2D splat positions [N, 2] in image coordinates
            image_size: (H, W) image dimensions
            
        Returns:
            features: Flow features [N, feature_dim]
        """
        H, W = image_size
        N = positions.shape[0]
        
        # Sample flow at splat positions using bilinear interpolation
        # Normalize positions to [-1, 1] for grid_sample
        grid = positions.clone()
        grid[:, 0] = 2 * grid[:, 0] / W - 1  # x coordinates
        grid[:, 1] = 2 * grid[:, 1] / H - 1  # y coordinates
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        
        # Reshape flow for grid_sample: [1, 2, H, W]
        flow_tensor = flow.permute(2, 0, 1).unsqueeze(0)
        
        # Sample flow at splat positions
        sampled_flow = F.grid_sample(
            flow_tensor, grid, mode='bilinear', padding_mode='border', align_corners=False
        )  # [1, 2, 1, N]
        
        sampled_flow = sampled_flow.squeeze(0).squeeze(1).transpose(0, 1)  # [N, 2]
        
        # Compute flow magnitude and direction
        flow_magnitude = torch.norm(sampled_flow, dim=1, keepdim=True)  # [N, 1]
        flow_direction = sampled_flow / (flow_magnitude + 1e-8)  # [N, 2]
        
        # Spatial coordinates (normalized)
        spatial_features = positions / torch.tensor([W, H], device=self.device)  # [N, 2]
        
        # Combine features: [magnitude, direction_x, direction_y, pos_x, pos_y]
        features = torch.cat([
            flow_magnitude,
            flow_direction,
            spatial_features,
        ], dim=1)  # [N, 5]
        
        return features
    
    def cluster_flow(
        self,
        flow_features: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform K-means clustering on flow features.
        
        Args:
            flow_features: Flow features [N, feature_dim]
            weights: Optional weights for each point [N]
            
        Returns:
            cluster_ids: Cluster assignment [N]
            centroids: Cluster centroids [K, feature_dim]
        """
        N, feature_dim = flow_features.shape
        
        if weights is None:
            weights = torch.ones(N, device=self.device)
        
        # Initialize centroids using K-means++
        centroids = self._kmeans_plus_plus_init(flow_features, weights)
        
        # K-means iterations
        for iteration in range(self.max_iters):
            # Assign points to nearest centroids
            distances = torch.cdist(flow_features, centroids)  # [N, K]
            cluster_ids = torch.argmin(distances, dim=1)  # [N]
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.num_clusters):
                mask = cluster_ids == k
                if mask.sum() > 0:
                    cluster_weights = weights[mask]
                    weighted_points = flow_features[mask] * cluster_weights.unsqueeze(1)
                    new_centroids[k] = weighted_points.sum(dim=0) / cluster_weights.sum()
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]
            
            # Check convergence
            centroid_shift = torch.norm(new_centroids - centroids)
            if centroid_shift < self.tol:
                break
                
            centroids = new_centroids
        
        return cluster_ids, centroids
    
    def _kmeans_plus_plus_init(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize centroids using K-means++ algorithm.
        
        Args:
            features: Feature vectors [N, D]
            weights: Point weights [N]
            
        Returns:
            centroids: Initial centroids [K, D]
        """
        N, D = features.shape
        centroids = torch.zeros(self.num_clusters, D, device=self.device)
        
        # Choose first centroid randomly (weighted by point weights)
        probs = weights / weights.sum()
        first_idx = torch.multinomial(probs, 1).item()
        centroids[0] = features[first_idx]
        
        # Choose remaining centroids
        for k in range(1, self.num_clusters):
            # Compute distances to nearest existing centroid
            distances = torch.cdist(features, centroids[:k])  # [N, k]
            min_distances = torch.min(distances, dim=1)[0]  # [N]
            
            # Choose next centroid with probability proportional to squared distance
            probs = weights * min_distances ** 2
            probs = probs / probs.sum()
            
            next_idx = torch.multinomial(probs, 1).item()
            centroids[k] = features[next_idx]
        
        return centroids
    
    def update_clusters_online(
        self,
        cluster_ids: torch.Tensor,
        centroids: torch.Tensor,
        new_features: torch.Tensor,
        learning_rate: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cluster assignments and centroids with new data.
        
        Args:
            cluster_ids: Current cluster assignments [N]
            centroids: Current centroids [K, D]
            new_features: New feature vectors [M, D]
            learning_rate: Learning rate for centroid updates
            
        Returns:
            new_cluster_ids: Updated cluster assignments [M]
            updated_centroids: Updated centroids [K, D]
        """
        # Assign new points to clusters
        distances = torch.cdist(new_features, centroids)  # [M, K]
        new_cluster_ids = torch.argmin(distances, dim=1)  # [M]
        
        # Update centroids with new points
        updated_centroids = centroids.clone()
        for k in range(self.num_clusters):
            mask = new_cluster_ids == k
            if mask.sum() > 0:
                new_points = new_features[mask]
                centroid_update = new_points.mean(dim=0)
                updated_centroids[k] = (1 - learning_rate) * centroids[k] + learning_rate * centroid_update
        
        return new_cluster_ids, updated_centroids


def initialize_rigid_clusters(
    splat_positions: torch.Tensor,
    optical_flow: torch.Tensor,
    image_size: Tuple[int, int],
    num_clusters: int = 8,
    device: str = 'cuda',
) -> Tuple[RigidTransform, torch.Tensor]:
    """
    Initialize rigid clusters using optical flow.
    
    Args:
        splat_positions: 2D positions of splats [N, 2]
        optical_flow: Optical flow field [H, W, 2]
        image_size: (H, W) image dimensions
        num_clusters: Number of rigid clusters
        device: Device for computations
        
    Returns:
        rigid_transforms: Initialized rigid transform parameters
        cluster_ids: Cluster assignment for each splat [N]
    """
    # Create clustering algorithm
    clusterer = OpticalFlowKMeans(num_clusters=num_clusters, device=device)
    
    # Extract flow features
    flow_features = clusterer.extract_flow_features(
        optical_flow, splat_positions, image_size
    )
    
    # Perform clustering
    cluster_ids, _ = clusterer.cluster_flow(flow_features)
    
    # Initialize rigid transforms
    rigid_transforms = RigidTransform(num_clusters=num_clusters, device=device)
    
    return rigid_transforms, cluster_ids