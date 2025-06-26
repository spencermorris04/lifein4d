# utils/advanced_clustering.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from collections import deque
import cv2

from rigid_cluster import RigidTransform, OpticalFlowKMeans


class AdvancedRigidClustering:
    """
    Advanced rigid clustering with temporal consistency and multiple frames.
    
    Features:
    - Multi-frame flow analysis
    - Temporal consistency enforcement
    - Adaptive cluster count selection
    - Spectral clustering option
    - Motion coherence scoring
    """
    
    def __init__(
        self,
        max_clusters: int = 16,
        min_clusters: int = 4,
        temporal_window: int = 8,
        device: str = 'cuda',
        clustering_method: str = 'spectral',
        consistency_weight: float = 0.3,
        motion_threshold: float = 0.5,
    ):
        """
        Initialize advanced clustering.
        
        Args:
            max_clusters: Maximum number of clusters
            min_clusters: Minimum number of clusters
            temporal_window: Number of frames to analyze
            device: Computation device
            clustering_method: 'kmeans', 'spectral', or 'adaptive'
            consistency_weight: Weight for temporal consistency
            motion_threshold: Minimum motion for clustering
        """
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.temporal_window = temporal_window
        self.device = device
        self.clustering_method = clustering_method
        self.consistency_weight = consistency_weight
        self.motion_threshold = motion_threshold
        
        # Temporal state
        self.flow_history = deque(maxlen=temporal_window)
        self.cluster_history = deque(maxlen=temporal_window)
        self.motion_patterns = {}
        
        # Feature extraction
        self.feature_dim = 12  # [flow_mag, flow_dir_x, flow_dir_y, pos_x, pos_y, 
                              #  temporal_consistency, motion_coherence, spatial_grad_x, 
                              #  spatial_grad_y, optical_flow_divergence, curl, laplacian]
    
    def initialize_clusters_multi_frame(
        self,
        viewpoint_cams: List,
        splat_positions_3d: torch.Tensor,
        flow_manager,
        max_frames: Optional[int] = None
    ) -> Tuple[RigidTransform, torch.Tensor, Dict[str, Any]]:
        """
        Initialize clusters using multiple frames.
        
        Args:
            viewpoint_cams: List of camera viewpoints
            splat_positions_3d: 3D splat positions [N, 3]
            flow_manager: Async flow manager
            max_frames: Maximum frames to use (None = use all available)
            
        Returns:
            rigid_transforms: Initialized rigid transforms
            cluster_ids: Cluster assignments [N]
            info: Clustering information and statistics
        """
        num_frames = min(len(viewpoint_cams), max_frames or self.temporal_window)
        if num_frames < 2:
            return self._fallback_initialization(splat_positions_3d)
        
        print(f"Initializing clusters using {num_frames} frames...")
        
        # Extract multi-frame flow features
        flow_features = self._extract_multi_frame_features(
            viewpoint_cams[:num_frames], splat_positions_3d, flow_manager
        )
        
        if flow_features is None:
            return self._fallback_initialization(splat_positions_3d)
        
        # Determine optimal cluster count
        optimal_clusters = self._determine_optimal_clusters(flow_features)
        
        # Perform clustering
        cluster_ids, cluster_centers = self._perform_clustering(
            flow_features, optimal_clusters
        )
        
        # Initialize rigid transforms
        rigid_transforms = RigidTransform(
            num_clusters=optimal_clusters,
            device=self.device
        )
        
        # Optimize initial transforms using cluster assignments
        self._optimize_initial_transforms(
            rigid_transforms, splat_positions_3d, cluster_ids
        )
        
        # Compute clustering quality metrics
        info = self._compute_clustering_info(
            flow_features, cluster_ids, cluster_centers
        )
        info['num_clusters'] = optimal_clusters
        info['num_frames_used'] = num_frames
        
        print(f"Initialized {optimal_clusters} clusters with silhouette score: {info.get('silhouette_score', 'N/A'):.3f}")
        
        return rigid_transforms, cluster_ids, info
    
    def _extract_multi_frame_features(
        self,
        viewpoint_cams: List,
        splat_positions_3d: torch.Tensor,
        flow_manager
    ) -> Optional[torch.Tensor]:
        """Extract features from multiple frames."""
        N = splat_positions_3d.shape[0]
        all_features = []
        
        # Project splats to 2D for each camera
        splat_2d_all = []
        for cam in viewpoint_cams:
            splat_2d = self._project_splats_to_2d(splat_positions_3d, cam)
            splat_2d_all.append(splat_2d)
        
        # Process frame pairs
        for i in range(len(viewpoint_cams) - 1):
            cam1, cam2 = viewpoint_cams[i], viewpoint_cams[i + 1]
            splat_2d1, splat_2d2 = splat_2d_all[i], splat_2d_all[i + 1]
            
            # Convert images for flow computation
            img1 = self._camera_to_numpy(cam1)
            img2 = self._camera_to_numpy(cam2)
            
            # Request flow computation
            frame_pair_id = flow_manager.request_flow(i, i + 1, img1, img2)
            if frame_pair_id is None:
                continue
            
            # Get flow (with timeout)
            flow = flow_manager.get_flow_blocking(frame_pair_id, max_wait=5.0)
            if flow is None:
                print(f"Flow computation failed for frames {i}-{i+1}")
                continue
            
            # Extract features for this frame pair
            features = self._extract_frame_pair_features(
                flow, splat_2d1, splat_2d2, cam1.image_height, cam1.image_width
            )
            
            if features is not None:
                all_features.append(features)
                
                # Store for temporal consistency
                self.flow_history.append({
                    'flow': flow,
                    'splat_2d': splat_2d1,
                    'frame_id': i,
                    'timestamp': getattr(cam1, 'time', i)
                })
        
        if not all_features:
            print("No valid flow features extracted")
            return None
        
        # Combine features across time
        combined_features = self._combine_temporal_features(all_features)
        
        return combined_features
    
    def _extract_frame_pair_features(
        self,
        flow: np.ndarray,
        splat_2d1: torch.Tensor,
        splat_2d2: torch.Tensor,
        H: int,
        W: int
    ) -> Optional[torch.Tensor]:
        """Extract features for a single frame pair."""
        try:
            N = splat_2d1.shape[0]
            features = torch.zeros(N, self.feature_dim, device=self.device)
            
            # Convert flow to tensor
            flow_tensor = torch.from_numpy(flow).float().to(self.device)
            
            # Sample flow at splat positions
            grid = splat_2d1.clone()
            grid[:, 0] = 2 * grid[:, 0] / W - 1  # Normalize to [-1, 1]
            grid[:, 1] = 2 * grid[:, 1] / H - 1
            grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
            
            # Reshape flow for grid_sample: [1, 2, H, W]
            flow_tensor = flow_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Sample flow at splat positions
            sampled_flow = F.grid_sample(
                flow_tensor, grid, mode='bilinear', 
                padding_mode='border', align_corners=False
            ).squeeze(0).squeeze(1).transpose(0, 1)  # [N, 2]
            
            # Basic flow features
            flow_magnitude = torch.norm(sampled_flow, dim=1, keepdim=True)  # [N, 1]
            flow_direction = sampled_flow / (flow_magnitude + 1e-8)  # [N, 2]
            
            # Spatial position features (normalized)
            spatial_features = splat_2d1 / torch.tensor([W, H], device=self.device)  # [N, 2]
            
            # Advanced flow features
            flow_features = self._compute_advanced_flow_features(
                flow_tensor.squeeze(0), splat_2d1, H, W
            )  # [N, 6]
            
            # Temporal consistency (if history available)
            temporal_consistency = self._compute_temporal_consistency(
                splat_2d1, sampled_flow
            )  # [N, 1]
            
            # Combine all features
            features[:, 0:1] = flow_magnitude
            features[:, 1:3] = flow_direction
            features[:, 3:5] = spatial_features
            features[:, 5:11] = flow_features
            features[:, 11:12] = temporal_consistency
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def _compute_advanced_flow_features(
        self,
        flow_field: torch.Tensor,
        splat_2d: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """Compute advanced flow field features."""
        N = splat_2d.shape[0]
        features = torch.zeros(N, 6, device=self.device)
        
        try:
            # Compute spatial gradients of flow field
            flow_u, flow_v = flow_field[0], flow_field[1]
            
            # Gradient computation
            grad_u_x = torch.gradient(flow_u, dim=1)[0]
            grad_u_y = torch.gradient(flow_u, dim=0)[0]
            grad_v_x = torch.gradient(flow_v, dim=1)[0]
            grad_v_y = torch.gradient(flow_v, dim=0)[0]
            
            # Sample gradients at splat positions
            def sample_at_positions(field, positions):
                grid = positions.clone()
                grid[:, 0] = 2 * grid[:, 0] / W - 1
                grid[:, 1] = 2 * grid[:, 1] / H - 1
                grid = grid.unsqueeze(0).unsqueeze(0)
                
                field_batch = field.unsqueeze(0).unsqueeze(0)
                sampled = F.grid_sample(
                    field_batch, grid, mode='bilinear',
                    padding_mode='border', align_corners=False
                )
                return sampled.squeeze(0).squeeze(0).squeeze(0)
            
            # Sample all gradients
            features[:, 0] = sample_at_positions(grad_u_x, splat_2d)  # ∂u/∂x
            features[:, 1] = sample_at_positions(grad_u_y, splat_2d)  # ∂u/∂y
            features[:, 2] = sample_at_positions(grad_v_x, splat_2d)  # ∂v/∂x
            features[:, 3] = sample_at_positions(grad_v_y, splat_2d)  # ∂v/∂y
            
            # Compute divergence: ∂u/∂x + ∂v/∂y
            features[:, 4] = features[:, 0] + features[:, 3]
            
            # Compute curl: ∂v/∂x - ∂u/∂y
            features[:, 5] = features[:, 2] - features[:, 1]
            
        except Exception as e:
            print(f"Advanced flow feature computation error: {e}")
            # Return zero features on error
            pass
        
        return features
    
    def _compute_temporal_consistency(
        self,
        splat_2d: torch.Tensor,
        current_flow: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal consistency score."""
        N = splat_2d.shape[0]
        consistency = torch.ones(N, 1, device=self.device)
        
        if len(self.flow_history) < 2:
            return consistency
        
        try:
            # Compare with previous flows
            prev_flows = []
            for hist_entry in list(self.flow_history)[-3:]:  # Last 3 frames
                hist_flow = torch.from_numpy(hist_entry['flow']).float().to(self.device)
                hist_splat_2d = hist_entry['splat_2d']
                
                # Sample historical flow at current splat positions
                # (This requires tracking splat correspondences, simplified here)
                # For now, assume splat positions are roughly stable
                prev_flows.append(current_flow)  # Placeholder
            
            if prev_flows:
                # Compute variance across time
                flow_stack = torch.stack(prev_flows + [current_flow], dim=0)  # [T, N, 2]
                flow_variance = torch.var(flow_stack, dim=0).mean(dim=1, keepdim=True)  # [N, 1]
                
                # Convert variance to consistency (lower variance = higher consistency)
                consistency = torch.exp(-flow_variance)
        
        except Exception as e:
            print(f"Temporal consistency computation error: {e}")
        
        return consistency
    
    def _combine_temporal_features(self, all_features: List[torch.Tensor]) -> torch.Tensor:
        """Combine features across multiple time steps."""
        if len(all_features) == 1:
            return all_features[0]
        
        # Stack features across time
        feature_stack = torch.stack(all_features, dim=0)  # [T, N, F]
        T, N, F = feature_stack.shape
        
        # Temporal aggregation strategies
        # 1. Mean features
        mean_features = torch.mean(feature_stack, dim=0)  # [N, F]
        
        # 2. Max features (for motion magnitude)
        max_features = torch.max(feature_stack, dim=0)[0]  # [N, F]
        
        # 3. Temporal variance (consistency measure)
        var_features = torch.var(feature_stack, dim=0)  # [N, F]
        
        # Combine different aggregations
        combined = torch.cat([
            mean_features,      # [N, F]
            max_features[:, :3], # Max flow features only [N, 3]
            var_features[:, :3], # Variance of flow features [N, 3]
        ], dim=1)  # [N, F + 6]
        
        return combined
    
    def _determine_optimal_clusters(self, features: torch.Tensor) -> int:
        """Determine optimal number of clusters."""
        N = features.shape[0]
        
        if N < self.min_clusters * 10:  # Need enough points per cluster
            return max(self.min_clusters, N // 10)
        
        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        
        # Test different cluster counts
        cluster_counts = range(self.min_clusters, min(self.max_clusters + 1, N // 5))
        scores = []
        
        for k in cluster_counts:
            try:
                # Quick K-means for evaluation
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
                labels = kmeans.fit_predict(features_np)
                
                # Compute silhouette score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(features_np, labels)
                    scores.append(score)
                else:
                    scores.append(-1)
                    
            except Exception as e:
                print(f"Cluster evaluation error for k={k}: {e}")
                scores.append(-1)
        
        if not scores or max(scores) < 0:
            return self.min_clusters
        
        # Find best score with elbow method bias
        best_idx = np.argmax(scores)
        optimal_k = list(cluster_counts)[best_idx]
        
        print(f"Optimal clusters: {optimal_k} (silhouette score: {scores[best_idx]:.3f})")
        return optimal_k
    
    def _perform_clustering(
        self,
        features: torch.Tensor,
        num_clusters: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the actual clustering."""
        features_np = features.cpu().numpy()
        
        if self.clustering_method == 'spectral':
            # Spectral clustering for non-convex clusters
            clustering = SpectralClustering(
                n_clusters=num_clusters,
                affinity='rbf',
                gamma=1.0,
                random_state=42
            )
            labels = clustering.fit_predict(features_np)
            
            # Compute cluster centers
            centers = []
            for i in range(num_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    centers.append(features_np[mask].mean(axis=0))
                else:
                    centers.append(np.zeros(features.shape[1]))
            centers = np.array(centers)
            
        else:
            # K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_np)
            centers = kmeans.cluster_centers_
        
        # Convert back to tensors
        cluster_ids = torch.from_numpy(labels).long().to(self.device)
        cluster_centers = torch.from_numpy(centers).float().to(self.device)
        
        return cluster_ids, cluster_centers
    
    def _optimize_initial_transforms(
        self,
        rigid_transforms: RigidTransform,
        splat_positions: torch.Tensor,
        cluster_ids: torch.Tensor
    ):
        """Optimize initial rigid transform parameters."""
        num_clusters = rigid_transforms.num_clusters
        
        for k in range(num_clusters):
            mask = cluster_ids == k
            if mask.sum() < 3:  # Need at least 3 points
                continue
            
            cluster_points = splat_positions[mask]
            
            # Initialize to cluster centroid
            centroid = cluster_points.mean(dim=0)
            
            # Small random rotation and translation
            with torch.no_grad():
                rigid_transforms.rotation_params[k] = torch.randn(3, device=self.device) * 0.01
                rigid_transforms.translation_params[k] = (centroid - splat_positions.mean(dim=0)) * 0.1
    
    def _compute_clustering_info(
        self,
        features: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_centers: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute clustering quality metrics."""
        info = {}
        
        try:
            features_np = features.cpu().numpy()
            labels_np = cluster_ids.cpu().numpy()
            
            # Silhouette score
            if len(np.unique(labels_np)) > 1:
                info['silhouette_score'] = silhouette_score(features_np, labels_np)
            
            # Cluster sizes
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            info['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
            info['min_cluster_size'] = int(counts.min())
            info['max_cluster_size'] = int(counts.max())
            info['cluster_size_std'] = float(counts.std())
            
            # Inertia (within-cluster sum of squares)
            inertia = 0.0
            for k in range(cluster_centers.shape[0]):
                mask = cluster_ids == k
                if mask.sum() > 0:
                    cluster_features = features[mask]
                    center = cluster_centers[k:k+1]
                    distances = torch.norm(cluster_features - center, dim=1)
                    inertia += distances.sum().item()
            
            info['inertia'] = inertia
            
        except Exception as e:
            print(f"Clustering info computation error: {e}")
        
        return info
    
    def _project_splats_to_2d(
        self,
        splat_3d: torch.Tensor,
        viewpoint_cam
    ) -> torch.Tensor:
        """Project 3D splat positions to 2D image coordinates."""
        # Get 3D splat positions
        N = splat_3d.shape[0]
        
        # Project to camera coordinates
        world_view_transform = viewpoint_cam.world_view_transform.to(self.device)
        
        # Convert to homogeneous coordinates
        splat_3d_hom = torch.cat([splat_3d, torch.ones(N, 1, device=self.device)], dim=1)
        
        # Transform to camera space
        splat_cam = torch.matmul(world_view_transform, splat_3d_hom.T).T[:, :3]
        
        # Project to image coordinates
        focal_length = 500.0  # Placeholder - should extract from camera
        principal_point = torch.tensor(
            [viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2],
            device=self.device
        )
        
        # Perspective projection
        splat_2d = torch.zeros(N, 2, device=self.device)
        mask = splat_cam[:, 2] > 0.1  # Points in front of camera
        
        if mask.sum() > 0:
            splat_2d[mask, 0] = (splat_cam[mask, 0] / splat_cam[mask, 2] * focal_length + 
                               principal_point[0])
            splat_2d[mask, 1] = (splat_cam[mask, 1] / splat_cam[mask, 2] * focal_length + 
                               principal_point[1])
        
        # Clamp to image bounds
        splat_2d[:, 0] = torch.clamp(splat_2d[:, 0], 0, viewpoint_cam.image_width - 1)
        splat_2d[:, 1] = torch.clamp(splat_2d[:, 1], 0, viewpoint_cam.image_height - 1)
        
        return splat_2d
    
    def _camera_to_numpy(self, viewpoint_cam) -> np.ndarray:
        """Convert camera image to numpy array."""
        img = viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return img
    
    def _fallback_initialization(
        self,
        splat_positions_3d: torch.Tensor
    ) -> Tuple[RigidTransform, torch.Tensor, Dict[str, Any]]:
        """Fallback initialization when flow computation fails."""
        print("Using fallback random cluster initialization")
        
        N = splat_positions_3d.shape[0]
        num_clusters = self.min_clusters
        
        # Random cluster assignment
        cluster_ids = torch.randint(0, num_clusters, (N,), device=self.device)
        
        # Initialize rigid transforms
        rigid_transforms = RigidTransform(num_clusters=num_clusters, device=self.device)
        
        info = {
            'num_clusters': num_clusters,
            'method': 'fallback_random',
            'silhouette_score': 0.0
        }
        
        return rigid_transforms, cluster_ids, info


class TemporalClusterTracker:
    """Track cluster assignments over time for consistency."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.cluster_history = deque(maxlen=max_history)
        self.assignment_consistency = {}
    
    def update(self, cluster_ids: torch.Tensor, timestamp: float):
        """Update cluster history."""
        self.cluster_history.append({
            'cluster_ids': cluster_ids.clone(),
            'timestamp': timestamp
        })
        
        self._update_consistency_scores()
    
    def _update_consistency_scores(self):
        """Update consistency scores for each splat."""
        if len(self.cluster_history) < 2:
            return
        
        # Analyze cluster stability over time
        recent_assignments = [entry['cluster_ids'] for entry in list(self.cluster_history)[-5:]]
        
        if len(recent_assignments) > 1:
            # Compute assignment consistency
            assignment_stack = torch.stack(recent_assignments, dim=0)  # [T, N]
            
            # Count how often each splat changes cluster
            changes = torch.zeros_like(assignment_stack[0])
            for t in range(1, assignment_stack.shape[0]):
                changes += (assignment_stack[t] != assignment_stack[t-1]).float()
            
            # Consistency = 1 - (changes / total_comparisons)
            consistency = 1.0 - (changes / (len(recent_assignments) - 1))
            
            self.assignment_consistency = {
                'consistency_scores': consistency,
                'mean_consistency': consistency.mean().item(),
                'min_consistency': consistency.min().item(),
            }
    
    def get_consistent_assignment(
        self,
        new_cluster_ids: torch.Tensor,
        consistency_threshold: float = 0.7
    ) -> torch.Tensor:
        """Get temporally consistent cluster assignment."""
        if not self.cluster_history or 'consistency_scores' not in self.assignment_consistency:
            return new_cluster_ids
        
        # For inconsistent splats, use majority vote from history
        consistency_scores = self.assignment_consistency['consistency_scores']
        inconsistent_mask = consistency_scores < consistency_threshold
        
        if inconsistent_mask.sum() > 0:
            # Get majority vote for inconsistent splats
            recent_assignments = torch.stack([
                entry['cluster_ids'] for entry in list(self.cluster_history)[-3:]
            ], dim=0)  # [T, N]
            
            # Simple majority vote (could be improved with weighted voting)
            majority_vote = torch.mode(recent_assignments[:, inconsistent_mask], dim=0)[0]
            
            # Update inconsistent assignments
            result = new_cluster_ids.clone()
            result[inconsistent_mask] = majority_vote
            
            return result
        
        return new_cluster_ids