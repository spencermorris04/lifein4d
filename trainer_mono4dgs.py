# trainer_mono4dgs.py

import numpy as np
import random
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from time import time
import copy
from collections import deque
import threading
import queue

# Core 4DGS imports
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, render_with_depth_loss, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
from utils.scene_utils import render_training_image

# Mono4DGS imports
from scene.deformation_field import LowRankDeformationField, create_deformation_field
from rigid_cluster import RigidTransform, initialize_rigid_clusters
from loss_depth_pose import Mono4DGSLoss
from kernels.triton_deform import compile_kernels

# Optimized imports
from utils.async_flow_manager import AsyncOpticalFlowManager
from utils.advanced_clustering import AdvancedRigidClustering, TemporalClusterTracker
from utils.adaptive_loss_scheduler import AdaptiveLossScheduler, MultiStageScheduler

# Foundation model imports (these would need to be installed separately)
try:
    import depth_pro  # Depth Pro foundation model
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False
    print("Warning: Depth Pro not available. Install from Apple ML Research.")

try:
    import mast3r  # MASt3R-SfM for pose estimation
    MAST3R_AVAILABLE = True
except ImportError:
    MAST3R_AVAILABLE = False
    print("Warning: MASt3R not available. Install from Naver Labs.")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


class FoundationModelManager:
    """
    Manages foundation models for depth and pose estimation.
    Handles asynchronous inference and caching.
    """
    
    def __init__(self, device='cuda', cache_size=100):
        self.device = device
        self.cache_size = cache_size
        
        # Initialize models
        self.depth_model = None
        self.pose_model = None
        
        # Caches for computed results
        self.depth_cache = {}
        self.pose_cache = {}
        
        # Async processing
        self.depth_queue = queue.Queue(maxsize=10)
        self.pose_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        self._load_models()
        self._start_workers()
    
    def _load_models(self):
        """Load foundation models."""
        if DEPTH_PRO_AVAILABLE:
            print("Loading Depth Pro model...")
            self.depth_model = depth_pro.load_model()
            self.depth_model.to(self.device)
            self.depth_model.eval()
        
        if MAST3R_AVAILABLE:
            print("Loading MASt3R model...")
            self.pose_model = mast3r.load_model()
            self.pose_model.to(self.device)
            self.pose_model.eval()
    
    def _start_workers(self):
        """Start background worker threads."""
        if self.depth_model is not None:
            self.depth_worker = threading.Thread(target=self._depth_worker)
            self.depth_worker.daemon = True
            self.depth_worker.start()
        
        if self.pose_model is not None:
            self.pose_worker = threading.Thread(target=self._pose_worker)
            self.pose_worker.daemon = True
            self.pose_worker.start()
    
    def _depth_worker(self):
        """Background worker for depth estimation."""
        while True:
            try:
                item = self.depth_queue.get(timeout=1.0)
                if item is None:
                    break
                
                frame_id, image = item
                
                # Check cache first
                if frame_id in self.depth_cache:
                    continue
                
                # Compute depth
                with torch.no_grad():
                    depth, confidence = self.depth_model.infer(image)
                
                # Cache result
                self.depth_cache[frame_id] = (depth, confidence)
                
                # Manage cache size
                if len(self.depth_cache) > self.cache_size:
                    oldest_key = next(iter(self.depth_cache))
                    del self.depth_cache[oldest_key]
                
                self.result_queue.put(('depth', frame_id, depth, confidence))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Depth worker error: {e}")
    
    def _pose_worker(self):
        """Background worker for pose estimation."""
        while True:
            try:
                item = self.pose_queue.get(timeout=1.0)
                if item is None:
                    break
                
                frame_ids, images = item
                
                # Check cache
                cache_key = tuple(frame_ids)
                if cache_key in self.pose_cache:
                    continue
                
                # Compute poses
                with torch.no_grad():
                    poses, landmarks = self.pose_model.estimate_poses(images)
                
                # Cache result
                self.pose_cache[cache_key] = (poses, landmarks)
                
                # Manage cache size
                if len(self.pose_cache) > self.cache_size:
                    oldest_key = next(iter(self.pose_cache))
                    del self.pose_cache[oldest_key]
                
                self.result_queue.put(('pose', frame_ids, poses, landmarks))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Pose worker error: {e}")
    
    def request_depth(self, frame_id, image):
        """Request depth estimation for a frame."""
        try:
            self.depth_queue.put_nowait((frame_id, image))
        except queue.Full:
            pass  # Skip if queue is full
    
    def request_pose(self, frame_ids, images):
        """Request pose estimation for frame pairs."""
        try:
            self.pose_queue.put_nowait((frame_ids, images))
        except queue.Full:
            pass  # Skip if queue is full
    
    def get_depth(self, frame_id):
        """Get cached depth result."""
        return self.depth_cache.get(frame_id, (None, None))
    
    def get_pose(self, frame_ids):
        """Get cached pose result."""
        cache_key = tuple(frame_ids)
        return self.pose_cache.get(cache_key, (None, None))
    
    def cleanup(self):
        """Cleanup resources."""
        # Stop workers
        if hasattr(self, 'depth_worker'):
            self.depth_queue.put(None)
            self.depth_worker.join()
        
        if hasattr(self, 'pose_worker'):
            self.pose_queue.put(None)
            self.pose_worker.join()


class TwoFrameBuffer:
    """
    Streaming buffer for two-frame processing.
    Maintains current and previous frames for optical flow and pose estimation.
    """
    
    def __init__(self, max_buffer_size=10):
        self.max_buffer_size = max_buffer_size
        self.frames = deque(maxlen=max_buffer_size)
        self.current_idx = 0
    
    def add_frame(self, viewpoint_cam, gt_depth=None, depth_confidence=None):
        """Add a new frame to the buffer."""
        frame_data = {
            'viewpoint': viewpoint_cam,
            'frame_id': self.current_idx,
            'gt_depth': gt_depth,
            'depth_confidence': depth_confidence,
            'timestamp': time(),
        }
        
        self.frames.append(frame_data)
        self.current_idx += 1
        
        return frame_data
    
    def get_frame_pair(self):
        """Get current and previous frame for processing."""
        if len(self.frames) < 2:
            return None, None
        
        current_frame = self.frames[-1]
        prev_frame = self.frames[-2]
        
        return current_frame, prev_frame
    
    def get_current_frame(self):
        """Get the most recent frame."""
        if len(self.frames) == 0:
            return None
        return self.frames[-1]


class Mono4DGSTrainer:
    """
    Main trainer for monocular 4D Gaussian Splatting.
    
    Integrates depth and pose foundation models with 4DGS optimization.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = 'cuda'
        
        # Initialize optimized components
        self.foundation_manager = FoundationModelManager(device=self.device)
        self.frame_buffer = TwoFrameBuffer()
        
        # Optimized optical flow manager
        self.flow_manager = AsyncOpticalFlowManager(
            device=self.device,
            max_queue_size=getattr(args, 'flow_queue_size', 10),
            cache_size=getattr(args, 'flow_cache_size', 50),
            flow_method=getattr(args, 'flow_method', 'auto')
        )
        
        # Advanced clustering system
        self.clustering_system = AdvancedRigidClustering(
            max_clusters=getattr(args, 'max_clusters', 16),
            min_clusters=getattr(args, 'min_clusters', 4),
            temporal_window=getattr(args, 'clustering_temporal_window', 8),
            device=self.device,
            clustering_method=getattr(args, 'clustering_method', 'spectral')
        )
        
        # Temporal cluster tracker
        self.cluster_tracker = TemporalClusterTracker(
            max_history=getattr(args, 'cluster_history_size', 10)
        )
        
        # Adaptive loss scheduler
        self.loss_scheduler = self._create_loss_scheduler()
        
        # Training state
        self.iteration = 0
        self.grad_mu_z_buffer = None  # Reusable gradient buffer
        
        # Models
        self.gaussians = None
        self.scene = None
        self.deformation_field = None
        self.rigid_transforms = None
        self.loss_function = None
        
        # Optimizers
        self.deform_optimizer = None
        self.rigid_optimizer = None
        self.deform_basis_optimizer = None  # Separate fp16 optimizer
        
        print("Mono4DGS Trainer initialized with optimized systems")
    
    def setup_models(self, dataset):
        """Initialize all models and optimizers."""
        # Initialize Gaussian model with mono4dgs enabled
        self.gaussians = GaussianModel(dataset.sh_degree, self.args)
        self.gaussians.enable_mono4dgs = True
        
        # Initialize scene
        self.scene = Scene(dataset, self.gaussians, load_coarse=None)
        
        # Initialize deformation field
        self.deformation_field, _ = create_deformation_field(
            rank=getattr(self.args, 'deform_rank', 4),
            time_embed_dim=getattr(self.args, 'time_embed_dim', 32),
            hidden_dim=getattr(self.args, 'deform_hidden_dim', 64),
            num_layers=getattr(self.args, 'deform_num_layers', 3),
        )
        self.deformation_field.to(self.device)
        
        # Initialize rigid transforms
        num_clusters = getattr(self.args, 'num_rigid_clusters', 8)
        self.rigid_transforms = RigidTransform(
            num_clusters=num_clusters,
            device=self.device
        )
        
        # Initialize loss function
        self.loss_function = Mono4DGSLoss(
            lambda_depth=getattr(self.args, 'lambda_depth', 1.0),
            lambda_pose=getattr(self.args, 'lambda_pose', 0.1),
            lambda_deform=getattr(self.args, 'lambda_deform', 1e-4),
            lambda_rigid=getattr(self.args, 'lambda_rigid', 1e-3),
        )
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Compile Triton kernels
        compile_kernels()
        
        print(f"Models initialized: {self.gaussians.get_xyz.shape[0]} splats")
    
    def _create_loss_scheduler(self):
        """Create adaptive loss scheduler."""
        initial_weights = {
            'photometric_loss': 1.0,
            'depth_loss': getattr(self.args, 'lambda_depth', 1.0),
            'pose_loss': getattr(self.args, 'lambda_pose', 0.1),
            'deform_reg_loss': getattr(self.args, 'lambda_deform', 1e-4),
            'rigid_reg_loss': getattr(self.args, 'lambda_rigid', 1e-3),
        }
        
        # Define warm-up periods
        warmup_iterations = {
            'photometric_loss': 0,      # Start immediately
            'depth_loss': getattr(self.args, 'depth_warmup', 100),
            'pose_loss': getattr(self.args, 'pose_warmup', 500),
            'deform_reg_loss': 0,
            'rigid_reg_loss': 0,
        }
        
        # Target loss ratios (relative to photometric loss)
        target_loss_ratios = {
            'depth_loss': 0.1,      # Depth loss should be ~10% of photometric
            'pose_loss': 0.05,      # Pose loss should be ~5% of photometric
        }
        
        scheduler_method = getattr(self.args, 'loss_adaptation', 'gradient_balance')
        
        return AdaptiveLossScheduler(
            initial_weights=initial_weights,
            adaptation_method=scheduler_method,
            window_size=getattr(self.args, 'loss_window_size', 50),
            adaptation_rate=getattr(self.args, 'loss_adaptation_rate', 0.1),
            warmup_iterations=warmup_iterations,
            target_loss_ratios=target_loss_ratios,
        )
    
    def _setup_optimizers(self):
        """Setup optimizers for new components."""
        # Deformation field optimizer
        deform_lr = getattr(self.args, 'deform_lr', 1e-3)
        self.deform_optimizer = torch.optim.AdamW(
            self.deformation_field.parameters(),
            lr=deform_lr,
            weight_decay=1e-4
        )
        
        # Rigid transform optimizer
        rigid_lr = getattr(self.args, 'rigid_lr', 1e-3)
        self.rigid_optimizer = torch.optim.AdamW([
            {'params': self.rigid_transforms.rotation_params, 'lr': rigid_lr},
            {'params': self.rigid_transforms.translation_params, 'lr': rigid_lr * 0.1},
        ], weight_decay=1e-4)
        
        # Separate fp16 optimizer for deformation basis matrices
        if self.gaussians._deformation_manager is not None:
            deform_basis_lr = getattr(self.args, 'deform_basis_lr', 1e-3)
            self.deform_basis_optimizer = torch.optim.AdamW(
                [self.gaussians._deformation_manager.W],
                lr=deform_basis_lr,
                weight_decay=1e-4,
                eps=1e-4  # Larger eps for fp16 stability
            )
    
    def initialize_clusters(self, viewpoint_cams):
        """Initialize rigid clusters using advanced multi-frame analysis."""
        if len(viewpoint_cams) < 2:
            print("Warning: Need at least 2 frames for cluster initialization")
            return
        
        print("Initializing clusters with advanced multi-frame analysis...")
        
        # Use the advanced clustering system
        try:
            rigid_transforms, cluster_ids, clustering_info = self.clustering_system.initialize_clusters_multi_frame(
                viewpoint_cams=viewpoint_cams,
                splat_positions_3d=self.gaussians.get_xyz,
                flow_manager=self.flow_manager,
                max_frames=getattr(self.args, 'clustering_max_frames', 8)
            )
            
            # Update rigid transforms and cluster assignments
            self.rigid_transforms = rigid_transforms
            self.gaussians.set_cluster_assignments(cluster_ids)
            
            # Initialize cluster tracker
            self.cluster_tracker.update(cluster_ids, timestamp=0.0)
            
            print(f"Advanced clustering complete:")
            print(f"  Clusters: {clustering_info['num_clusters']}")
            print(f"  Frames used: {clustering_info['num_frames_used']}")
            print(f"  Silhouette score: {clustering_info.get('silhouette_score', 'N/A'):.3f}")
            print(f"  Method: {clustering_info.get('method', 'advanced')}")
            
        except Exception as e:
            print(f"Advanced clustering failed: {e}")
            print("Falling back to simple initialization...")
            self._simple_cluster_initialization(viewpoint_cams)
    
    def _simple_cluster_initialization(self, viewpoint_cams):
        """Fallback simple cluster initialization."""
        # Simple fallback using first two frames
        cam1, cam2 = viewpoint_cams[0], viewpoint_cams[1]
        
        # Request optical flow computation
        img1 = self._camera_to_numpy(cam1)
        img2 = self._camera_to_numpy(cam2)
        
        frame_pair_id = self.flow_manager.request_flow(0, 1, img1, img2)
        optical_flow = None
        
        if frame_pair_id:
            optical_flow = self.flow_manager.get_flow_blocking(frame_pair_id, max_wait=10.0)
        
        if optical_flow is None:
            print("Flow computation failed, using random clusters")
            optical_flow = np.zeros((cam1.image_height, cam1.image_width, 2), dtype=np.float32)
        
        # Convert to tensor
        optical_flow = torch.from_numpy(optical_flow).float().to(self.device)
        
        # Project splats to 2D
        splat_2d = self._project_splats_to_2d(cam1)
        
        # Initialize clusters using simple method
        rigid_transforms, cluster_ids = initialize_rigid_clusters(
            splat_2d,
            optical_flow,
            (cam1.image_height, cam1.image_width),
            num_clusters=getattr(self.args, 'num_rigid_clusters', 8),
            device=self.device
        )
        
        # Update components
        self.rigid_transforms = rigid_transforms
        self.gaussians.set_cluster_assignments(cluster_ids)
        self.cluster_tracker.update(cluster_ids, timestamp=0.0)
        
        print(f"Simple clustering complete: {self.rigid_transforms.num_clusters} clusters")
    
    def _compute_optical_flow(self, img1_gray, img2_gray):
        """Compute optical flow between two grayscale images."""
        try:
            import cv2
            # Use OpenCV's Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                img1_gray, img2_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            print("Using OpenCV optical flow")
            return flow
            
        except ImportError:
            try:
                # Fallback to RAFT model from torch.hub
                import torch
                raft_model = torch.hub.load('pytorch/vision:v0.10.0', 'raft_small', pretrained=True)
                raft_model.eval()
                raft_model.to(self.device)
                
                # Prepare images for RAFT
                img1_tensor = torch.from_numpy(img1_gray).float().unsqueeze(0).unsqueeze(0).to(self.device)
                img2_tensor = torch.from_numpy(img2_gray).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Repeat grayscale to RGB
                img1_tensor = img1_tensor.repeat(1, 3, 1, 1)
                img2_tensor = img2_tensor.repeat(1, 3, 1, 1)
                
                with torch.no_grad():
                    flow_predictions = raft_model(img1_tensor, img2_tensor)
                    flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()
                
                print("Using RAFT optical flow")
                return flow
                
            except Exception as e:
                print(f"Warning: Could not compute optical flow ({e}), using zero flow")
                # Fallback to zero flow
                H, W = img1_gray.shape
                return np.zeros((H, W, 2), dtype=np.float32)
    
    def _project_splats_to_2d(self, viewpoint_cam):
        """Project 3D splat positions to 2D image coordinates."""
        # Get 3D splat positions
        splat_3d = self.gaussians.get_xyz  # [N, 3]
        
        # Project to camera coordinates
        # Using the camera's world_view_transform
        world_view_transform = viewpoint_cam.world_view_transform.cuda()
        
        # Convert to homogeneous coordinates
        splat_3d_hom = torch.cat([splat_3d, torch.ones(splat_3d.shape[0], 1, device=self.device)], dim=1)
        
        # Transform to camera space
        splat_cam = torch.matmul(world_view_transform, splat_3d_hom.T).T[:, :3]
        
        # Project to image coordinates (simplified projection)
        # This is a rough approximation - real projection would use full camera model
        focal_length = 500.0  # Placeholder focal length
        principal_point = torch.tensor([viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2], device=self.device)
        
        # Perspective projection
        splat_2d = torch.zeros(splat_3d.shape[0], 2, device=self.device)
        mask = splat_cam[:, 2] > 0  # Only points in front of camera
        
        if mask.sum() > 0:
            splat_2d[mask, 0] = splat_cam[mask, 0] / splat_cam[mask, 2] * focal_length + principal_point[0]
            splat_2d[mask, 1] = splat_cam[mask, 1] / splat_cam[mask, 2] * focal_length + principal_point[1]
        
        # Clamp to image bounds
        splat_2d[:, 0] = torch.clamp(splat_2d[:, 0], 0, viewpoint_cam.image_width - 1)
        splat_2d[:, 1] = torch.clamp(splat_2d[:, 1], 0, viewpoint_cam.image_height - 1)
        
        return splat_2d
    
    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        gradient_norms = {}
        
        # Gaussian parameters
        if self.gaussians._xyz.grad is not None:
            gradient_norms['xyz'] = torch.norm(self.gaussians._xyz.grad).item()
        
        if self.gaussians._opacity.grad is not None:
            gradient_norms['opacity'] = torch.norm(self.gaussians._opacity.grad).item()
        
        # Deformation parameters
        if hasattr(self.gaussians, '_deformation_manager') and self.gaussians._deformation_manager is not None:
            if self.gaussians._deformation_manager.W.grad is not None:
                gradient_norms['deform_basis'] = torch.norm(self.gaussians._deformation_manager.W.grad).item()
        
        # Deformation field parameters
        deform_grad_norm = 0.0
        for param in self.deformation_field.parameters():
            if param.grad is not None:
                deform_grad_norm += torch.norm(param.grad).item() ** 2
        if deform_grad_norm > 0:
            gradient_norms['deform_field'] = np.sqrt(deform_grad_norm)
        
        # Rigid transform parameters
        if self.rigid_transforms.rotation_params.grad is not None:
            gradient_norms['rigid_rotation'] = torch.norm(self.rigid_transforms.rotation_params.grad).item()
        
        if self.rigid_transforms.translation_params.grad is not None:
            gradient_norms['rigid_translation'] = torch.norm(self.rigid_transforms.translation_params.grad).item()
        
        return gradient_norms
    
    def _update_cluster_assignments(self, viewpoint_cam):
        """Update cluster assignments with temporal consistency."""
        try:
            # Get current cluster assignments
            current_clusters = self.gaussians.get_cluster_ids()
            
            if current_clusters is None:
                return
            
            # Apply temporal consistency
            consistent_clusters = self.cluster_tracker.get_consistent_assignment(
                current_clusters,
                consistency_threshold=getattr(self.args, 'cluster_consistency_threshold', 0.7)
            )
            
            # Update assignments if they changed significantly
            changes = (current_clusters != consistent_clusters).sum().item()
            if changes > 0:
                print(f"Updated {changes} inconsistent cluster assignments")
                self.gaussians.set_cluster_assignments(consistent_clusters)
            
            # Update tracker
            self.cluster_tracker.update(consistent_clusters, timestamp=viewpoint_cam.time)
            
        except Exception as e:
            print(f"Cluster update error: {e}")
    
    def _camera_to_numpy(self, viewpoint_cam) -> np.ndarray:
        """Convert camera image to numpy array."""
        img = viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return img
    
    def training_step(self, viewpoint_cam, background):
        """Single training step with optimized mono4dgs pipeline."""
        # Add frame to buffer
        frame_data = self.frame_buffer.add_frame(viewpoint_cam)
        
        # Request foundation model processing (async)
        self._request_foundation_processing(frame_data)
        
        # Get depth and pose data
        depth_data = self._get_depth_data(frame_data)
        pose_data = self._get_pose_data()
        
        # Generate time coefficients
        time_tensor = torch.tensor([viewpoint_cam.time], device=self.device)
        time_coefficients = self.deformation_field(time_tensor).squeeze(0)  # [r]
        
        # Render with deformation
        render_output = render_with_depth_loss(
            viewpoint_cam,
            self.gaussians,
            self.args,  # Using args as pipe for simplicity
            background,
            depth_data['gt_depth'] if depth_data else None,
            depth_data['confidence'] if depth_data else None,
            time_coefficients=time_coefficients,
            rigid_transforms=self.rigid_transforms,
        )
        
        # Compute losses with current weights
        loss_info = self._compute_losses(
            render_output,
            viewpoint_cam,
            depth_data,
            pose_data,
            time_coefficients
        )
        
        # Update cluster assignments with temporal consistency
        if self.iteration % getattr(self.args, 'cluster_update_interval', 100) == 0:
            self._update_cluster_assignments(viewpoint_cam)
        
        # Backward pass
        total_loss = loss_info['total_loss']
        total_loss.backward()
        
        # Compute gradient norms for adaptive scheduling
        gradient_norms = self._compute_gradient_norms()
        
        # Update loss weights using adaptive scheduler
        updated_weights = self.loss_scheduler.update(
            iteration=self.iteration,
            loss_values=loss_info,
            gradient_norms=gradient_norms
        )
        
        # Apply updated weights to loss function
        self.loss_function.lambda_depth = updated_weights['depth_loss']
        self.loss_function.lambda_pose = updated_weights['pose_loss']
        self.loss_function.lambda_deform = updated_weights['deform_reg_loss']
        self.loss_function.lambda_rigid = updated_weights['rigid_reg_loss']
        
        # Optimizer steps
        self._optimizer_step()
        
        # Add scheduler info to loss_info
        loss_info.update({
            'weights': updated_weights,
            'scheduler_info': self.loss_scheduler.get_schedule_info()
        })
        
        return loss_info
    
    def _request_foundation_processing(self, frame_data):
        """Request processing from foundation models."""
        viewpoint = frame_data['viewpoint']
        frame_id = frame_data['frame_id']
        
        # Convert camera image to format expected by foundation models
        image = viewpoint.original_image.clone()
        
        # Request depth estimation
        if self.foundation_manager.depth_model is not None:
            self.foundation_manager.request_depth(frame_id, image)
        
        # Request pose estimation for frame pairs
        current_frame, prev_frame = self.frame_buffer.get_frame_pair()
        if current_frame and prev_frame and self.foundation_manager.pose_model is not None:
            frame_ids = [prev_frame['frame_id'], current_frame['frame_id']]
            images = [
                prev_frame['viewpoint'].original_image,
                current_frame['viewpoint'].original_image
            ]
            self.foundation_manager.request_pose(frame_ids, images)
    
    def _get_depth_data(self, frame_data):
        """Get depth data for current frame."""
        frame_id = frame_data['frame_id']
        gt_depth, confidence = self.foundation_manager.get_depth(frame_id)
        
        if gt_depth is not None:
            return {
                'gt_depth': gt_depth,
                'confidence': confidence,
            }
        
        # Fallback: use placeholder depth if foundation model not available
        H, W = frame_data['viewpoint'].image_height, frame_data['viewpoint'].image_width
        return {
            'gt_depth': torch.ones(H, W, device=self.device) * 5.0,  # 5m default depth
            'confidence': torch.ones(H, W, device=self.device) * 0.1,  # Low confidence
        }
    
    def _get_pose_data(self):
        """Get pose data for current frame pair."""
        current_frame, prev_frame = self.frame_buffer.get_frame_pair()
        
        if not current_frame or not prev_frame:
            return None
        
        frame_ids = [prev_frame['frame_id'], current_frame['frame_id']]
        poses, landmarks = self.foundation_manager.get_pose(frame_ids)
        
        if poses is not None and landmarks is not None:
            return {
                'poses': poses,
                'landmarks_3d': landmarks,
                'camera_intrinsics': self._get_camera_intrinsics(current_frame['viewpoint']),
            }
        
        return None
    
    def _get_camera_intrinsics(self, viewpoint):
        """Extract camera intrinsics matrix."""
        # This is a simplified extraction - adjust based on your camera model
        fx = fy = 500.0  # Placeholder focal length
        cx = viewpoint.image_width / 2
        cy = viewpoint.image_height / 2
        
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        return K
    
    def _compute_losses(self, render_output, viewpoint_cam, depth_data, pose_data, time_coefficients):
        """Compute all loss components."""
        # Standard photometric loss
        rendered_image = render_output["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        photometric_loss = l1_loss(rendered_image, gt_image[:3, :, :])
        
        # Prepare inputs for combined loss
        splat_positions = render_output.get("means3D_final", self.gaussians.get_xyz)
        
        # Camera pose (placeholder - extract from viewpoint_cam)
        camera_pose = torch.eye(4, device=self.device)
        
        # Depth loss inputs
        splat_depths = render_output.get("splat_depths")
        id_buffer = render_output.get("id_buffer")
        
        # Pose loss inputs
        landmarks_3d = pose_data['landmarks_3d'] if pose_data else None
        landmarks_2d = None  # Would need 2D correspondences
        camera_intrinsics = pose_data.get('camera_intrinsics') if pose_data else None
        
        # Deformation inputs
        basis_matrices = self.gaussians.get_deformation_basis()
        
        # Compute combined loss
        total_loss, loss_info = self.loss_function(
            rendered_image=rendered_image,
            gt_image=gt_image[:3, :, :],
            splat_positions=splat_positions,
            splat_depths=splat_depths,
            id_buffer=id_buffer,
            gt_depth=depth_data['gt_depth'] if depth_data else None,
            depth_confidence=depth_data['confidence'] if depth_data else None,
            camera_pose=camera_pose,
            landmarks_3d=landmarks_3d,
            landmarks_2d=landmarks_2d,
            camera_intrinsics=camera_intrinsics,
            basis_matrices=basis_matrices,
            time_coefficients=time_coefficients,
            rigid_transforms=self.rigid_transforms,
            photometric_loss=photometric_loss,
        )
        
        return loss_info
    
    def _optimizer_step(self):
        """Perform optimizer steps for all components."""
        # Standard Gaussian optimizer step
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        # Deformation field optimizer step
        self.deform_optimizer.step()
        self.deform_optimizer.zero_grad(set_to_none=True)
        
        # Rigid transform optimizer step
        self.rigid_optimizer.step()
        self.rigid_optimizer.zero_grad(set_to_none=True)
        
        # Separate fp16 deformation basis optimizer step
        if self.deform_basis_optimizer is not None:
            self.deform_basis_optimizer.step()
            self.deform_basis_optimizer.zero_grad(set_to_none=True)
    
    def train(self, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
        """Main training loop."""
        print("Starting Mono4DGS training...")
        
        # Setup models
        self.setup_models(dataset)
        
        # Setup training
        self.gaussians.training_setup(opt)
        
        # Background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Get training cameras
        train_cams = self.scene.getTrainCameras()
        
        # Initialize clusters
        self.initialize_clusters(train_cams)
        
        # Allocate reusable gradient buffer for depth loss
        if self.gaussians.get_xyz.shape[0] > 0:
            self.grad_mu_z_buffer = torch.zeros(
                self.gaussians.get_xyz.shape[0], 
                device=self.device, 
                dtype=torch.float32
            )
        
        # Training loop
        progress_bar = tqdm(range(opt.iterations), desc="Training progress")
        
        for iteration in range(opt.iterations):
            self.iteration = iteration
            
            # Update learning rates
            self.gaussians.update_learning_rate(iteration)
            
            # Select viewpoint
            viewpoint_cam = train_cams[randint(0, len(train_cams) - 1)]
            
            # Training step
            loss_info = self.training_step(viewpoint_cam, background)
            
            # Logging and checkpointing
            if iteration % 10 == 0:
                # Enhanced progress reporting
                scheduler_info = loss_info.get('scheduler_info', {})
                weights = loss_info.get('weights', {})
                
                progress_bar.set_postfix({
                    "Loss": f"{loss_info['total_loss']:.7f}",
                    "Depth": f"{loss_info.get('depth_loss', 0):.7f}",
                    "Pose": f"{loss_info.get('pose_loss', 0):.7f}",
                    "λ_d": f"{weights.get('depth_loss', 0):.4f}",
                    "λ_p": f"{weights.get('pose_loss', 0):.4f}",
                    "Conv": f"{len(scheduler_info.get('converged_losses', []))}"
                })
                progress_bar.update(10)
            
            # Print detailed statistics every 100 iterations
            if iteration % 100 == 0:
                self._print_detailed_stats(loss_info)
            
            # Save scheduler plots periodically
            if iteration % 1000 == 0 and iteration > 0:
                try:
                    plot_path = f"{self.scene.model_path}/loss_weights_iter_{iteration}.png"
                    self.loss_scheduler.plot_weight_evolution(save_path=plot_path)
                except Exception as e:
                    print(f"Failed to save scheduler plot: {e}")
            
            # Densification
            if iteration < opt.densify_until_iter:
                self._handle_densification(iteration, opt)
                
                # Reallocate gradient buffer if splat count changed
                if self.grad_mu_z_buffer.shape[0] != self.gaussians.get_xyz.shape[0]:
                    self.grad_mu_z_buffer = torch.zeros(
                        self.gaussians.get_xyz.shape[0], 
                        device=self.device, 
                        dtype=torch.float32
                    )
            
            # Testing and saving
            if iteration in testing_iterations:
                self._run_testing(iteration)
            
            if iteration in saving_iterations:
                self._save_checkpoint(iteration)
        
        progress_bar.close()
        print("Training complete!")
    
    def _handle_densification(self, iteration, opt):
        """Handle Gaussian densification."""
        # Standard 4DGS densification logic
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            
            self.gaussians.densify(
                opt.densify_grad_threshold,
                opt.opacity_threshold,
                self.scene.cameras_extent,
                size_threshold
            )
        
        if iteration % opt.opacity_reset_interval == 0:
            self.gaussians.reset_opacity()
    
    def _run_testing(self, iteration):
        """Run testing on validation set."""
        # TODO: Implement testing logic
        pass
    
    def _save_checkpoint(self, iteration):
        """Save model checkpoint."""
        print(f"\n[ITER {iteration}] Saving Checkpoint")
        
        # Save Gaussian model
        torch.save(
            (self.gaussians.capture(), iteration),
            self.scene.model_path + f"/chkpnt_mono4dgs_{iteration}.pth"
        )
        
        # Save additional components
        checkpoint_data = {
            'deformation_field': self.deformation_field.state_dict(),
            'rigid_transforms': {
                'rotation_params': self.rigid_transforms.rotation_params,
                'translation_params': self.rigid_transforms.translation_params,
            },
            'optimizers': {
                'deform': self.deform_optimizer.state_dict(),
                'rigid': self.rigid_optimizer.state_dict(),
            },
            'iteration': iteration,
        }
        
        torch.save(
            checkpoint_data,
            self.scene.model_path + f"/mono4dgs_state_{iteration}.pth"
        )
    
    def cleanup(self):
        """Cleanup resources."""
        self.foundation_manager.cleanup()


def main():
    """Main entry point."""
    parser = ArgumentParser(description="Mono4DGS Training")
    
    # Add standard 4DGS arguments
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    # Add mono4dgs specific arguments
    parser.add_argument('--enable_mono4dgs', action='store_true', default=True)
    parser.add_argument('--deform_rank', type=int, default=4)
    parser.add_argument('--num_rigid_clusters', type=int, default=8)
    parser.add_argument('--deform_lr', type=float, default=1e-3)
    parser.add_argument('--rigid_lr', type=float, default=1e-3)
    parser.add_argument('--lambda_depth', type=float, default=1.0)
    parser.add_argument('--lambda_pose', type=float, default=0.1)
    parser.add_argument('--lambda_deform', type=float, default=1e-4)
    parser.add_argument('--lambda_rigid', type=float, default=1e-3)
    
    # Standard arguments
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[14000, 20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="mono4dgs")
    parser.add_argument("--configs", type=str, default="")
    
    args = parser.parse_args()
    
    # Process arguments
    args.save_iterations.append(args.iterations)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print("Optimizing " + args.model_path)
    
    # Initialize system state
    safe_state(args.quiet)
    
    # Start GUI server
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Create trainer and run
    trainer = Mono4DGSTrainer(args)
    
    try:
        trainer.train(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from
        )
    finally:
        trainer.cleanup()
    
    print("\nMono4DGS Training complete.")


if __name__ == "__main__":
    main()