# tests/test_depth_loss.py

import torch
import pytest
import numpy as np
import time
from typing import Tuple, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.triton_deform import compute_depth_loss
from loss_depth_pose import DepthReprojectionLoss, PoseConsistencyLoss, Mono4DGSLoss


class TestDepthLoss:
    """Test suite for depth loss computation."""
    
    @pytest.fixture
    def setup_device(self):
        """Ensure CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.cuda.empty_cache()
    
    @pytest.fixture
    def random_depth_data(self):
        """Generate random data for depth testing."""
        N = 1000      # Number of splats
        H, W = 256, 256  # Image dimensions
        
        # Random splat depths
        splat_depths = torch.randn(N, device='cuda', dtype=torch.float32, requires_grad=True)
        splat_depths = torch.abs(splat_depths) + 0.1  # Ensure positive depths
        
        # Random ID buffer (some pixels map to splats, others are background)
        id_buffer = torch.randint(-1, N, (H, W), device='cuda', dtype=torch.int32)
        # Set some pixels to background (-1)
        background_mask = torch.rand(H, W, device='cuda') < 0.3
        id_buffer[background_mask] = -1
        
        # Random ground truth depth
        gt_depth = torch.rand(H, W, device='cuda', dtype=torch.float32) * 10 + 0.1
        
        # Random confidence weights
        confidence = torch.rand(H, W, device='cuda', dtype=torch.float32)
        
        return splat_depths, id_buffer, gt_depth, confidence
    
    @pytest.fixture
    def large_depth_data(self):
        """Generate large-scale data for performance testing."""
        N = 100000
        H, W = 1080, 1920  # Full HD
        
        splat_depths = torch.abs(torch.randn(N, device='cuda', dtype=torch.float32)) + 0.1
        splat_depths.requires_grad_(True)
        
        id_buffer = torch.randint(-1, N, (H, W), device='cuda', dtype=torch.int32)
        background_mask = torch.rand(H, W, device='cuda') < 0.4
        id_buffer[background_mask] = -1
        
        gt_depth = torch.rand(H, W, device='cuda', dtype=torch.float32) * 20 + 0.1
        confidence = torch.rand(H, W, device='cuda', dtype=torch.float32)
        
        return splat_depths, id_buffer, gt_depth, confidence
    
    def test_depth_loss_forward_shape(self, setup_device, random_depth_data):
        """Test that depth loss produces correct output shapes."""
        splat_depths, id_buffer, gt_depth, confidence = random_depth_data
        
        # Compute depth loss
        loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        
        # Check output shape and type
        assert loss.shape == torch.Size([])  # Scalar loss
        assert loss.dtype == torch.float32
        assert loss.device.type == 'cuda'
        assert loss.requires_grad
    
    def test_depth_loss_mathematical_correctness(self, setup_device, random_depth_data):
        """Test mathematical correctness of depth loss computation."""
        splat_depths, id_buffer, gt_depth, confidence = random_depth_data
        
        # Compute loss using Triton kernel
        loss_triton = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        
        # Compute reference using standard PyTorch operations
        H, W = gt_depth.shape
        valid_mask = (id_buffer >= 0) & (confidence > 0)
        
        if valid_mask.sum() > 0:
            valid_ids = id_buffer[valid_mask]
            valid_gt = gt_depth[valid_mask]
            valid_conf = confidence[valid_mask]
            valid_pred = splat_depths[valid_ids]
            
            # L1 loss with confidence weighting
            residuals = torch.abs(valid_pred - valid_gt)
            weighted_loss = valid_conf * residuals
            loss_ref = weighted_loss.mean()
        else:
            loss_ref = torch.tensor(0.0, device=splat_depths.device)
        
        # Check numerical accuracy
        error = torch.abs(loss_triton - loss_ref)
        print(f"Depth loss error: {error.item()}")
        
        # Allow for reasonable numerical differences
        assert error < 1e-5, f"Error too large: {error}"
    
    def test_depth_loss_backward_pass(self, setup_device, random_depth_data):
        """Test backward pass and gradient computation."""
        splat_depths, id_buffer, gt_depth, confidence = random_depth_data
        
        # Compute depth loss
        loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and have correct shapes
        assert splat_depths.grad is not None
        assert splat_depths.grad.shape == splat_depths.shape
        assert splat_depths.grad.dtype == torch.float32
        
        # Check gradient magnitudes are reasonable
        assert torch.all(torch.isfinite(splat_depths.grad))
        
        # Gradients should be zero for splats not visible in any pixel
        visible_splats = torch.unique(id_buffer[id_buffer >= 0])
        invisible_mask = torch.ones(splat_depths.shape[0], dtype=torch.bool, device='cuda')
        invisible_mask[visible_splats] = False
        
        if invisible_mask.sum() > 0:
            assert torch.allclose(splat_depths.grad[invisible_mask], torch.zeros_like(splat_depths.grad[invisible_mask]))
    
    def test_depth_loss_gradient_correctness(self, setup_device):
        """Test gradient correctness using finite differences."""
        # Use smaller data for finite difference testing
        N = 50
        H, W = 32, 32
        
        splat_depths = torch.abs(torch.randn(N, device='cuda', dtype=torch.float32)) + 0.1
        splat_depths.requires_grad_(True)
        
        id_buffer = torch.randint(0, N, (H, W), device='cuda', dtype=torch.int32)
        gt_depth = torch.rand(H, W, device='cuda', dtype=torch.float32) * 5 + 0.1
        confidence = torch.ones(H, W, device='cuda', dtype=torch.float32)
        
        # Compute analytical gradients
        loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        loss.backward()
        grad_analytical = splat_depths.grad.clone()
        
        # Compute finite difference gradients
        eps = 1e-4
        grad_fd = torch.zeros_like(splat_depths)
        
        for i in range(N):
            splat_depths_copy = splat_depths.detach().clone()
            splat_depths_copy[i] += eps
            loss_plus = compute_depth_loss(splat_depths_copy, id_buffer, gt_depth, confidence)
            
            splat_depths_copy = splat_depths.detach().clone()
            splat_depths_copy[i] -= eps
            loss_minus = compute_depth_loss(splat_depths_copy, id_buffer, gt_depth, confidence)
            
            grad_fd[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Check gradient accuracy
        error = torch.max(torch.abs(grad_analytical - grad_fd))
        print(f"Gradient error: {error.item()}")
        
        # Allow for reasonable finite difference error
        assert error < 1e-2, f"Gradient error too large: {error}"
    
    def test_depth_loss_performance(self, setup_device, large_depth_data):
        """Test performance of depth loss computation."""
        splat_depths, id_buffer, gt_depth, confidence = large_depth_data
        
        # Warm up
        for _ in range(5):
            _ = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        
        torch.cuda.synchronize()
        
        # Time the kernel
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        H, W = gt_depth.shape
        pixel_throughput = H * W / avg_time / 1e6  # Million pixels per second
        
        print(f"Average time: {avg_time*1000:.3f} ms")
        print(f"Pixel throughput: {pixel_throughput:.2f} M pixels/second")
        
        # Performance target: should handle 1080p in < 2ms
        assert avg_time < 0.002, f"Too slow: {avg_time*1000:.3f} ms"
    
    def test_edge_cases(self, setup_device):
        """Test edge cases and boundary conditions."""
        N = 100
        H, W = 64, 64
        
        # Test with all background pixels
        splat_depths = torch.ones(N, device='cuda', dtype=torch.float32, requires_grad=True)
        id_buffer = torch.full((H, W), -1, device='cuda', dtype=torch.int32)
        gt_depth = torch.ones(H, W, device='cuda', dtype=torch.float32)
        confidence = torch.ones(H, W, device='cuda', dtype=torch.float32)
        
        loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        assert loss.item() == 0.0
        
        loss.backward()
        assert torch.allclose(splat_depths.grad, torch.zeros_like(splat_depths.grad))
        
        # Test with zero confidence
        id_buffer = torch.zeros((H, W), device='cuda', dtype=torch.int32)
        confidence = torch.zeros(H, W, device='cuda', dtype=torch.float32)
        
        splat_depths.grad.zero_()
        loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        assert loss.item() == 0.0
        
        loss.backward()
        assert torch.allclose(splat_depths.grad, torch.zeros_like(splat_depths.grad))
        
        # Test with single pixel
        H, W = 1, 1
        id_buffer = torch.zeros((H, W), device='cuda', dtype=torch.int32)
        gt_depth = torch.tensor([[5.0]], device='cuda', dtype=torch.float32)
        confidence = torch.tensor([[1.0]], device='cuda', dtype=torch.float32)
        splat_depths = torch.tensor([3.0], device='cuda', dtype=torch.float32, requires_grad=True)
        
        loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
        expected_loss = 2.0  # |3.0 - 5.0| = 2.0
        assert torch.allclose(loss, torch.tensor(expected_loss, device='cuda'))


class TestDepthReprojectionLoss:
    """Test the high-level DepthReprojectionLoss class."""
    
    @pytest.fixture
    def setup_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.cuda.empty_cache()
    
    def test_depth_reprojection_loss_forward(self, setup_device):
        """Test forward pass of DepthReprojectionLoss."""
        loss_fn = DepthReprojectionLoss()
        
        N = 100
        H, W = 64, 64
        
        splat_depths = torch.abs(torch.randn(N, device='cuda', dtype=torch.float32)) + 0.1
        id_buffer = torch.randint(-1, N, (H, W), device='cuda', dtype=torch.int32)
        gt_depth = torch.rand(H, W, device='cuda', dtype=torch.float32) * 10 + 0.1
        confidence = torch.rand(H, W, device='cuda', dtype=torch.float32)
        
        loss, info = loss_fn(splat_depths, id_buffer, gt_depth, confidence)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert isinstance(info, dict)
        assert 'num_valid' in info
        assert 'mean_confidence' in info
    
    def test_depth_reprojection_loss_with_confidence_threshold(self, setup_device):
        """Test confidence threshold filtering."""
        confidence_threshold = 0.5
        loss_fn = DepthReprojectionLoss(confidence_threshold=confidence_threshold)
        
        N = 100
        H, W = 64, 64
        
        splat_depths = torch.ones(N, device='cuda', dtype=torch.float32)
        id_buffer = torch.zeros((H, W), device='cuda', dtype=torch.int32)
        gt_depth = torch.ones(H, W, device='cuda', dtype=torch.float32) * 2.0
        
        # Low confidence - should be filtered out
        confidence_low = torch.ones(H, W, device='cuda', dtype=torch.float32) * 0.1
        loss_low, info_low = loss_fn(splat_depths, id_buffer, gt_depth, confidence_low)
        
        # High confidence - should be included
        confidence_high = torch.ones(H, W, device='cuda', dtype=torch.float32) * 0.8
        loss_high, info_high = loss_fn(splat_depths, id_buffer, gt_depth, confidence_high)
        
        assert info_low['num_valid'] == 0
        assert info_high['num_valid'] > 0
        assert loss_low.item() == 0.0
        assert loss_high.item() > 0.0


class TestCombinedLoss:
    """Test the combined Mono4DGSLoss function."""
    
    @pytest.fixture
    def setup_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.cuda.empty_cache()
    
    def test_mono4dgs_loss_integration(self, setup_device):
        """Test the integrated loss function."""
        loss_fn = Mono4DGSLoss(
            lambda_depth=1.0,
            lambda_pose=0.1,
            lambda_deform=1e-4,
            lambda_rigid=1e-3,
        )
        
        # Create dummy data
        C, H, W = 3, 64, 64
        N = 100
        
        rendered_image = torch.rand(C, H, W, device='cuda', dtype=torch.float32)
        gt_image = torch.rand(C, H, W, device='cuda', dtype=torch.float32)
        splat_positions = torch.rand(N, 3, device='cuda', dtype=torch.float32)
        splat_depths = torch.abs(torch.randn(N, device='cuda', dtype=torch.float32)) + 0.1
        id_buffer = torch.randint(-1, N, (H, W), device='cuda', dtype=torch.int32)
        gt_depth = torch.rand(H, W, device='cuda', dtype=torch.float32) * 10 + 0.1
        depth_confidence = torch.rand(H, W, device='cuda', dtype=torch.float32)
        camera_pose = torch.eye(4, device='cuda', dtype=torch.float32)
        
        # Create dummy deformation data
        r = 4
        basis_matrices = torch.randn(N, 3, r, device='cuda', dtype=torch.float16)
        time_coefficients = torch.randn(r, device='cuda', dtype=torch.float16)
        
        # Compute combined loss
        total_loss, loss_info = loss_fn(
            rendered_image=rendered_image,
            gt_image=gt_image,
            splat_positions=splat_positions,
            splat_depths=splat_depths,
            id_buffer=id_buffer,
            gt_depth=gt_depth,
            depth_confidence=depth_confidence,
            camera_pose=camera_pose,
            basis_matrices=basis_matrices,
            time_coefficients=time_coefficients,
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == torch.Size([])
        assert isinstance(loss_info, dict)
        assert 'total_loss' in loss_info
        assert 'photometric_loss' in loss_info
        assert 'depth_loss' in loss_info
    
    def test_loss_weight_scheduling(self, setup_device):
        """Test loss weight scheduling functionality."""
        loss_fn = Mono4DGSLoss()
        
        # Test weight updates
        initial_depth_weight = loss_fn.lambda_depth
        
        schedule = {
            'depth': {0: 0.5, 100: 1.0, 200: 2.0},
        }
        
        loss_fn.update_weights(0, schedule)
        assert loss_fn.lambda_depth == 0.5
        
        loss_fn.update_weights(100, schedule)
        assert loss_fn.lambda_depth == 1.0
        
        loss_fn.update_weights(150, schedule)  # Should keep previous value
        assert loss_fn.lambda_depth == 1.0
        
        loss_fn.update_weights(200, schedule)
        assert loss_fn.lambda_depth == 2.0


def run_depth_loss_benchmark():
    """Run comprehensive depth loss benchmark."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    print("\n" + "="*50)
    print("DEPTH LOSS PERFORMANCE BENCHMARK")
    print("="*50)
    
    test_configs = [
        (1000, 256, 256),      # Small scale
        (10000, 512, 512),     # Medium scale
        (100000, 1024, 1024),  # Large scale
        (100000, 1080, 1920),  # Full HD
        (1000000, 2160, 3840), # 4K (if memory allows)
    ]
    
    for N, H, W in test_configs:
        print(f"\nTesting N={N:,} splats, {H}x{W} image")
        
        try:
            # Generate data
            splat_depths = torch.abs(torch.randn(N, device='cuda', dtype=torch.float32)) + 0.1
            splat_depths.requires_grad_(True)
            
            id_buffer = torch.randint(-1, N, (H, W), device='cuda', dtype=torch.int32)
            background_mask = torch.rand(H, W, device='cuda') < 0.3
            id_buffer[background_mask] = -1
            
            gt_depth = torch.rand(H, W, device='cuda', dtype=torch.float32) * 10 + 0.1
            confidence = torch.rand(H, W, device='cuda', dtype=torch.float32)
            
            # Warm up
            for _ in range(3):
                loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
                loss.backward()
                splat_depths.grad.zero_()
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                loss = compute_depth_loss(splat_depths, id_buffer, gt_depth, confidence)
                loss.backward()
                splat_depths.grad.zero_()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            pixel_throughput = H * W / avg_time / 1e6
            
            # Memory usage
            memory_mb = (N * 4 + H * W * (4 + 4 + 4)) / 1e6
            
            print(f"  Time: {avg_time*1000:.3f} ms")
            print(f"  Throughput: {pixel_throughput:.2f} M pixels/s")
            print(f"  Memory: {memory_mb:.2f} MB")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run benchmark
    run_depth_loss_benchmark()