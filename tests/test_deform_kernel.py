# tests/test_deform_kernel.py

import torch
import pytest
import numpy as np
import time
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.triton_deform import apply_deformation, compute_depth_loss


class TestDeformationKernel:
    """Test suite for deformation kernel functionality."""
    
    @pytest.fixture
    def setup_device(self):
        """Ensure CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.cuda.empty_cache()
    
    @pytest.fixture
    def random_deformation_data(self):
        """Generate random data for deformation testing."""
        N = 1000  # Number of splats
        r = 4     # Deformation rank
        
        # Random splat centroids
        mu = torch.randn(N, 3, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Random deformation basis matrices
        W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16, requires_grad=True)
        
        # Random time coefficients
        b = torch.randn(r, device='cuda', dtype=torch.float16, requires_grad=True)
        
        return mu, W, b
    
    @pytest.fixture
    def large_deformation_data(self):
        """Generate large-scale data for performance testing."""
        N = 100000  # 100k splats
        r = 4
        
        mu = torch.randn(N, 3, device='cuda', dtype=torch.float32, requires_grad=True)
        W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16, requires_grad=True)
        b = torch.randn(r, device='cuda', dtype=torch.float16, requires_grad=True)
        
        return mu, W, b
    
    def test_deformation_forward_shape(self, setup_device, random_deformation_data):
        """Test that deformation produces correct output shapes."""
        mu, W, b = random_deformation_data
        
        # Apply deformation
        mu_deformed = apply_deformation(mu, W, b)
        
        # Check output shape and type
        assert mu_deformed.shape == mu.shape
        assert mu_deformed.dtype == torch.float32
        assert mu_deformed.device == mu.device
        assert mu_deformed.requires_grad
    
    def test_deformation_mathematical_correctness(self, setup_device, random_deformation_data):
        """Test mathematical correctness of deformation operation."""
        mu, W, b = random_deformation_data
        N, _, r = W.shape
        
        # Apply deformation using Triton kernel
        mu_deformed_triton = apply_deformation(mu, W, b)
        
        # Compute reference using standard PyTorch operations
        # μ_new = μ + W @ b
        W_float = W.float()  # Convert to fp32 for computation
        b_float = b.float()
        
        # Manually compute W @ b for each splat
        mu_deformed_ref = mu.clone()
        for i in range(N):
            delta = torch.matmul(W_float[i], b_float)  # [3, r] @ [r] -> [3]
            mu_deformed_ref[i] += delta
        
        # Check numerical accuracy (allow for fp16 precision differences)
        max_error = torch.max(torch.abs(mu_deformed_triton - mu_deformed_ref))
        print(f"Maximum error: {max_error.item()}")
        
        # Allow for reasonable fp16 precision loss
        assert max_error < 1e-2, f"Error too large: {max_error}"
        
        # Check mean relative error
        rel_error = torch.mean(torch.abs(mu_deformed_triton - mu_deformed_ref) / 
                              (torch.abs(mu_deformed_ref) + 1e-6))
        print(f"Mean relative error: {rel_error.item()}")
        assert rel_error < 1e-3
    
    def test_deformation_backward_pass(self, setup_device, random_deformation_data):
        """Test backward pass and gradient computation."""
        mu, W, b = random_deformation_data
        
        # Apply deformation
        mu_deformed = apply_deformation(mu, W, b)
        
        # Create dummy loss
        loss = torch.sum(mu_deformed ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and have correct shapes
        assert mu.grad is not None
        assert mu.grad.shape == mu.shape
        assert mu.grad.dtype == torch.float32
        
        assert W.grad is not None
        assert W.grad.shape == W.shape
        assert W.grad.dtype == torch.float16
        
        assert b.grad is not None
        assert b.grad.shape == b.shape
        assert b.grad.dtype == torch.float16
        
        # Check gradient magnitudes are reasonable
        assert torch.all(torch.isfinite(mu.grad))
        assert torch.all(torch.isfinite(W.grad))
        assert torch.all(torch.isfinite(b.grad))
    
    def test_deformation_gradient_correctness(self, setup_device, random_deformation_data):
        """Test gradient correctness using finite differences."""
        mu, W, b = random_deformation_data
        
        # Select a small subset for finite difference testing
        subset_size = 10
        mu_small = mu[:subset_size].clone().requires_grad_(True)
        W_small = W[:subset_size].clone().requires_grad_(True)
        b_small = b.clone().requires_grad_(True)
        
        def loss_fn(mu_in, W_in, b_in):
            mu_def = apply_deformation(mu_in, W_in, b_in)
            return torch.sum(mu_def ** 2)
        
        # Compute analytical gradients
        loss = loss_fn(mu_small, W_small, b_small)
        loss.backward()
        
        grad_mu_analytical = mu_small.grad.clone()
        grad_W_analytical = W_small.grad.clone()
        grad_b_analytical = b_small.grad.clone()
        
        # Finite difference check for mu
        eps = 1e-4
        grad_mu_fd = torch.zeros_like(mu_small)
        
        for i in range(subset_size):
            for j in range(3):
                mu_small_copy = mu_small.detach().clone()
                mu_small_copy[i, j] += eps
                loss_plus = loss_fn(mu_small_copy, W_small.detach(), b_small.detach())
                
                mu_small_copy = mu_small.detach().clone()
                mu_small_copy[i, j] -= eps
                loss_minus = loss_fn(mu_small_copy, W_small.detach(), b_small.detach())
                
                grad_mu_fd[i, j] = (loss_plus - loss_minus) / (2 * eps)
        
        # Check gradient accuracy
        mu_error = torch.max(torch.abs(grad_mu_analytical - grad_mu_fd))
        print(f"μ gradient error: {mu_error.item()}")
        assert mu_error < 1e-2
    
    def test_deformation_performance(self, setup_device, large_deformation_data):
        """Test performance of deformation kernel."""
        mu, W, b = large_deformation_data
        N = mu.shape[0]
        
        # Warm up
        for _ in range(5):
            _ = apply_deformation(mu, W, b)
        
        torch.cuda.synchronize()
        
        # Time the kernel
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            mu_deformed = apply_deformation(mu, W, b)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = N / avg_time / 1e6  # Million splats per second
        
        print(f"Average time: {avg_time*1000:.3f} ms")
        print(f"Throughput: {throughput:.2f} M splats/second")
        
        # Performance target: should handle 100k splats in < 1ms
        assert avg_time < 0.001, f"Too slow: {avg_time*1000:.3f} ms"
    
    def test_deformation_memory_efficiency(self, setup_device, large_deformation_data):
        """Test memory usage of deformation kernel."""
        mu, W, b = large_deformation_data
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Apply deformation
        mu_deformed = apply_deformation(mu, W, b)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # Calculate expected memory usage
        # Input: mu (N*3*4 bytes) + W (N*3*r*2 bytes) + b (r*2 bytes)
        # Output: mu_deformed (N*3*4 bytes)
        N, _, r = W.shape
        expected_memory = N * 3 * 4 + N * 3 * r * 2 + r * 2 + N * 3 * 4
        
        print(f"Memory used: {memory_used / 1e6:.2f} MB")
        print(f"Expected: {expected_memory / 1e6:.2f} MB")
        
        # Allow some overhead but shouldn't be excessive
        assert memory_used < expected_memory * 2
    
    def test_edge_cases(self, setup_device):
        """Test edge cases and error conditions."""
        # Test with rank 1
        N, r = 100, 1
        mu = torch.randn(N, 3, device='cuda', dtype=torch.float32, requires_grad=True)
        W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16, requires_grad=True)
        b = torch.randn(r, device='cuda', dtype=torch.float16, requires_grad=True)
        
        mu_deformed = apply_deformation(mu, W, b)
        assert mu_deformed.shape == mu.shape
        
        # Test with larger rank
        r = 16
        W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16, requires_grad=True)
        b = torch.randn(r, device='cuda', dtype=torch.float16, requires_grad=True)
        
        mu_deformed = apply_deformation(mu, W, b)
        assert mu_deformed.shape == mu.shape
        
        # Test with single splat
        N = 1
        mu = torch.randn(N, 3, device='cuda', dtype=torch.float32, requires_grad=True)
        W = torch.randn(N, 3, 4, device='cuda', dtype=torch.float16, requires_grad=True)
        b = torch.randn(4, device='cuda', dtype=torch.float16, requires_grad=True)
        
        mu_deformed = apply_deformation(mu, W, b)
        assert mu_deformed.shape == mu.shape


class TestUtilityFunctions:
    """Test utility functions and integration."""
    
    def test_kernel_compilation(self):
        """Test that kernels compile without errors."""
        from kernels.triton_deform import compile_kernels
        
        # This should not raise any errors
        compile_kernels()
    
    def test_mixed_precision_handling(self):
        """Test handling of mixed precision inputs."""
        N, r = 100, 4
        
        # Test different input precisions
        mu_fp32 = torch.randn(N, 3, device='cuda', dtype=torch.float32, requires_grad=True)
        W_fp16 = torch.randn(N, 3, r, device='cuda', dtype=torch.float16, requires_grad=True)
        b_fp16 = torch.randn(r, device='cuda', dtype=torch.float16, requires_grad=True)
        
        # Should work with mixed precision
        result = apply_deformation(mu_fp32, W_fp16, b_fp16)
        assert result.dtype == torch.float32
        
        # Test backward pass
        loss = torch.sum(result ** 2)
        loss.backward()
        
        assert mu_fp32.grad.dtype == torch.float32
        assert W_fp16.grad.dtype == torch.float16
        assert b_fp16.grad.dtype == torch.float16


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    print("\n" + "="*50)
    print("DEFORMATION KERNEL PERFORMANCE BENCHMARK")
    print("="*50)
    
    test_configs = [
        (1000, 4),      # Small scale
        (10000, 4),     # Medium scale
        (100000, 4),    # Large scale
        (1000000, 4),   # Very large scale
        (100000, 8),    # Higher rank
        (100000, 16),   # Even higher rank
    ]
    
    for N, r in test_configs:
        print(f"\nTesting N={N:,} splats, rank={r}")
        
        try:
            # Generate data
            mu = torch.randn(N, 3, device='cuda', dtype=torch.float32)
            W = torch.randn(N, 3, r, device='cuda', dtype=torch.float16)
            b = torch.randn(r, device='cuda', dtype=torch.float16)
            
            # Warm up
            for _ in range(3):
                _ = apply_deformation(mu, W, b)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                result = apply_deformation(mu, W, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            throughput = N / avg_time / 1e6
            
            # Memory usage
            memory_mb = (N * (3 * 4 + 3 * r * 2) + r * 2) / 1e6
            
            print(f"  Time: {avg_time*1000:.3f} ms")
            print(f"  Throughput: {throughput:.2f} M splats/s")
            print(f"  Memory: {memory_mb:.2f} MB")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run benchmark
    run_performance_benchmark()