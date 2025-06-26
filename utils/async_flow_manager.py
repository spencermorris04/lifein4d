# utils/async_flow_manager.py

import torch
import torch.nn as nn
import numpy as np
import threading
import queue
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any
import cv2


class AsyncOpticalFlowManager:
    """
    Asynchronous optical flow computation manager.
    
    Handles background optical flow computation with model caching
    and temporal consistency enforcement.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        max_queue_size: int = 10,
        cache_size: int = 50,
        use_raft: bool = True,
        flow_method: str = 'auto'
    ):
        """
        Initialize async flow manager.
        
        Args:
            device: Computation device
            max_queue_size: Maximum pending requests
            cache_size: Maximum cached flow results
            use_raft: Whether to use RAFT model
            flow_method: 'opencv', 'raft', or 'auto'
        """
        self.device = device
        self.cache_size = cache_size
        self.flow_method = flow_method
        
        # Threading components
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        
        # Caching
        self.flow_cache = {}
        self.cache_order = deque(maxlen=cache_size)
        self.cache_lock = threading.Lock()
        
        # Models (cached)
        self.raft_model = None
        self.model_loaded = False
        
        # Temporal consistency
        self.temporal_window = 5
        self.flow_history = deque(maxlen=self.temporal_window)
        
        # Statistics
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'compute_time_ms': [],
            'queue_size': []
        }
        
        self._initialize_models()
        self._start_worker()
    
    def _initialize_models(self):
        """Initialize optical flow models."""
        if self.flow_method in ['raft', 'auto']:
            try:
                print("Loading RAFT model...")
                # Load once and cache
                self.raft_model = torch.hub.load(
                    'pytorch/vision:v0.10.0', 'raft_small', 
                    pretrained=True, progress=True
                )
                self.raft_model.eval()
                self.raft_model.to(self.device)
                
                # Warm up the model
                dummy_img = torch.randn(1, 3, 256, 256, device=self.device)
                with torch.no_grad():
                    _ = self.raft_model(dummy_img, dummy_img)
                
                self.model_loaded = True
                print("RAFT model loaded and warmed up")
                
            except Exception as e:
                print(f"Failed to load RAFT model: {e}")
                if self.flow_method == 'raft':
                    raise
                else:
                    print("Falling back to OpenCV")
                    self.flow_method = 'opencv'
    
    def _start_worker(self):
        """Start background worker thread."""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _worker_loop(self):
        """Main worker loop for processing flow requests."""
        while not self.shutdown_event.is_set():
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=1.0)
                if request is None:  # Shutdown signal
                    break
                
                frame_pair_id, img1, img2, timestamp = request
                
                # Check cache first
                with self.cache_lock:
                    if frame_pair_id in self.flow_cache:
                        flow = self.flow_cache[frame_pair_id]
                        self.stats['cache_hits'] += 1
                        self.result_queue.put(('cache_hit', frame_pair_id, flow, timestamp))
                        continue
                
                # Compute flow
                start_time = time.time()
                flow = self._compute_flow(img1, img2)
                compute_time = (time.time() - start_time) * 1000
                
                # Apply temporal consistency
                flow = self._apply_temporal_consistency(flow, timestamp)
                
                # Cache result
                with self.cache_lock:
                    self._cache_flow(frame_pair_id, flow)
                
                # Update statistics
                self.stats['compute_time_ms'].append(compute_time)
                if len(self.stats['compute_time_ms']) > 100:
                    self.stats['compute_time_ms'].pop(0)
                
                # Send result
                self.result_queue.put(('computed', frame_pair_id, flow, timestamp))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Flow worker error: {e}")
                continue
    
    def _compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two images."""
        if self.flow_method == 'opencv':
            return self._compute_opencv_flow(img1, img2)
        elif self.flow_method == 'raft' and self.model_loaded:
            return self._compute_raft_flow(img1, img2)
        else:
            # Fallback to OpenCV
            return self._compute_opencv_flow(img1, img2)
    
    def _compute_opencv_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Compute flow using OpenCV Farneback method."""
        try:
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                img1_gray = img1.astype(np.uint8)
                img2_gray = img2.astype(np.uint8)
            
            # Compute flow with optimized parameters
            flow = cv2.calcOpticalFlowFarneback(
                img1_gray, img2_gray, None,
                pyr_scale=0.5,      # Pyramid scale
                levels=3,           # Pyramid levels
                winsize=15,         # Window size
                iterations=3,       # Iterations per level
                poly_n=5,          # Polynomial expansion neighborhood
                poly_sigma=1.2,    # Gaussian standard deviation
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
            
            return flow
            
        except Exception as e:
            print(f"OpenCV flow computation failed: {e}")
            # Return zero flow as fallback
            H, W = img1.shape[:2]
            return np.zeros((H, W, 2), dtype=np.float32)
    
    def _compute_raft_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Compute flow using RAFT model."""
        try:
            # Convert numpy to tensor
            def preprocess_image(img):
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                
                # Ensure RGB format
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                
                # Convert to tensor [1, 3, H, W]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                return img_tensor.to(self.device)
            
            img1_tensor = preprocess_image(img1)
            img2_tensor = preprocess_image(img2)
            
            # Compute flow
            with torch.no_grad():
                flow_predictions = self.raft_model(img1_tensor, img2_tensor)
                flow_tensor = flow_predictions[-1][0]  # Take final prediction
                flow = flow_tensor.permute(1, 2, 0).cpu().numpy()
            
            return flow
            
        except Exception as e:
            print(f"RAFT flow computation failed: {e}")
            # Fallback to OpenCV
            return self._compute_opencv_flow(img1, img2)
    
    def _apply_temporal_consistency(self, flow: np.ndarray, timestamp: float) -> np.ndarray:
        """Apply temporal consistency to flow field."""
        # Store in history
        self.flow_history.append((flow.copy(), timestamp))
        
        if len(self.flow_history) < 2:
            return flow
        
        # Temporal smoothing with exponential decay
        weights = np.exp(-np.arange(len(self.flow_history))[::-1] * 0.5)
        weights = weights / weights.sum()
        
        # Weighted average of recent flows
        smoothed_flow = np.zeros_like(flow)
        for i, (hist_flow, hist_time) in enumerate(self.flow_history):
            # Time-based weight adjustment
            time_weight = np.exp(-abs(timestamp - hist_time) * 2.0)
            smoothed_flow += weights[i] * time_weight * hist_flow
        
        # Blend with current flow
        alpha = 0.7  # Current flow weight
        final_flow = alpha * flow + (1 - alpha) * smoothed_flow
        
        return final_flow
    
    def _cache_flow(self, frame_pair_id: str, flow: np.ndarray):
        """Cache computed flow with LRU policy."""
        # Remove oldest if cache full
        if len(self.flow_cache) >= self.cache_size:
            oldest_id = self.cache_order.popleft()
            if oldest_id in self.flow_cache:
                del self.flow_cache[oldest_id]
        
        # Add new flow
        self.flow_cache[frame_pair_id] = flow.copy()
        self.cache_order.append(frame_pair_id)
    
    def request_flow(
        self, 
        frame_id1: int, 
        frame_id2: int, 
        img1: np.ndarray, 
        img2: np.ndarray
    ) -> Optional[str]:
        """
        Request optical flow computation.
        
        Args:
            frame_id1, frame_id2: Frame identifiers
            img1, img2: Input images as numpy arrays
            
        Returns:
            Request ID if queued, None if queue full
        """
        frame_pair_id = f"{frame_id1}_{frame_id2}"
        timestamp = time.time()
        
        try:
            self.request_queue.put_nowait((frame_pair_id, img1, img2, timestamp))
            self.stats['requests'] += 1
            self.stats['queue_size'].append(self.request_queue.qsize())
            return frame_pair_id
        except queue.Full:
            print("Flow computation queue full, skipping request")
            return None
    
    def get_flow(self, frame_pair_id: str, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get computed flow result.
        
        Args:
            frame_pair_id: Request identifier
            timeout: Maximum wait time
            
        Returns:
            Flow array or None if not ready
        """
        # Check cache first
        with self.cache_lock:
            if frame_pair_id in self.flow_cache:
                return self.flow_cache[frame_pair_id].copy()
        
        # Check result queue
        try:
            while True:
                result_type, result_id, flow, timestamp = self.result_queue.get(timeout=timeout)
                
                if result_id == frame_pair_id:
                    return flow.copy()
                else:
                    # Put back in queue for other requests
                    self.result_queue.put((result_type, result_id, flow, timestamp))
                    break
        except queue.Empty:
            return None
    
    def get_flow_blocking(self, frame_pair_id: str, max_wait: float = 5.0) -> Optional[np.ndarray]:
        """Get flow with blocking wait."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            flow = self.get_flow(frame_pair_id, timeout=0.5)
            if flow is not None:
                return flow
        
        print(f"Flow request {frame_pair_id} timed out after {max_wait}s")
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        if stats['compute_time_ms']:
            stats['avg_compute_time_ms'] = np.mean(stats['compute_time_ms'])
            stats['max_compute_time_ms'] = np.max(stats['compute_time_ms'])
        
        if stats['queue_size']:
            stats['avg_queue_size'] = np.mean(stats['queue_size'])
            stats['max_queue_size'] = np.max(stats['queue_size'])
        
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['requests'], 1) * 100
        )
        
        return stats
    
    def clear_cache(self):
        """Clear flow cache."""
        with self.cache_lock:
            self.flow_cache.clear()
            self.cache_order.clear()
    
    def shutdown(self):
        """Shutdown the flow manager."""
        print("Shutting down flow manager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Send shutdown signal to worker
        try:
            self.request_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Wait for worker to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        # Clear resources
        self.clear_cache()
        
        # Print final statistics
        stats = self.get_statistics()
        print(f"Flow manager stats: {stats}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


class OpticalFlowCache:
    """Simple caching layer for optical flow results."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_order = deque(maxlen=max_size)
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached flow."""
        with self.lock:
            if key in self.cache:
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key].copy()
            return None
    
    def put(self, key: str, flow: np.ndarray):
        """Cache flow result."""
        with self.lock:
            # Remove oldest if necessary
            if len(self.cache) >= self.max_size:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = flow.copy()
            self.access_order.append(key)
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()