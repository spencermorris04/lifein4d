# utils/adaptive_loss_scheduler.py

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import math


class AdaptiveLossScheduler:
    """
    Adaptive loss weight scheduler that adjusts weights based on training progress.
    
    Features:
    - Convergence-based scheduling
    - Loss balance monitoring
    - Automatic warm-up periods
    - Gradient magnitude balancing
    - Performance-based adaptation
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_method: str = 'gradient_balance',
        window_size: int = 50,
        adaptation_rate: float = 0.1,
        min_weights: Optional[Dict[str, float]] = None,
        max_weights: Optional[Dict[str, float]] = None,
        warmup_iterations: Dict[str, int] = None,
        target_loss_ratios: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize adaptive loss scheduler.
        
        Args:
            initial_weights: Initial loss weights
            adaptation_method: 'gradient_balance', 'loss_ratio', 'convergence', or 'hybrid'
            window_size: Window for computing moving averages
            adaptation_rate: Rate of weight adaptation
            min_weights: Minimum allowed weights
            max_weights: Maximum allowed weights
            warmup_iterations: Warm-up periods for each loss
            target_loss_ratios: Target ratios between losses
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.adaptation_method = adaptation_method
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # Constraints
        self.min_weights = min_weights or {k: v * 0.01 for k, v in initial_weights.items()}
        self.max_weights = max_weights or {k: v * 100 for k, v in initial_weights.items()}
        
        # Warm-up configuration
        self.warmup_iterations = warmup_iterations or {}
        self.target_loss_ratios = target_loss_ratios or {}
        
        # History tracking
        self.loss_history = {k: deque(maxlen=window_size) for k in initial_weights.keys()}
        self.gradient_history = {k: deque(maxlen=window_size) for k in initial_weights.keys()}
        self.weight_history = {k: deque(maxlen=1000) for k in initial_weights.keys()}
        
        # Statistics
        self.stats = {
            'total_adaptations': 0,
            'convergence_detected': {},
            'adaptation_history': [],
            'gradient_magnitudes': {},
            'loss_ratios': {},
        }
        
        # State tracking
        self.iteration = 0
        self.converged_losses = set()
        self.is_warmup_active = True
        
        print(f"Initialized adaptive scheduler with method: {adaptation_method}")
        print(f"Initial weights: {initial_weights}")
    
    def update(
        self,
        iteration: int,
        loss_values: Dict[str, float],
        gradient_norms: Optional[Dict[str, float]] = None,
        model_parameters: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Update loss weights based on training progress.
        
        Args:
            iteration: Current training iteration
            loss_values: Current loss values
            gradient_norms: Gradient norms for each loss component
            model_parameters: Model parameters for gradient computation
            
        Returns:
            Updated loss weights
        """
        self.iteration = iteration
        
        # Update histories
        self._update_histories(loss_values, gradient_norms)
        
        # Apply warm-up if needed
        if self._is_warmup_period():
            weights = self._apply_warmup()
        else:
            # Main adaptation logic
            if self.adaptation_method == 'gradient_balance':
                weights = self._gradient_balance_adaptation(gradient_norms, model_parameters)
            elif self.adaptation_method == 'loss_ratio':
                weights = self._loss_ratio_adaptation()
            elif self.adaptation_method == 'convergence':
                weights = self._convergence_based_adaptation()
            elif self.adaptation_method == 'hybrid':
                weights = self._hybrid_adaptation(gradient_norms, model_parameters)
            else:
                weights = self.current_weights.copy()
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        # Update current weights and history
        self.current_weights = weights
        for k, v in weights.items():
            self.weight_history[k].append(v)
        
        # Update statistics
        self._update_statistics(loss_values, gradient_norms)
        
        return weights
    
    def _update_histories(
        self,
        loss_values: Dict[str, float],
        gradient_norms: Optional[Dict[str, float]]
    ):
        """Update loss and gradient histories."""
        for k, v in loss_values.items():
            if k in self.loss_history:
                self.loss_history[k].append(v)
        
        if gradient_norms:
            for k, v in gradient_norms.items():
                if k in self.gradient_history:
                    self.gradient_history[k].append(v)
    
    def _is_warmup_period(self) -> bool:
        """Check if any loss is still in warm-up period."""
        for loss_name, warmup_iters in self.warmup_iterations.items():
            if self.iteration < warmup_iters:
                return True
        return False
    
    def _apply_warmup(self) -> Dict[str, float]:
        """Apply warm-up scheduling."""
        weights = {}
        
        for loss_name, initial_weight in self.initial_weights.items():
            warmup_iters = self.warmup_iterations.get(loss_name, 0)
            
            if self.iteration < warmup_iters:
                # Linear warm-up from 0 to initial weight
                progress = self.iteration / warmup_iters
                weights[loss_name] = initial_weight * progress
            else:
                weights[loss_name] = initial_weight
        
        return weights
    
    def _gradient_balance_adaptation(
        self,
        gradient_norms: Optional[Dict[str, float]],
        model_parameters: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Adapt weights to balance gradient magnitudes."""
        if not gradient_norms and model_parameters is None:
            return self.current_weights.copy()
        
        # Compute or use provided gradient norms
        if gradient_norms is None:
            gradient_norms = self._compute_gradient_norms(model_parameters)
        
        if not gradient_norms:
            return self.current_weights.copy()
        
        # Target: balance gradient magnitudes across loss components
        target_magnitude = np.mean(list(gradient_norms.values()))
        
        weights = {}
        for loss_name in self.current_weights.keys():
            if loss_name in gradient_norms:
                current_grad = gradient_norms[loss_name]
                current_weight = self.current_weights[loss_name]
                
                if current_grad > 0:
                    # Adjust weight to bring gradient closer to target
                    ratio = target_magnitude / (current_grad + 1e-8)
                    new_weight = current_weight * (1 - self.adaptation_rate + self.adaptation_rate * ratio)
                else:
                    new_weight = current_weight
                
                weights[loss_name] = new_weight
            else:
                weights[loss_name] = self.current_weights[loss_name]
        
        return weights
    
    def _loss_ratio_adaptation(self) -> Dict[str, float]:
        """Adapt weights to maintain target loss ratios."""
        if not self.target_loss_ratios:
            return self.current_weights.copy()
        
        # Compute current loss ratios
        current_losses = {k: v[-1] if v else 0.0 for k, v in self.loss_history.items()}
        
        # Reference loss (usually photometric)
        ref_loss = 'photometric_loss'
        if ref_loss not in current_losses or current_losses[ref_loss] <= 0:
            return self.current_weights.copy()
        
        weights = self.current_weights.copy()
        ref_loss_value = current_losses[ref_loss]
        
        for loss_name, target_ratio in self.target_loss_ratios.items():
            if loss_name in current_losses and loss_name != ref_loss:
                current_loss_value = current_losses[loss_name]
                current_ratio = current_loss_value / (ref_loss_value + 1e-8)
                
                if current_ratio > 0:
                    # Adjust weight to achieve target ratio
                    ratio_error = target_ratio / (current_ratio + 1e-8)
                    current_weight = self.current_weights[loss_name]
                    new_weight = current_weight * (1 - self.adaptation_rate + self.adaptation_rate * ratio_error)
                    weights[loss_name] = new_weight
        
        return weights
    
    def _convergence_based_adaptation(self) -> Dict[str, float]:
        """Adapt weights based on convergence detection."""
        weights = self.current_weights.copy()
        
        for loss_name, loss_history in self.loss_history.items():
            if len(loss_history) < self.window_size // 2:
                continue
            
            # Detect convergence
            is_converged = self._detect_convergence(loss_history)
            
            if is_converged and loss_name not in self.converged_losses:
                # Reduce weight for converged losses
                weights[loss_name] *= 0.5
                self.converged_losses.add(loss_name)
                print(f"Detected convergence for {loss_name}, reducing weight to {weights[loss_name]:.6f}")
            
            elif not is_converged and loss_name in self.converged_losses:
                # Re-increase weight if loss starts changing again
                weights[loss_name] = min(weights[loss_name] * 2.0, self.initial_weights[loss_name])
                self.converged_losses.discard(loss_name)
                print(f"Re-activating {loss_name}, increasing weight to {weights[loss_name]:.6f}")
        
        return weights
    
    def _hybrid_adaptation(
        self,
        gradient_norms: Optional[Dict[str, float]],
        model_parameters: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Hybrid adaptation combining multiple strategies."""
        # Start with gradient balancing
        weights_grad = self._gradient_balance_adaptation(gradient_norms, model_parameters)
        
        # Apply convergence-based adjustments
        weights_conv = self._convergence_based_adaptation()
        
        # Combine strategies
        weights = {}
        for loss_name in self.current_weights.keys():
            # Weighted combination of strategies
            w_grad = weights_grad.get(loss_name, self.current_weights[loss_name])
            w_conv = weights_conv.get(loss_name, self.current_weights[loss_name])
            
            # More weight to convergence-based when training progresses
            progress = min(self.iteration / 5000, 1.0)  # Assume 5000 iterations for full progression
            combined_weight = (1 - progress) * w_grad + progress * w_conv
            
            weights[loss_name] = combined_weight
        
        return weights
    
    def _detect_convergence(self, loss_history: deque, threshold: float = 1e-4) -> bool:
        """Detect if a loss has converged."""
        if len(loss_history) < self.window_size // 2:
            return False
        
        recent_losses = list(loss_history)[-self.window_size // 2:]
        
        # Check relative change
        if len(recent_losses) >= 2:
            relative_changes = []
            for i in range(1, len(recent_losses)):
                if recent_losses[i-1] > 0:
                    rel_change = abs(recent_losses[i] - recent_losses[i-1]) / recent_losses[i-1]
                    relative_changes.append(rel_change)
            
            if relative_changes:
                mean_rel_change = np.mean(relative_changes)
                return mean_rel_change < threshold
        
        return False
    
    def _compute_gradient_norms(
        self,
        model_parameters: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        gradient_norms = {}
        
        for param_name, param in model_parameters.items():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                gradient_norms[param_name] = grad_norm
        
        return gradient_norms
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max constraints to weights."""
        constrained_weights = {}
        
        for loss_name, weight in weights.items():
            min_w = self.min_weights.get(loss_name, 0.0)
            max_w = self.max_weights.get(loss_name, float('inf'))
            
            constrained_weights[loss_name] = np.clip(weight, min_w, max_w)
        
        return constrained_weights
    
    def _update_statistics(
        self,
        loss_values: Dict[str, float],
        gradient_norms: Optional[Dict[str, float]]
    ):
        """Update scheduler statistics."""
        # Update gradient magnitude statistics
        if gradient_norms:
            for k, v in gradient_norms.items():
                if k not in self.stats['gradient_magnitudes']:
                    self.stats['gradient_magnitudes'][k] = []
                self.stats['gradient_magnitudes'][k].append(v)
                
                # Keep only recent history
                if len(self.stats['gradient_magnitudes'][k]) > 1000:
                    self.stats['gradient_magnitudes'][k].pop(0)
        
        # Update loss ratios
        ref_loss = 'photometric_loss'
        if ref_loss in loss_values and loss_values[ref_loss] > 0:
            for k, v in loss_values.items():
                if k != ref_loss:
                    ratio = v / loss_values[ref_loss]
                    if k not in self.stats['loss_ratios']:
                        self.stats['loss_ratios'][k] = []
                    self.stats['loss_ratios'][k].append(ratio)
                    
                    # Keep only recent history
                    if len(self.stats['loss_ratios'][k]) > 1000:
                        self.stats['loss_ratios'][k].pop(0)
        
        # Record adaptation if weights changed significantly
        prev_weights = {k: v[-2] if len(v) >= 2 else v[-1] for k, v in self.weight_history.items() if v}
        
        significant_change = False
        for k, current_w in self.current_weights.items():
            if k in prev_weights:
                relative_change = abs(current_w - prev_weights[k]) / (prev_weights[k] + 1e-8)
                if relative_change > 0.05:  # 5% change threshold
                    significant_change = True
                    break
        
        if significant_change:
            self.stats['total_adaptations'] += 1
            self.stats['adaptation_history'].append({
                'iteration': self.iteration,
                'weights': self.current_weights.copy(),
                'losses': loss_values.copy()
            })
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about the current schedule state."""
        info = {
            'iteration': self.iteration,
            'current_weights': self.current_weights.copy(),
            'initial_weights': self.initial_weights.copy(),
            'converged_losses': list(self.converged_losses),
            'warmup_active': self._is_warmup_period(),
            'adaptation_method': self.adaptation_method,
            'total_adaptations': self.stats['total_adaptations'],
        }
        
        # Add weight change statistics
        weight_changes = {}
        for k, history in self.weight_history.items():
            if len(history) >= 2:
                recent_change = (history[-1] - history[-2]) / (history[-2] + 1e-8)
                weight_changes[k] = recent_change
        
        info['recent_weight_changes'] = weight_changes
        
        # Add convergence information
        convergence_info = {}
        for k, history in self.loss_history.items():
            convergence_info[k] = {
                'is_converged': self._detect_convergence(history),
                'recent_mean': np.mean(list(history)[-10:]) if len(history) >= 10 else 0.0,
                'trend': self._compute_trend(history)
            }
        
        info['convergence_info'] = convergence_info
        
        return info
    
    def _compute_trend(self, history: deque) -> str:
        """Compute trend direction for loss history."""
        if len(history) < 10:
            return 'insufficient_data'
        
        recent = list(history)[-10:]
        first_half = np.mean(recent[:5])
        second_half = np.mean(recent[5:])
        
        relative_change = (second_half - first_half) / (first_half + 1e-8)
        
        if relative_change > 0.05:
            return 'increasing'
        elif relative_change < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def plot_weight_evolution(self, save_path: Optional[str] = None):
        """Plot weight evolution over time."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Adaptive Loss Weight Evolution')
            
            # Weight evolution
            ax1 = axes[0, 0]
            for loss_name, history in self.weight_history.items():
                if history:
                    ax1.plot(list(history), label=loss_name)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Weight')
            ax1.set_title('Weight Evolution')
            ax1.legend()
            ax1.grid(True)
            
            # Loss ratios
            ax2 = axes[0, 1]
            for loss_name, ratios in self.stats['loss_ratios'].items():
                if ratios:
                    ax2.plot(ratios[-100:], label=f'{loss_name}/photometric')
            ax2.set_xlabel('Recent Iterations')
            ax2.set_ylabel('Loss Ratio')
            ax2.set_title('Loss Ratios (Recent)')
            ax2.legend()
            ax2.grid(True)
            
            # Gradient magnitudes
            ax3 = axes[1, 0]
            for param_name, grads in self.stats['gradient_magnitudes'].items():
                if grads:
                    ax3.semilogy(grads[-100:], label=param_name)
            ax3.set_xlabel('Recent Iterations')
            ax3.set_ylabel('Gradient Norm (log scale)')
            ax3.set_title('Gradient Magnitudes (Recent)')
            ax3.legend()
            ax3.grid(True)
            
            # Adaptation events
            ax4 = axes[1, 1]
            if self.stats['adaptation_history']:
                adaptation_iters = [event['iteration'] for event in self.stats['adaptation_history']]
                ax4.hist(adaptation_iters, bins=20, alpha=0.7)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Number of Adaptations')
            ax4.set_title('Adaptation Events')
            ax4.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Weight evolution plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def save_state(self, path: str):
        """Save scheduler state."""
        state = {
            'current_weights': self.current_weights,
            'iteration': self.iteration,
            'converged_losses': list(self.converged_losses),
            'stats': self.stats,
            'weight_history': {k: list(v) for k, v in self.weight_history.items()},
        }
        
        torch.save(state, path)
        print(f"Scheduler state saved to {path}")
    
    def load_state(self, path: str):
        """Load scheduler state."""
        state = torch.load(path, map_location='cpu')
        
        self.current_weights = state['current_weights']
        self.iteration = state['iteration']
        self.converged_losses = set(state['converged_losses'])
        self.stats = state['stats']
        
        # Restore weight history
        for k, history_list in state['weight_history'].items():
            self.weight_history[k] = deque(history_list, maxlen=1000)
        
        print(f"Scheduler state loaded from {path}")


class MultiStageScheduler:
    """Multi-stage scheduler with different strategies for different training phases."""
    
    def __init__(
        self,
        stages: List[Dict[str, Any]],
        stage_transitions: List[int]
    ):
        """
        Initialize multi-stage scheduler.
        
        Args:
            stages: List of stage configurations
            stage_transitions: Iteration numbers for stage transitions
        """
        self.stages = stages
        self.stage_transitions = stage_transitions
        self.current_stage = 0
        self.schedulers = []
        
        # Create scheduler for each stage
        for stage_config in stages:
            scheduler = AdaptiveLossScheduler(**stage_config)
            self.schedulers.append(scheduler)
    
    def update(self, iteration: int, **kwargs) -> Dict[str, float]:
        """Update weights using current stage scheduler."""
        # Check for stage transition
        while (self.current_stage < len(self.stage_transitions) and 
               iteration >= self.stage_transitions[self.current_stage]):
            self.current_stage += 1
            print(f"Transitioning to stage {self.current_stage} at iteration {iteration}")
        
        # Use current stage scheduler
        stage_idx = min(self.current_stage, len(self.schedulers) - 1)
        return self.schedulers[stage_idx].update(iteration, **kwargs)
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about current stage and scheduler."""
        stage_idx = min(self.current_stage, len(self.schedulers) - 1)
        info = self.schedulers[stage_idx].get_schedule_info()
        info['current_stage'] = self.current_stage
        info['total_stages'] = len(self.stages)
        return info