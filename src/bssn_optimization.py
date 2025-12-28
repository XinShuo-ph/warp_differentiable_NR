"""
Gradient-Based Parameter Optimization for BSSN

Demonstrates using Warp's autodiff to optimize initial data parameters
and gauge conditions through gradient descent.
"""

import sys
sys.path.insert(0, '/workspace/src')

import warp as wp
import numpy as np
from bssn_vars import BSSNGrid
from bssn_initial_data import set_schwarzschild_puncture
from bssn_rhs_full import compute_bssn_rhs_full_kernel
from bssn_boundary import apply_standard_bssn_boundaries
from bssn_losses import (
    DifferentiableLoss, constraint_loss_kernel, 
    stability_loss_kernel, asymptotic_flatness_loss_kernel
)
from bssn_constraints import (
    compute_hamiltonian_constraint_kernel, 
    compute_momentum_constraint_kernel
)


@wp.kernel
def sgd_update_kernel(
    param: wp.array(dtype=wp.float32),
    grad: wp.array(dtype=wp.float32),
    lr: float
):
    """Stochastic gradient descent update: param = param - lr * grad"""
    tid = wp.tid()
    param[tid] = param[tid] - lr * grad[tid]


@wp.kernel
def set_gauge_parameter_kernel(
    alpha: wp.array(dtype=wp.float32),
    param: wp.array(dtype=wp.float32)
):
    """Set lapse from a parameter array (for optimization)."""
    tid = wp.tid()
    alpha[tid] = param[tid]


@wp.kernel
def copy_to_param_kernel(
    src: wp.array(dtype=wp.float32),
    param: wp.array(dtype=wp.float32)
):
    """Copy array to parameter."""
    tid = wp.tid()
    param[tid] = src[tid]


class LapseOptimizer:
    """
    Optimizer for lapse function to minimize constraint violations.
    
    This demonstrates how autodiff can be used to learn better gauge conditions.
    """
    def __init__(self, grid, learning_rate=0.01):
        self.grid = grid
        self.lr = learning_rate
        
        # Parameter to optimize (copy of alpha)
        self.alpha_param = wp.zeros(grid.n_points, dtype=wp.float32, 
                                     requires_grad=True)
        
        # Constraint arrays
        self.H = wp.zeros(grid.n_points, dtype=wp.float32, requires_grad=True)
        self.M1 = wp.zeros(grid.n_points, dtype=wp.float32, requires_grad=True)
        self.M2 = wp.zeros(grid.n_points, dtype=wp.float32, requires_grad=True)
        self.M3 = wp.zeros(grid.n_points, dtype=wp.float32, requires_grad=True)
        
        # Loss
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    def initialize(self):
        """Initialize parameter from current grid lapse."""
        wp.launch(
            copy_to_param_kernel,
            dim=self.grid.n_points,
            inputs=[self.grid.alpha, self.alpha_param]
        )
    
    def compute_loss(self):
        """Compute constraint violation loss for current parameters."""
        # Copy parameters to grid
        wp.launch(
            set_gauge_parameter_kernel,
            dim=self.grid.n_points,
            inputs=[self.grid.alpha, self.alpha_param]
        )
        
        # Compute constraints
        inv_dx = 1.0 / self.grid.dx
        wp.launch(
            compute_hamiltonian_constraint_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.grid.phi, self.grid.gt11, self.grid.gt12, self.grid.gt13,
                self.grid.gt22, self.grid.gt23, self.grid.gt33,
                self.grid.trK, self.grid.At11, self.grid.At12, self.grid.At13,
                self.grid.At22, self.grid.At23, self.grid.At33,
                self.H,
                self.grid.nx, self.grid.ny, self.grid.nz, inv_dx
            ]
        )
        
        wp.launch(
            compute_momentum_constraint_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.grid.phi, self.grid.gt11, self.grid.gt12, self.grid.gt13,
                self.grid.gt22, self.grid.gt23, self.grid.gt33,
                self.grid.trK, self.grid.At11, self.grid.At12, self.grid.At13,
                self.grid.At22, self.grid.At23, self.grid.At14,
                self.grid.Xt1, self.grid.Xt2, self.grid.Xt3,
                self.M1, self.M2, self.M3,
                self.grid.nx, self.grid.ny, self.grid.nz, inv_dx
            ]
        )
        
        # Compute loss
        self.loss.zero_()
        wp.launch(
            constraint_loss_kernel,
            dim=self.grid.n_points,
            inputs=[self.H, self.M1, self.M2, self.M3, 
                    self.loss, self.grid.n_points]
        )
        
        return self.loss
    
    def step(self, tape):
        """Perform one optimization step."""
        tape.backward(loss=self.loss)
        
        if self.alpha_param.grad is not None:
            wp.launch(
                sgd_update_kernel,
                dim=self.grid.n_points,
                inputs=[self.alpha_param, self.alpha_param.grad, self.lr]
            )
        
        tape.zero()


def optimize_initial_lapse():
    """
    Example: Optimize lapse function to minimize constraint violations.
    """
    wp.init()
    print("=" * 60)
    print("Lapse Optimization via Gradient Descent")
    print("=" * 60)
    
    # Create grid with Schwarzschild data
    nx, ny, nz = 24, 24, 24
    dx = 0.5
    grid = BSSNGrid(nx, ny, nz, dx, requires_grad=True)
    set_schwarzschild_puncture(grid, bh_mass=1.0, pre_collapse_lapse=True)
    
    print(f"\nGrid: {nx}x{ny}x{nz}, dx = {dx}")
    print("Initial data: Schwarzschild puncture")
    print("\nOptimizing lapse to minimize asymptotic flatness deviation...\n")
    
    # Use simpler asymptotic loss that clearly depends on alpha
    loss_arr = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    # Copy initial lapse
    alpha_param = wp.zeros_like(grid.alpha, requires_grad=True)
    wp.copy(alpha_param, grid.alpha)
    
    n_iters = 20
    lr = 0.1
    
    print("Iter  |  Loss")
    print("-" * 25)
    
    for i in range(n_iters):
        tape = wp.Tape()
        with tape:
            # Copy param to grid
            wp.copy(grid.alpha, alpha_param)
            
            # Compute asymptotic flatness loss
            loss_arr.zero_()
            wp.launch(
                asymptotic_flatness_loss_kernel,
                dim=grid.n_points,
                inputs=[grid.phi, grid.alpha,
                        grid.gt11, grid.gt22, grid.gt33,
                        grid.nx, grid.ny, grid.nz,
                        grid.dx, loss_arr]
            )
        
        loss_val = float(loss_arr.numpy()[0])
        
        if i % 5 == 0 or i == n_iters - 1:
            print(f"{i:5d} | {loss_val:.6e}")
        
        tape.backward(loss=loss_arr)
        
        # Update parameters
        if alpha_param.grad is not None:
            wp.launch(
                sgd_update_kernel,
                dim=grid.n_points,
                inputs=[alpha_param, alpha_param.grad, lr]
            )
        
        tape.zero()
    
    print("-" * 25)
    
    # Compare initial and optimized lapse at boundaries
    initial_grid = BSSNGrid(nx, ny, nz, dx)
    set_schwarzschild_puncture(initial_grid, bh_mass=1.0, pre_collapse_lapse=True)
    
    initial_alpha = initial_grid.alpha.numpy()
    optimized_alpha = alpha_param.numpy()
    
    print(f"\nBoundary lapse comparison:")
    print(f"  Initial α at corner:   {initial_alpha[0]:.4f}")
    print(f"  Optimized α at corner: {optimized_alpha[0]:.4f}")
    print(f"  Target (flat):         1.0000")
    
    print("\n✓ Lapse optimization completed!")
    print("  Gradient-based optimization working through BSSN loss functions.")


def demonstrate_parameter_sensitivity():
    """
    Demonstrate computing gradients of loss with respect to parameters.
    """
    wp.init()
    print("=" * 60)
    print("Parameter Sensitivity Analysis via Autodiff")
    print("=" * 60)
    
    nx, ny, nz = 24, 24, 24
    dx = 0.5
    grid = BSSNGrid(nx, ny, nz, dx, requires_grad=True)
    set_schwarzschild_puncture(grid, bh_mass=1.0, pre_collapse_lapse=True)
    
    print(f"\nComputing ∂Loss/∂α for asymptotic flatness loss...")
    
    # Mark alpha for gradient computation
    grid.alpha.requires_grad = True
    
    loss_arr = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    tape = wp.Tape()
    with tape:
        loss_arr.zero_()
        wp.launch(
            asymptotic_flatness_loss_kernel,
            dim=grid.n_points,
            inputs=[grid.phi, grid.alpha,
                    grid.gt11, grid.gt22, grid.gt33,
                    grid.nx, grid.ny, grid.nz,
                    grid.dx, loss_arr]
        )
    
    tape.backward(loss=loss_arr)
    
    grad = grid.alpha.grad.numpy()
    
    print(f"\nGradient statistics:")
    print(f"  max(∂L/∂α): {np.max(grad):.6e}")
    print(f"  min(∂L/∂α): {np.min(grad):.6e}")
    print(f"  mean(∂L/∂α): {np.mean(grad):.6e}")
    print(f"  std(∂L/∂α): {np.std(grad):.6e}")
    
    # Find location of max gradient
    grad_3d = grad.reshape(nx, ny, nz)
    idx = np.unravel_index(np.argmax(np.abs(grad_3d)), grad_3d.shape)
    print(f"\nMax gradient at grid point {idx}")
    
    print("\n✓ Parameter sensitivity analysis completed!")
    print("  Autodiff provides gradients for optimization and ML training.")


if __name__ == "__main__":
    demonstrate_parameter_sensitivity()
    print("\n")
    optimize_initial_lapse()
