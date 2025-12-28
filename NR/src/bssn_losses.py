"""
Differentiable Loss Functions for BSSN Evolution

Provides loss functions suitable for ML integration and optimization,
all compatible with Warp's autodiff system.
"""

import warp as wp
import numpy as np


@wp.kernel
def constraint_loss_kernel(
    H: wp.array(dtype=wp.float32),
    M1: wp.array(dtype=wp.float32),
    M2: wp.array(dtype=wp.float32),
    M3: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    n_points: int
):
    """
    Compute L2 norm of constraint violations.
    Loss = (1/N) * Σ(H² + M₁² + M₂² + M₃²)
    """
    tid = wp.tid()
    
    H_val = H[tid]
    M1_val = M1[tid]
    M2_val = M2[tid]
    M3_val = M3[tid]
    
    local_loss = (H_val * H_val + M1_val * M1_val + 
                  M2_val * M2_val + M3_val * M3_val) / float(n_points)
    
    wp.atomic_add(loss, 0, local_loss)


@wp.kernel
def alpha_target_loss_kernel(
    alpha: wp.array(dtype=wp.float32),
    alpha_target: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    n_points: int
):
    """
    Compute MSE between lapse and target lapse.
    Useful for gauge optimization.
    """
    tid = wp.tid()
    
    diff = alpha[tid] - alpha_target[tid]
    local_loss = (diff * diff) / float(n_points)
    
    wp.atomic_add(loss, 0, local_loss)


@wp.kernel
def stability_loss_kernel(
    phi: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    trK: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    n_points: int
):
    """
    Compute loss that penalizes large values and NaN-inducing conditions.
    """
    tid = wp.tid()
    
    phi_val = phi[tid]
    alpha_val = alpha[tid]
    trK_val = trK[tid]
    
    # Penalize very small lapse (collapse instability)
    alpha_penalty = 0.0
    if alpha_val < 0.1:
        alpha_penalty = (0.1 - alpha_val) * (0.1 - alpha_val)
    
    # Penalize large phi (near puncture blowup)
    phi_penalty = 0.0
    if phi_val > 2.0:
        phi_penalty = (phi_val - 2.0) * (phi_val - 2.0)
    
    # Penalize large trK (singularity)
    trK_penalty = trK_val * trK_val * 0.01
    
    local_loss = (alpha_penalty + phi_penalty + trK_penalty) / float(n_points)
    wp.atomic_add(loss, 0, local_loss)


@wp.kernel
def asymptotic_flatness_loss_kernel(
    phi: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    gt11: wp.array(dtype=wp.float32),
    gt22: wp.array(dtype=wp.float32),
    gt33: wp.array(dtype=wp.float32),
    nx: int, ny: int, nz: int,
    dx: float,
    loss: wp.array(dtype=wp.float32)
):
    """
    Loss that enforces asymptotic flatness at boundaries.
    Penalizes deviation from Minkowski at outer boundaries.
    """
    tid = wp.tid()
    
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Only apply at boundary
    at_boundary = (i < 3 or i >= nx - 3 or 
                   j < 3 or j >= ny - 3 or 
                   k < 3 or k >= nz - 3)
    
    if not at_boundary:
        return
    
    # Target values for flat spacetime
    phi_target = 0.0
    alpha_target = 1.0
    gt_diag_target = 1.0
    
    phi_err = phi[tid] - phi_target
    alpha_err = alpha[tid] - alpha_target
    gt11_err = gt11[tid] - gt_diag_target
    gt22_err = gt22[tid] - gt_diag_target
    gt33_err = gt33[tid] - gt_diag_target
    
    boundary_count = float(6 * nx * ny + 6 * ny * nz + 6 * nx * nz)  # approx
    local_loss = (phi_err * phi_err + alpha_err * alpha_err +
                  gt11_err * gt11_err + gt22_err * gt22_err + 
                  gt33_err * gt33_err) / boundary_count
    
    wp.atomic_add(loss, 0, local_loss)


@wp.kernel
def waveform_loss_kernel(
    alpha: wp.array(dtype=wp.float32),
    alpha_target: wp.array(dtype=wp.float32),
    extraction_mask: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32)
):
    """
    Loss for matching extracted waveform at extraction sphere.
    Uses mask to select extraction region.
    """
    tid = wp.tid()
    
    mask = extraction_mask[tid]
    if mask < 0.5:
        return
    
    diff = alpha[tid] - alpha_target[tid]
    wp.atomic_add(loss, 0, diff * diff * mask)


class DifferentiableLoss:
    """
    Collection of differentiable loss functions for BSSN optimization.
    """
    def __init__(self, grid, requires_grad=False):
        self.grid = grid
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=requires_grad)
    
    def reset(self):
        """Reset loss accumulator."""
        self.loss.zero_()
    
    def get_loss(self):
        """Get accumulated loss value."""
        return float(self.loss.numpy()[0])
    
    def compute_constraint_loss(self, H, M1, M2, M3):
        """Compute constraint violation loss."""
        self.reset()
        wp.launch(
            constraint_loss_kernel,
            dim=self.grid.n_points,
            inputs=[H, M1, M2, M3, self.loss, self.grid.n_points]
        )
        return self.loss
    
    def compute_stability_loss(self):
        """Compute stability penalty loss."""
        self.reset()
        wp.launch(
            stability_loss_kernel,
            dim=self.grid.n_points,
            inputs=[self.grid.phi, self.grid.alpha, self.grid.trK,
                    self.loss, self.grid.n_points]
        )
        return self.loss
    
    def compute_asymptotic_loss(self):
        """Compute asymptotic flatness loss."""
        self.reset()
        wp.launch(
            asymptotic_flatness_loss_kernel,
            dim=self.grid.n_points,
            inputs=[self.grid.phi, self.grid.alpha,
                    self.grid.gt11, self.grid.gt22, self.grid.gt33,
                    self.grid.nx, self.grid.ny, self.grid.nz,
                    self.grid.dx, self.loss]
        )
        return self.loss


def test_losses():
    """Test differentiable loss functions."""
    import sys
    sys.path.insert(0, '/workspace/NR/src')
    from bssn_vars import BSSNGrid
    from bssn_initial_data import set_schwarzschild_puncture
    from bssn_constraints import ConstraintMonitor
    
    wp.init()
    print("=== Differentiable Loss Functions Test ===\n")
    
    # Create grid with Schwarzschild data
    nx, ny, nz = 32, 32, 32
    dx = 0.5
    grid = BSSNGrid(nx, ny, nz, dx, requires_grad=True)
    set_schwarzschild_puncture(grid, bh_mass=1.0)
    
    # Test constraint loss
    monitor = ConstraintMonitor(grid)
    monitor.compute()
    
    loss_fn = DifferentiableLoss(grid)
    
    # Constraint loss
    constraint_loss = loss_fn.compute_constraint_loss(
        monitor.H, monitor.M1, monitor.M2, monitor.M3)
    print(f"Constraint loss: {loss_fn.get_loss():.6e}")
    
    # Stability loss
    stability_loss = loss_fn.compute_stability_loss()
    print(f"Stability loss: {loss_fn.get_loss():.6e}")
    
    # Asymptotic loss
    asymptotic_loss = loss_fn.compute_asymptotic_loss()
    print(f"Asymptotic loss: {loss_fn.get_loss():.6e}")
    
    # Test autodiff through asymptotic loss (depends on alpha directly)
    print("\nTesting autodiff through asymptotic loss (depends on alpha)...")
    
    # Create new loss function with requires_grad
    loss_fn_grad = DifferentiableLoss(grid, requires_grad=True)
    
    # Set requires_grad on alpha
    grid.alpha.requires_grad = True
    
    tape = wp.Tape()
    with tape:
        loss_fn_grad.reset()
        wp.launch(
            asymptotic_flatness_loss_kernel,
            dim=grid.n_points,
            inputs=[grid.phi, grid.alpha,
                    grid.gt11, grid.gt22, grid.gt33,
                    grid.nx, grid.ny, grid.nz,
                    grid.dx, loss_fn_grad.loss]
        )
    
    tape.backward(loss=loss_fn_grad.loss)
    
    # Check gradients
    if grid.alpha.grad is not None:
        grad_max = np.abs(grid.alpha.grad.numpy()).max()
        print(f"  Loss value: {loss_fn_grad.get_loss():.6e}")
        print(f"  alpha.grad max: {grad_max:.6e}")
        if grad_max > 0:
            print("  ✓ Non-zero gradients computed successfully!")
        else:
            print("  ✗ Gradients are zero (check computation graph)")
    else:
        print("  ✗ No gradients computed")
    
    tape.zero()
    
    print("\n✓ Differentiable loss functions test completed.")


if __name__ == "__main__":
    test_losses()
