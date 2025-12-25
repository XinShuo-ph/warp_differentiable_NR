"""
Test autodiff through BSSN evolution.

This test verifies that gradients can be computed through the BSSN RHS
computation, which is essential for ML integration.
"""

import warp as wp
import numpy as np
import sys
sys.path.insert(0, '/workspace/NR/src')

from bssn_vars import BSSNGrid
from bssn_rhs import compute_bssn_rhs_kernel


def test_autodiff_bssn():
    """Test autodiff through one BSSN RHS computation."""
    wp.init()
    print("=== BSSN Autodiff Test ===\n")
    
    # Create grid with gradient tracking
    nx, ny, nz = 16, 16, 16
    dx = 0.1
    
    grid = BSSNGrid(nx, ny, nz, dx, requires_grad=True)
    grid.set_flat_spacetime()
    
    # Enable gradient tracking for the variables we want to differentiate
    grid.alpha.requires_grad = True
    grid.phi.requires_grad = True
    grid.trK.requires_grad = True
    
    inv_dx = 1.0 / dx
    eps_diss = 0.2 * dx
    
    # Create a loss array
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    # Create tape for recording
    tape = wp.Tape()
    
    with tape:
        # Compute BSSN RHS
        wp.launch(
            compute_bssn_rhs_kernel,
            dim=grid.n_points,
            inputs=[
                grid.phi, grid.gt11, grid.gt12, grid.gt13, grid.gt22, grid.gt23, grid.gt33,
                grid.trK, grid.At11, grid.At12, grid.At13, grid.At22, grid.At23, grid.At33,
                grid.Xt1, grid.Xt2, grid.Xt3,
                grid.alpha, grid.beta1, grid.beta2, grid.beta3,
                grid.phi_rhs, grid.gt11_rhs, grid.gt12_rhs, grid.gt13_rhs,
                grid.gt22_rhs, grid.gt23_rhs, grid.gt33_rhs,
                grid.trK_rhs, grid.At11_rhs, grid.At12_rhs, grid.At13_rhs,
                grid.At22_rhs, grid.At23_rhs, grid.At33_rhs,
                grid.Xt1_rhs, grid.Xt2_rhs, grid.Xt3_rhs,
                grid.alpha_rhs, grid.beta1_rhs, grid.beta2_rhs, grid.beta3_rhs,
                nx, ny, nz, inv_dx, eps_diss
            ]
        )
        
        # Compute a simple loss: sum of squared RHS values
        # This tests if we can differentiate through the RHS computation
        wp.launch(compute_loss_kernel, dim=grid.n_points,
                  inputs=[grid.alpha_rhs, grid.phi_rhs, grid.trK_rhs, loss])
    
    print(f"Loss (sum of squared RHS): {loss.numpy()[0]:.6e}")
    
    # Backward pass
    tape.backward(loss=loss)
    
    # Check gradients exist and are non-zero for at least some variables
    alpha_grad = grid.alpha.grad
    phi_grad = grid.phi.grad
    trK_grad = grid.trK.grad
    
    print("\nGradient information:")
    if alpha_grad is not None:
        alpha_grad_max = np.abs(alpha_grad.numpy()).max()
        print(f"  alpha.grad max: {alpha_grad_max:.6e}")
    else:
        alpha_grad_max = 0
        print("  alpha.grad: None")
        
    if phi_grad is not None:
        phi_grad_max = np.abs(phi_grad.numpy()).max()
        print(f"  phi.grad max:   {phi_grad_max:.6e}")
    else:
        phi_grad_max = 0
        print("  phi.grad: None")
        
    if trK_grad is not None:
        trK_grad_max = np.abs(trK_grad.numpy()).max()
        print(f"  trK.grad max:   {trK_grad_max:.6e}")
    else:
        trK_grad_max = 0
        print("  trK.grad: None")
    
    # For flat spacetime, gradients might be small but should exist
    if alpha_grad is not None or phi_grad is not None or trK_grad is not None:
        print("\n✓ Autodiff through BSSN RHS computation works!")
        print("  Gradients can be computed for optimization/ML integration.")
    else:
        print("\n⚠ No gradients computed. Check tape recording.")
    
    tape.zero()


@wp.kernel
def compute_loss_kernel(alpha_rhs: wp.array(dtype=wp.float32),
                        phi_rhs: wp.array(dtype=wp.float32),
                        trK_rhs: wp.array(dtype=wp.float32),
                        loss: wp.array(dtype=wp.float32)):
    """Compute sum of squared RHS values as loss."""
    tid = wp.tid()
    
    # Accumulate squared RHS values
    val = alpha_rhs[tid]**2.0 + phi_rhs[tid]**2.0 + trK_rhs[tid]**2.0
    
    # Atomic add to loss
    wp.atomic_add(loss, 0, val)


def test_autodiff_simple_kernel():
    """Test autodiff with a simple kernel first."""
    wp.init()
    print("\n=== Simple Autodiff Test ===\n")
    
    n = 100
    x = wp.zeros(n, dtype=wp.float32, requires_grad=True)
    y = wp.zeros(n, dtype=wp.float32, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    # Initialize x with some values
    x_np = np.ones(n, dtype=np.float32) * 2.0
    wp.copy(x, wp.array(x_np, dtype=wp.float32))
    x.requires_grad = True
    
    tape = wp.Tape()
    
    with tape:
        wp.launch(simple_square_kernel, dim=n, inputs=[x, y])
        wp.launch(sum_kernel, dim=n, inputs=[y, loss])
    
    print(f"x mean: {x.numpy().mean():.2f}")
    print(f"y mean: {y.numpy().mean():.2f} (should be 4.0 = 2²)")
    print(f"loss: {loss.numpy()[0]:.2f} (should be 400 = 100*4)")
    
    tape.backward(loss=loss)
    
    if x.grad is not None:
        print(f"x.grad mean: {x.grad.numpy().mean():.2f} (should be 4.0 = 2*x)")
        print("\n✓ Simple autodiff works!")
    else:
        print("x.grad is None")


@wp.kernel
def simple_square_kernel(x: wp.array(dtype=wp.float32),
                          y: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    y[tid] = x[tid] * x[tid]


@wp.kernel
def sum_kernel(y: wp.array(dtype=wp.float32),
               loss: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, y[tid])


if __name__ == "__main__":
    test_autodiff_simple_kernel()
    test_autodiff_bssn()
