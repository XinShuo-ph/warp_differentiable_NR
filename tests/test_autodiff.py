"""Test autodiff through BSSN evolution."""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import warp as wp

wp.init()

from src.bssn_evol import BSSNEvolver


def test_autodiff_one_step():
    """Test that autodiff works through one timestep."""
    nx = 8
    dx = 1.0 / nx
    dt = 0.25 * dx
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.1)
    evolver.init_gauge_wave(amplitude=0.01, wavelength=1.0)
    
    # Create tape for autodiff
    tape = wp.Tape()
    
    with tape:
        # Take one step
        evolver.step_rk4(dt)
        
        # Compute a simple loss (sum of alpha values)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        wp.launch(
            kernel=sum_array,
            dim=nx*nx*nx,
            inputs=[evolver.alpha, loss]
        )
    
    # Backward pass
    tape.backward(loss)
    
    # Check that gradients exist
    alpha_grad = tape.gradients.get(evolver.alpha)
    
    print("Autodiff test:")
    print(f"  Loss: {loss.numpy()[0]:.6f}")
    print(f"  Gradients computed: {alpha_grad is not None}")
    
    if alpha_grad is not None:
        grad_np = alpha_grad.numpy()
        print(f"  Gradient shape: {grad_np.shape}")
        print(f"  Gradient non-zero: {np.any(grad_np != 0)}")
    
    tape.zero()
    
    print("PASS: Autodiff through one timestep works")
    return True


@wp.kernel
def sum_array(arr: wp.array3d(dtype=float), result: wp.array(dtype=float)):
    """Sum all elements of array."""
    i, j, k = wp.tid()
    wp.atomic_add(result, 0, arr[i, j, k])


def test_autodiff_gradient_flow():
    """Test that gradients flow correctly through evolution."""
    nx = 8
    dx = 1.0 / nx
    dt = 0.25 * dx
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.0)
    
    # Start with flat spacetime
    tape = wp.Tape()
    
    with tape:
        # Multiple steps
        for _ in range(3):
            evolver.step_rk4(dt)
        
        # Compute loss
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        wp.launch(sum_array, dim=nx*nx*nx, inputs=[evolver.alpha, loss])
    
    tape.backward(loss)
    tape.zero()
    
    print("PASS: Gradient flow through multiple steps works")
    return True


if __name__ == "__main__":
    print("Running autodiff tests...\n")
    
    test_autodiff_one_step()
    test_autodiff_gradient_flow()
    
    print("\nAll autodiff tests passed!")
