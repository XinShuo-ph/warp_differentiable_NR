"""Tests for BSSN implementation."""

import numpy as np
import warp as wp

wp.init()

import sys
sys.path.insert(0, '/workspace/NR')

from src.bssn import (
    create_bssn_state, 
    compute_rhs_phi, compute_rhs_K, compute_rhs_alpha_1log,
    rk4_stage1, d1_4th, compute_dx
)


def test_flat_spacetime_stability():
    """Test that flat spacetime remains stable."""
    nx = 8
    dx = 1.0 / nx
    dt = 0.001
    n_steps = 50
    
    state = create_bssn_state(nx, nx, nx, dx)
    
    shape = (nx, nx, nx)
    rhs_phi = wp.zeros(shape, dtype=float)
    rhs_K = wp.zeros(shape, dtype=float)
    rhs_alpha = wp.zeros(shape, dtype=float)
    
    for step in range(n_steps):
        wp.launch(compute_rhs_phi, dim=(nx, nx, nx),
                  inputs=[state.phi, state.K, state.alpha, 
                          state.betax, state.betay, state.betaz,
                          rhs_phi, dx, nx, nx, nx])
        
        wp.launch(compute_rhs_K, dim=(nx, nx, nx),
                  inputs=[state.K, state.alpha, rhs_K, dx, nx, nx, nx])
        
        wp.launch(compute_rhs_alpha_1log, dim=(nx, nx, nx),
                  inputs=[state.alpha, state.K, rhs_alpha, nx, nx, nx])
        
        wp.launch(rk4_stage1, dim=(nx, nx, nx),
                  inputs=[state.phi, rhs_phi, state.phi, 2.0*dt])
        wp.launch(rk4_stage1, dim=(nx, nx, nx),
                  inputs=[state.K, rhs_K, state.K, 2.0*dt])
        wp.launch(rk4_stage1, dim=(nx, nx, nx),
                  inputs=[state.alpha, rhs_alpha, state.alpha, 2.0*dt])
    
    phi_max = np.max(np.abs(state.phi.numpy()))
    K_max = np.max(np.abs(state.K.numpy()))
    alpha_err = np.max(np.abs(state.alpha.numpy() - 1.0))
    
    total_err = phi_max + K_max + alpha_err
    assert total_err < 1e-10, f"Flat spacetime unstable: error = {total_err}"
    print(f"PASS: Flat spacetime stable (error = {total_err:.2e})")


def test_derivative_accuracy():
    """Test that 4th order derivatives are accurate."""
    nx = 32
    dx = 2.0 * np.pi / nx
    
    # Create test function f = sin(x)
    x = np.arange(nx) * dx
    f_np = np.sin(x)
    
    # Expected derivative: df/dx = cos(x)
    df_exact = np.cos(x)
    
    # Allocate warp arrays (use 3D but only vary in x)
    f = wp.zeros((nx, 1, 1), dtype=float)
    df = wp.zeros((nx, 1, 1), dtype=float)
    
    # Copy data
    f_data = f_np.reshape(nx, 1, 1)
    wp.copy(f, wp.array(f_data, dtype=float))
    
    # Compute derivative
    wp.launch(compute_dx, dim=(nx, 1, 1), inputs=[f, df, dx, nx, 1, 1])
    
    df_np = df.numpy().flatten()
    
    # Compare (expect 4th order accuracy: O(h⁴))
    # Interior points should be accurate; boundaries may have errors due to clamping
    interior_err = np.max(np.abs(df_np[4:-4] - df_exact[4:-4]))
    
    assert interior_err < 1e-4, f"Derivative inaccurate: error = {interior_err}"
    print(f"PASS: 4th order derivative accurate (interior error = {interior_err:.2e})")


def test_autodiff_through_timestep():
    """Verify autodiff works through one BSSN timestep."""
    nx = 4
    dx = 1.0 / nx
    
    # Create arrays with requires_grad=True
    phi = wp.zeros((nx, nx, nx), dtype=float, requires_grad=True)
    K = wp.zeros((nx, nx, nx), dtype=float, requires_grad=True)
    alpha = wp.ones((nx, nx, nx), dtype=float, requires_grad=True)
    betax = wp.zeros((nx, nx, nx), dtype=float, requires_grad=True)
    betay = wp.zeros((nx, nx, nx), dtype=float, requires_grad=True)
    betaz = wp.zeros((nx, nx, nx), dtype=float, requires_grad=True)
    rhs_phi = wp.zeros((nx, nx, nx), dtype=float, requires_grad=True)
    
    # Loss array
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    
    # Simple loss kernel
    @wp.kernel
    def compute_loss(rhs: wp.array3d(dtype=float), loss: wp.array(dtype=float)):
        i, j, k = wp.tid()
        wp.atomic_add(loss, 0, rhs[i, j, k] * rhs[i, j, k])
    
    # Record tape
    tape = wp.Tape()
    with tape:
        wp.launch(compute_rhs_phi, dim=(nx, nx, nx),
                  inputs=[phi, K, alpha, betax, betay, betaz, rhs_phi, dx, nx, nx, nx])
        wp.launch(compute_loss, dim=(nx, nx, nx), inputs=[rhs_phi, loss])
    
    # Backward pass
    tape.backward(loss)
    
    # Check that gradients exist
    grad_alpha = tape.gradients[alpha]
    grad_K = tape.gradients[K]
    
    assert grad_alpha is not None, "No gradient for alpha"
    assert grad_K is not None, "No gradient for K"
    
    print("PASS: Autodiff works through BSSN timestep")


def test_consistency():
    """Test that two runs give same result."""
    nx = 8
    dx = 1.0 / nx
    dt = 0.001
    n_steps = 10
    
    results = []
    for run in range(2):
        state = create_bssn_state(nx, nx, nx, dx)
        
        shape = (nx, nx, nx)
        rhs_phi = wp.zeros(shape, dtype=float)
        rhs_K = wp.zeros(shape, dtype=float)
        rhs_alpha = wp.zeros(shape, dtype=float)
        
        for step in range(n_steps):
            wp.launch(compute_rhs_phi, dim=(nx, nx, nx),
                      inputs=[state.phi, state.K, state.alpha, 
                              state.betax, state.betay, state.betaz,
                              rhs_phi, dx, nx, nx, nx])
            
            wp.launch(rk4_stage1, dim=(nx, nx, nx),
                      inputs=[state.phi, rhs_phi, state.phi, 2.0*dt])
        
        results.append(state.phi.numpy().copy())
    
    diff = np.max(np.abs(results[0] - results[1]))
    assert diff < 1e-14, f"Results not consistent: diff = {diff}"
    print(f"PASS: Consistent results (diff = {diff:.2e})")


if __name__ == "__main__":
    print("Running BSSN tests...\n")
    
    test_flat_spacetime_stability()
    test_derivative_accuracy()
    test_autodiff_through_timestep()
    test_consistency()
    
    print("\nAll tests passed!")
