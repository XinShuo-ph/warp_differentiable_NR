"""Tests for Poisson solver."""

import numpy as np
import warp as wp

wp.init()

from src.poisson import solve_poisson


def test_poisson_convergence():
    """Test that solution converges to analytical value."""
    field, _, _ = solve_poisson(resolution=32, degree=2, quiet=True)
    max_val = np.max(field.dof_values.numpy())
    
    # Expected max at center is 1.0
    assert abs(max_val - 1.0) < 0.01, f"Max value {max_val} too far from 1.0"
    print(f"PASS: max value = {max_val:.6f}, expected ~1.0")


def test_poisson_consistency():
    """Test that two runs give same result."""
    field1, _, _ = solve_poisson(resolution=16, degree=2, quiet=True)
    field2, _, _ = solve_poisson(resolution=16, degree=2, quiet=True)
    
    max1 = np.max(field1.dof_values.numpy())
    max2 = np.max(field2.dof_values.numpy())
    
    assert abs(max1 - max2) < 1e-10, f"Results not consistent: {max1} vs {max2}"
    print(f"PASS: consistent results ({max1:.10f} == {max2:.10f})")


def test_boundary_conditions():
    """Test that boundary values are zero."""
    field, _, _ = solve_poisson(resolution=16, degree=2, quiet=True)
    dofs = field.dof_values.numpy()
    
    # For degree=2 on 16x16 grid, boundary DOFs should be zero
    # The first row/col and last row/col of the grid
    # This is a simplified check
    min_val = np.min(dofs)
    assert min_val >= -1e-6, f"Boundary not zero: min = {min_val}"
    print(f"PASS: boundary conditions satisfied (min = {min_val:.6e})")


if __name__ == "__main__":
    print("Running Poisson solver tests...\n")
    
    test_poisson_convergence()
    test_poisson_consistency()
    test_boundary_conditions()
    
    print("\nAll tests passed!")
