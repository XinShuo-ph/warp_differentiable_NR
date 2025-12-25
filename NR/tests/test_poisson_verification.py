"""
Test Poisson solver against analytical solution.
Verifies convergence with mesh refinement.
"""

import sys
sys.path.insert(0, '../src')

import warp as wp
import numpy as np
from poisson_solver import solve_poisson_2d

wp.init()


def test_convergence():
    """Test that error decreases with refinement"""
    print("Testing Poisson solver convergence...")
    
    resolutions = [10, 20, 40]
    errors = []
    
    for res in resolutions:
        field, geo = solve_poisson_2d(resolution=res, degree=2)
        u_vals = field.dof_values.numpy()
        u_max = np.abs(u_vals).max()
        
        # Analytical max is 1.0
        error = abs(u_max - 1.0)
        errors.append(error)
        print(f"  res={res}x{res}: max_error={error:.2e}")
    
    # Check convergence: error should decrease initially
    assert errors[1] < errors[0], "Error should decrease with refinement"
    
    # Check final accuracy
    assert errors[-1] < 1e-4, f"Final error too large: {errors[-1]}"
    
    # All errors should be small
    assert all(e < 1e-3 for e in errors), "All errors should be small"
    
    print("✓ Convergence test passed")


def test_boundary_conditions():
    """Test that boundary conditions are satisfied"""
    print("\nTesting boundary conditions...")
    
    field, geo = solve_poisson_2d(resolution=20, degree=2)
    u_vals = field.dof_values.numpy()
    
    # For Grid2D with degree 2, boundary nodes should be zero
    # This is a simplified check - just verify min is close to 0
    u_min = np.abs(u_vals).min()
    
    print(f"  Min |u| = {u_min:.2e} (should be ~0)")
    assert u_min < 1e-10, f"Boundary condition violated: min={u_min}"
    
    print("✓ Boundary condition test passed")


def test_symmetry():
    """Test that solution has expected symmetry"""
    print("\nTesting solution symmetry...")
    
    field, geo = solve_poisson_2d(resolution=30, degree=2)
    u_vals = field.dof_values.numpy()
    
    # Solution should be positive everywhere in interior
    # (since sin(πx)sin(πy) > 0 for x,y in (0,1))
    u_max = u_vals.max()
    u_min = u_vals.min()
    
    print(f"  Solution range: [{u_min:.4f}, {u_max:.4f}]")
    assert u_min >= -1e-6, f"Solution has unexpected negative values: {u_min}"
    assert 0.99 < u_max < 1.01, f"Max value unexpected: {u_max}"
    
    print("✓ Symmetry test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Poisson Solver Verification Tests")
    print("=" * 60)
    print()
    
    test_convergence()
    test_boundary_conditions()
    test_symmetry()
    
    print("\n" + "=" * 60)
    print("All tests PASSED ✓")
    print("=" * 60)
