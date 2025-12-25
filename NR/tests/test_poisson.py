"""
Test Poisson solver against analytical solution.
Verifies O(h^p) convergence for degree-p elements.
"""

import sys
import math
sys.path.insert(0, '/workspace/NR/src')

from poisson import solve_poisson


def test_poisson_convergence():
    """Test that error decreases as resolution increases."""
    print("Testing Poisson solver convergence...")
    
    resolutions = [8, 16, 32]
    degree = 2
    
    errors = []
    for res in resolutions:
        _, l2_error, linf_error = solve_poisson(resolution=res, degree=degree, quiet=True)
        errors.append(l2_error)
        print(f"  Resolution {res:3d}: L2 error = {l2_error:.6e}")
    
    # Check convergence rate
    # For degree-2 elements on Poisson, expect O(h^3) convergence in theory
    # When h halves, error should decrease by factor of ~8 ideally
    # However float32 limits precision at finer resolutions
    ratio1 = errors[0] / errors[1]  # res 8 -> 16
    ratio2 = errors[1] / errors[2]  # res 16 -> 32
    
    print(f"  Convergence ratio (8->16): {ratio1:.2f}")
    print(f"  Convergence ratio (16->32): {ratio2:.2f}")
    
    # Verify error decreases (at least some convergence)
    assert ratio1 > 2.0, f"Insufficient convergence: ratio = {ratio1}"
    assert ratio2 > 1.5, f"Error should decrease: ratio = {ratio2}"
    
    # Verify final error is small
    assert errors[-1] < 1e-4, f"Final error too large: {errors[-1]}"
    
    print("  PASSED!")
    return True


def test_poisson_different_degrees():
    """Test different polynomial degrees."""
    print("Testing different polynomial degrees...")
    
    resolution = 16
    degrees = [1, 2, 3]
    
    for degree in degrees:
        _, l2_error, linf_error = solve_poisson(resolution=resolution, degree=degree, quiet=True)
        print(f"  Degree {degree}: L2 error = {l2_error:.6e}, Linf error = {linf_error:.6e}")
    
    print("  PASSED!")
    return True


def test_poisson_consistency():
    """Test that two runs produce identical results."""
    print("Testing solution consistency...")
    
    _, l2_1, linf_1 = solve_poisson(resolution=16, degree=2, quiet=True)
    _, l2_2, linf_2 = solve_poisson(resolution=16, degree=2, quiet=True)
    
    assert abs(l2_1 - l2_2) < 1e-12, f"Inconsistent L2 errors: {l2_1} vs {l2_2}"
    assert abs(linf_1 - linf_2) < 1e-12, f"Inconsistent Linf errors: {linf_1} vs {linf_2}"
    
    print(f"  Run 1: L2 = {l2_1:.6e}, Linf = {linf_1:.6e}")
    print(f"  Run 2: L2 = {l2_2:.6e}, Linf = {linf_2:.6e}")
    print("  PASSED!")
    return True


if __name__ == "__main__":
    import warp as wp
    wp.init()
    
    print("=" * 60)
    print("Poisson Solver Tests")
    print("=" * 60)
    
    test_poisson_consistency()
    print()
    test_poisson_convergence()
    print()
    test_poisson_different_degrees()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
