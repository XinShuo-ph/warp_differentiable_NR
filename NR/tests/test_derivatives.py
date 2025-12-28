"""
Test 4th order finite difference derivatives.
"""

import warp as wp
import numpy as np
from bssn_derivatives import deriv_x_4th, deriv_y_4th, deriv_z_4th
from bssn_derivatives import deriv2_x_4th, deriv2_y_4th, deriv2_z_4th

wp.init()


@wp.kernel
def compute_deriv_x_kernel(
    f: wp.array3d(dtype=float),
    df_dx: wp.array3d(dtype=float),
    idx: float
):
    i, j, k = wp.tid()
    
    # Skip boundary points (need 2 points on each side for 4th order)
    if i >= 2 and i < f.shape[0] - 2:
        df_dx[i, j, k] = deriv_x_4th(f, i, j, k, idx)


@wp.kernel
def compute_deriv2_x_kernel(
    f: wp.array3d(dtype=float),
    d2f_dx2: wp.array3d(dtype=float),
    idx: float
):
    i, j, k = wp.tid()
    
    if i >= 2 and i < f.shape[0] - 2:
        d2f_dx2[i, j, k] = deriv2_x_4th(f, i, j, k, idx)


def test_derivatives():
    """Test derivative operators on analytic functions"""
    
    print("Testing 4th order finite difference derivatives")
    print("=" * 60)
    
    # Setup grid
    nx, ny, nz = 32, 32, 32
    xmin, xmax = -1.0, 1.0
    dx = (xmax - xmin) / (nx - 1)
    idx = 1.0 / dx
    
    # Create coordinate arrays
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(xmin, xmax, ny)
    z = np.linspace(xmin, xmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Test function: f(x,y,z) = sin(pi*x) * cos(pi*y)
    f_np = np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    # Analytical derivatives
    dfdx_exact = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)
    d2fdx2_exact = -np.pi**2 * np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    # Convert to warp arrays
    f = wp.array(f_np, dtype=wp.float32)
    df_dx = wp.zeros((nx, ny, nz), dtype=wp.float32)
    d2f_dx2 = wp.zeros((nx, ny, nz), dtype=wp.float32)
    
    # Compute derivatives
    wp.launch(compute_deriv_x_kernel, dim=(nx, ny, nz), inputs=[f, df_dx, idx])
    wp.launch(compute_deriv2_x_kernel, dim=(nx, ny, nz), inputs=[f, d2f_dx2, idx])
    
    # Convert back to numpy
    df_dx_np = df_dx.numpy()
    d2f_dx2_np = d2f_dx2.numpy()
    
    # Compute errors (exclude boundaries)
    interior = np.s_[3:-3, 3:-3, 3:-3]
    
    error_dx = np.abs(df_dx_np[interior] - dfdx_exact[interior])
    error_d2x = np.abs(d2f_dx2_np[interior] - d2fdx2_exact[interior])
    
    print(f"\nGrid: {nx} x {ny} x {nz}, dx = {dx:.4f}")
    print(f"\nFirst derivative df/dx:")
    print(f"  Max error: {np.max(error_dx):.6e}")
    print(f"  Mean error: {np.mean(error_dx):.6e}")
    print(f"  RMS error: {np.sqrt(np.mean(error_dx**2)):.6e}")
    
    print(f"\nSecond derivative d²f/dx²:")
    print(f"  Max error: {np.max(error_d2x):.6e}")
    print(f"  Mean error: {np.mean(error_d2x):.6e}")
    print(f"  RMS error: {np.sqrt(np.mean(error_d2x**2)):.6e}")
    
    # Check convergence order
    print("\n" + "=" * 60)
    print("Testing convergence order")
    print("=" * 60)
    
    resolutions = [16, 32, 64]
    errors = []
    
    for n in resolutions:
        dx_test = (xmax - xmin) / (n - 1)
        idx_test = 1.0 / dx_test
        
        x_test = np.linspace(xmin, xmax, n)
        X_test, Y_test, Z_test = np.meshgrid(x_test, x_test, x_test, indexing='ij')
        
        f_test = np.sin(np.pi * X_test) * np.cos(np.pi * Y_test)
        dfdx_test_exact = np.pi * np.cos(np.pi * X_test) * np.cos(np.pi * Y_test)
        
        f_wp = wp.array(f_test, dtype=wp.float32)
        df_dx_wp = wp.zeros((n, n, n), dtype=wp.float32)
        
        wp.launch(compute_deriv_x_kernel, dim=(n, n, n), inputs=[f_wp, df_dx_wp, idx_test])
        
        df_dx_test = df_dx_wp.numpy()
        interior_test = np.s_[3:-3, 3:-3, 3:-3]
        error = np.sqrt(np.mean((df_dx_test[interior_test] - dfdx_test_exact[interior_test])**2))
        errors.append(error)
        
        print(f"Resolution {n:3d}: RMS error = {error:.6e}")
    
    # Compute convergence rates
    if len(errors) >= 2:
        for i in range(1, len(errors)):
            ratio = errors[i-1] / errors[i]
            order = np.log2(ratio)
            print(f"Convergence rate {resolutions[i-1]}->{resolutions[i]}: {order:.2f} (expected: 4.0)")
    
    print("\n" + "=" * 60)
    print("Derivative tests complete")
    print("=" * 60)
    
    # Check if errors are reasonable (4th order should give small errors)
    if np.max(error_dx) < 1e-2 and np.max(error_d2x) < 1e-1:
        print("✓ Derivative accuracy acceptable")
        return True
    else:
        print("✗ Derivative errors too large")
        return False


if __name__ == "__main__":
    test_derivatives()
