"""
Fourth-Order Finite Difference Derivative Kernels for BSSN

Stencils:
- 1st derivative: D_0 = (-f_{i+2} + 8*f_{i+1} - 8*f_{i-1} + f_{i-2}) / (12*h)
- 2nd derivative: D_00 = (-f_{i+2} + 16*f_{i+1} - 30*f_i + 16*f_{i-1} - f_{i-2}) / (12*h^2)
- Kreiss-Oliger dissipation (6th order): D_KO = (-f_{i+3} + 6*f_{i+2} - 15*f_{i+1} + 20*f_i - 15*f_{i-1} + 6*f_{i-2} - f_{i-3}) / 64
"""

import warp as wp
import numpy as np

wp.init()


# ============ First Derivatives (4th order) ============

@wp.func
def d1_x(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_12h: float) -> float:
    """4th order first derivative in x direction"""
    return inv_12h * (
        -f[i+2, j, k] + 8.0*f[i+1, j, k] - 8.0*f[i-1, j, k] + f[i-2, j, k]
    )


@wp.func
def d1_y(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_12h: float) -> float:
    """4th order first derivative in y direction"""
    return inv_12h * (
        -f[i, j+2, k] + 8.0*f[i, j+1, k] - 8.0*f[i, j-1, k] + f[i, j-2, k]
    )


@wp.func
def d1_z(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_12h: float) -> float:
    """4th order first derivative in z direction"""
    return inv_12h * (
        -f[i, j, k+2] + 8.0*f[i, j, k+1] - 8.0*f[i, j, k-1] + f[i, j, k-2]
    )


# ============ Second Derivatives (4th order) ============

@wp.func
def d2_xx(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_12h2: float) -> float:
    """4th order second derivative in x direction"""
    return inv_12h2 * (
        -f[i+2, j, k] + 16.0*f[i+1, j, k] - 30.0*f[i, j, k] + 16.0*f[i-1, j, k] - f[i-2, j, k]
    )


@wp.func
def d2_yy(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_12h2: float) -> float:
    """4th order second derivative in y direction"""
    return inv_12h2 * (
        -f[i, j+2, k] + 16.0*f[i, j+1, k] - 30.0*f[i, j, k] + 16.0*f[i, j-1, k] - f[i, j-2, k]
    )


@wp.func
def d2_zz(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_12h2: float) -> float:
    """4th order second derivative in z direction"""
    return inv_12h2 * (
        -f[i, j, k+2] + 16.0*f[i, j, k+1] - 30.0*f[i, j, k] + 16.0*f[i, j, k-1] - f[i, j, k-2]
    )


@wp.func
def d2_xy(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_144h2: float) -> float:
    """4th order mixed derivative d^2/dxdy"""
    # Apply d1_x to d1_y or equivalently use the 2D stencil
    return inv_144h2 * (
        -(-f[i+2, j+2, k] + 8.0*f[i+1, j+2, k] - 8.0*f[i-1, j+2, k] + f[i-2, j+2, k])
        + 8.0*(-f[i+2, j+1, k] + 8.0*f[i+1, j+1, k] - 8.0*f[i-1, j+1, k] + f[i-2, j+1, k])
        - 8.0*(-f[i+2, j-1, k] + 8.0*f[i+1, j-1, k] - 8.0*f[i-1, j-1, k] + f[i-2, j-1, k])
        + (-f[i+2, j-2, k] + 8.0*f[i+1, j-2, k] - 8.0*f[i-1, j-2, k] + f[i-2, j-2, k])
    )


@wp.func
def d2_xz(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_144h2: float) -> float:
    """4th order mixed derivative d^2/dxdz"""
    return inv_144h2 * (
        -(-f[i+2, j, k+2] + 8.0*f[i+1, j, k+2] - 8.0*f[i-1, j, k+2] + f[i-2, j, k+2])
        + 8.0*(-f[i+2, j, k+1] + 8.0*f[i+1, j, k+1] - 8.0*f[i-1, j, k+1] + f[i-2, j, k+1])
        - 8.0*(-f[i+2, j, k-1] + 8.0*f[i+1, j, k-1] - 8.0*f[i-1, j, k-1] + f[i-2, j, k-1])
        + (-f[i+2, j, k-2] + 8.0*f[i+1, j, k-2] - 8.0*f[i-1, j, k-2] + f[i-2, j, k-2])
    )


@wp.func
def d2_yz(f: wp.array3d(dtype=float), i: int, j: int, k: int, inv_144h2: float) -> float:
    """4th order mixed derivative d^2/dydz"""
    return inv_144h2 * (
        -(-f[i, j+2, k+2] + 8.0*f[i, j+1, k+2] - 8.0*f[i, j-1, k+2] + f[i, j-2, k+2])
        + 8.0*(-f[i, j+2, k+1] + 8.0*f[i, j+1, k+1] - 8.0*f[i, j-1, k+1] + f[i, j-2, k+1])
        - 8.0*(-f[i, j+2, k-1] + 8.0*f[i, j+1, k-1] - 8.0*f[i, j-1, k-1] + f[i, j-2, k-1])
        + (-f[i, j+2, k-2] + 8.0*f[i, j+1, k-2] - 8.0*f[i, j-1, k-2] + f[i, j-2, k-2])
    )


# ============ Kreiss-Oliger Dissipation (6th order) ============

@wp.func
def ko_diss_x(f: wp.array3d(dtype=float), i: int, j: int, k: int) -> float:
    """6th order Kreiss-Oliger dissipation in x"""
    return (
        -f[i+3, j, k] + 6.0*f[i+2, j, k] - 15.0*f[i+1, j, k] 
        + 20.0*f[i, j, k] 
        - 15.0*f[i-1, j, k] + 6.0*f[i-2, j, k] - f[i-3, j, k]
    ) / 64.0


@wp.func
def ko_diss_y(f: wp.array3d(dtype=float), i: int, j: int, k: int) -> float:
    """6th order Kreiss-Oliger dissipation in y"""
    return (
        -f[i, j+3, k] + 6.0*f[i, j+2, k] - 15.0*f[i, j+1, k] 
        + 20.0*f[i, j, k] 
        - 15.0*f[i, j-1, k] + 6.0*f[i, j-2, k] - f[i, j-3, k]
    ) / 64.0


@wp.func
def ko_diss_z(f: wp.array3d(dtype=float), i: int, j: int, k: int) -> float:
    """6th order Kreiss-Oliger dissipation in z"""
    return (
        -f[i, j, k+3] + 6.0*f[i, j, k+2] - 15.0*f[i, j, k+1] 
        + 20.0*f[i, j, k] 
        - 15.0*f[i, j, k-1] + 6.0*f[i, j, k-2] - f[i, j, k-3]
    ) / 64.0


@wp.func
def ko_dissipation(f: wp.array3d(dtype=float), i: int, j: int, k: int, 
                   sigma_over_dt: float) -> float:
    """Total Kreiss-Oliger dissipation term (to be subtracted from RHS)"""
    return sigma_over_dt * (
        ko_diss_x(f, i, j, k) + ko_diss_y(f, i, j, k) + ko_diss_z(f, i, j, k)
    )


# ============ Test Kernels ============

@wp.kernel
def test_derivative_kernel(
    f: wp.array3d(dtype=float),
    df_x: wp.array3d(dtype=float),
    df_y: wp.array3d(dtype=float),
    df_z: wp.array3d(dtype=float),
    d2f_xx: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    inv_12h: float,
    inv_12h2: float
):
    """Test kernel to compute derivatives"""
    i, j, k = wp.tid()
    
    # Skip boundary points (need 2 ghost zones for 4th order)
    if i < 2 or i >= nx - 2:
        return
    if j < 2 or j >= ny - 2:
        return
    if k < 2 or k >= nz - 2:
        return
    
    df_x[i, j, k] = d1_x(f, i, j, k, inv_12h)
    df_y[i, j, k] = d1_y(f, i, j, k, inv_12h)
    df_z[i, j, k] = d1_z(f, i, j, k, inv_12h)
    d2f_xx[i, j, k] = d2_xx(f, i, j, k, inv_12h2)


def test_derivatives():
    """Test derivative accuracy on a known function"""
    print("Testing 4th order finite differences...")
    
    # Grid parameters
    n = 32
    L = 2.0 * np.pi
    h = L / n
    
    # Create coordinate arrays
    x = np.linspace(0, L - h, n)
    y = np.linspace(0, L - h, n)
    z = np.linspace(0, L - h, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Test function: sin(x)*cos(y)*sin(z)
    f_np = np.sin(X) * np.cos(Y) * np.sin(Z)
    
    # Analytical derivatives
    df_x_exact = np.cos(X) * np.cos(Y) * np.sin(Z)
    df_y_exact = -np.sin(X) * np.sin(Y) * np.sin(Z)
    df_z_exact = np.sin(X) * np.cos(Y) * np.cos(Z)
    d2f_xx_exact = -np.sin(X) * np.cos(Y) * np.sin(Z)
    
    # Create warp arrays
    f = wp.array(f_np.astype(np.float32), dtype=float, device="cpu")
    df_x = wp.zeros_like(f)
    df_y = wp.zeros_like(f)
    df_z = wp.zeros_like(f)
    d2f_xx = wp.zeros_like(f)
    
    inv_12h = 1.0 / (12.0 * h)
    inv_12h2 = 1.0 / (12.0 * h * h)
    
    # Launch kernel
    wp.launch(
        test_derivative_kernel,
        dim=(n, n, n),
        inputs=[f, df_x, df_y, df_z, d2f_xx, n, n, n, inv_12h, inv_12h2]
    )
    
    # Check accuracy (interior points only)
    margin = 2
    sl = slice(margin, n - margin)
    
    df_x_np = df_x.numpy()[sl, sl, sl]
    df_x_exact_int = df_x_exact[sl, sl, sl]
    error_dx = np.max(np.abs(df_x_np - df_x_exact_int))
    
    df_y_np = df_y.numpy()[sl, sl, sl]
    df_y_exact_int = df_y_exact[sl, sl, sl]
    error_dy = np.max(np.abs(df_y_np - df_y_exact_int))
    
    df_z_np = df_z.numpy()[sl, sl, sl]
    df_z_exact_int = df_z_exact[sl, sl, sl]
    error_dz = np.max(np.abs(df_z_np - df_z_exact_int))
    
    d2f_xx_np = d2f_xx.numpy()[sl, sl, sl]
    d2f_xx_exact_int = d2f_xx_exact[sl, sl, sl]
    error_d2x = np.max(np.abs(d2f_xx_np - d2f_xx_exact_int))
    
    print(f"Grid: {n}^3, h = {h:.4f}")
    print(f"Max error df/dx: {error_dx:.6e}")
    print(f"Max error df/dy: {error_dy:.6e}")
    print(f"Max error df/dz: {error_dz:.6e}")
    print(f"Max error d2f/dx2: {error_d2x:.6e}")
    
    # Expected 4th order accuracy: error ~ O(h^4)
    expected_order = h**4
    print(f"Expected O(h^4) ~ {expected_order:.6e}")
    
    if error_dx < 1e-4 and error_dy < 1e-4 and error_dz < 1e-4 and error_d2x < 1e-4:
        print("Derivative test PASSED!")
    else:
        print("Derivative test FAILED - errors too large")
    
    return error_dx, error_dy, error_dz, error_d2x


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_derivatives()
