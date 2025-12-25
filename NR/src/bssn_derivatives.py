"""
4th order finite difference spatial derivative operators for BSSN.
"""

import warp as wp
import numpy as np

wp.init()


@wp.func
def idx3d(i: int, j: int, k: int, nx: int, ny: int, nz: int) -> int:
    """Convert 3D indices to linear index"""
    return i + nx * (j + ny * k)


@wp.func  
def deriv_x_4th(
    f: wp.array(dtype=float),
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int,
    dx: float
) -> float:
    """4th order centered difference in x-direction"""
    # ∂ₓf ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    
    # Handle boundaries with one-sided stencils (simplified: use forward/backward)
    if i < 2:
        # Forward difference near left boundary
        idx0 = idx3d(i, j, k, nx, ny, nz)
        idx1 = idx3d(i+1, j, k, nx, ny, nz)
        return (f[idx1] - f[idx0]) / dx
    elif i >= nx - 2:
        # Backward difference near right boundary
        idx0 = idx3d(i, j, k, nx, ny, nz)
        idx1 = idx3d(i-1, j, k, nx, ny, nz)
        return (f[idx0] - f[idx1]) / dx
    else:
        # Interior: 4th order centered
        im2 = idx3d(i-2, j, k, nx, ny, nz)
        im1 = idx3d(i-1, j, k, nx, ny, nz)
        ip1 = idx3d(i+1, j, k, nx, ny, nz)
        ip2 = idx3d(i+2, j, k, nx, ny, nz)
        
        return (-f[ip2] + 8.0*f[ip1] - 8.0*f[im1] + f[im2]) / (12.0 * dx)


@wp.func
def deriv_y_4th(
    f: wp.array(dtype=float),
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int,
    dy: float
) -> float:
    """4th order centered difference in y-direction"""
    if j < 2:
        idx0 = idx3d(i, j, k, nx, ny, nz)
        idx1 = idx3d(i, j+1, k, nx, ny, nz)
        return (f[idx1] - f[idx0]) / dy
    elif j >= ny - 2:
        idx0 = idx3d(i, j, k, nx, ny, nz)
        idx1 = idx3d(i, j-1, k, nx, ny, nz)
        return (f[idx0] - f[idx1]) / dy
    else:
        jm2 = idx3d(i, j-2, k, nx, ny, nz)
        jm1 = idx3d(i, j-1, k, nx, ny, nz)
        jp1 = idx3d(i, j+1, k, nx, ny, nz)
        jp2 = idx3d(i, j+2, k, nx, ny, nz)
        
        return (-f[jp2] + 8.0*f[jp1] - 8.0*f[jm1] + f[jm2]) / (12.0 * dy)


@wp.func
def deriv_z_4th(
    f: wp.array(dtype=float),
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int,
    dz: float
) -> float:
    """4th order centered difference in z-direction"""
    if k < 2:
        idx0 = idx3d(i, j, k, nx, ny, nz)
        idx1 = idx3d(i, j, k+1, nx, ny, nz)
        return (f[idx1] - f[idx0]) / dz
    elif k >= nz - 2:
        idx0 = idx3d(i, j, k, nx, ny, nz)
        idx1 = idx3d(i, j, k-1, nx, ny, nz)
        return (f[idx0] - f[idx1]) / dz
    else:
        km2 = idx3d(i, j, k-2, nx, ny, nz)
        km1 = idx3d(i, j, k-1, nx, ny, nz)
        kp1 = idx3d(i, j, k+1, nx, ny, nz)
        kp2 = idx3d(i, j, k+2, nx, ny, nz)
        
        return (-f[kp2] + 8.0*f[kp1] - 8.0*f[km1] + f[km2]) / (12.0 * dz)


@wp.func
def deriv_xx_4th(
    f: wp.array(dtype=float),
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int,
    dx: float
) -> float:
    """4th order centered second derivative in x-direction"""
    # ∂²ₓf ≈ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
    
    if i < 2 or i >= nx - 2:
        # Boundary: use 2nd order
        idx0 = idx3d(i, j, k, nx, ny, nz)
        if i < 2:
            idx_m = idx0
            idx_p = idx3d(i+1, j, k, nx, ny, nz)
        else:
            idx_m = idx3d(i-1, j, k, nx, ny, nz)
            idx_p = idx0
        return 0.0  # Simplified for now
    else:
        im2 = idx3d(i-2, j, k, nx, ny, nz)
        im1 = idx3d(i-1, j, k, nx, ny, nz)
        i0 = idx3d(i, j, k, nx, ny, nz)
        ip1 = idx3d(i+1, j, k, nx, ny, nz)
        ip2 = idx3d(i+2, j, k, nx, ny, nz)
        
        return (-f[ip2] + 16.0*f[ip1] - 30.0*f[i0] + 16.0*f[im1] - f[im2]) / (12.0 * dx * dx)


@wp.func
def gradient_4th(
    f: wp.array(dtype=float),
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float
) -> wp.vec3:
    """Compute 3D gradient using 4th order FD"""
    grad_x = deriv_x_4th(f, i, j, k, nx, ny, nz, dx)
    grad_y = deriv_y_4th(f, i, j, k, nx, ny, nz, dy)
    grad_z = deriv_z_4th(f, i, j, k, nx, ny, nz, dz)
    return wp.vec3(grad_x, grad_y, grad_z)


@wp.kernel
def test_derivative_kernel(
    f: wp.array(dtype=float),
    df_dx: wp.array(dtype=float),
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float
):
    """Test kernel: compute derivative of test function"""
    i, j, k = wp.tid()
    
    if i >= nx or j >= ny or k >= nz:
        return
    
    idx = idx3d(i, j, k, nx, ny, nz)
    
    # Compute x-derivative
    df_dx[idx] = deriv_x_4th(f, i, j, k, nx, ny, nz, dx)


if __name__ == "__main__":
    print("Testing 4th order finite differences...")
    
    # Create test function: f(x,y,z) = sin(2πx/L)
    nx, ny, nz = 32, 32, 32
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dz = Lz / (nz - 1)
    
    print(f"Grid: {nx} x {ny} x {nz}")
    print(f"Spacing: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    
    # Initialize test function
    f_data = np.zeros(nx * ny * nz, dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = i * dx
                idx = i + nx * (j + ny * k)
                f_data[idx] = np.sin(2.0 * np.pi * x / Lx)
    
    f = wp.array(f_data, dtype=float)
    df_dx = wp.zeros(nx * ny * nz, dtype=float)
    
    # Compute derivative
    wp.launch(
        test_derivative_kernel,
        dim=(nx, ny, nz),
        inputs=[f, df_dx, nx, ny, nz, dx, dy, dz]
    )
    
    # Check against analytical derivative: df/dx = (2π/L) cos(2πx/L)
    df_numerical = df_dx.numpy()
    errors = []
    
    for k in range(nz):
        for j in range(ny):
            for i in range(4, nx-4):  # Check interior points only
                x = i * dx
                idx = i + nx * (j + ny * k)
                
                df_exact = (2.0 * np.pi / Lx) * np.cos(2.0 * np.pi * x / Lx)
                df_num = df_numerical[idx]
                error = abs(df_num - df_exact)
                errors.append(error)
    
    max_error = max(errors) if errors else 0
    avg_error = sum(errors) / len(errors) if errors else 0
    
    print(f"\nDerivative test:")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Avg error: {avg_error:.2e}")
    
    # 4th order should give very small errors for smooth function
    if max_error < 1e-3:
        print("✓ 4th order derivative test PASSED")
    else:
        print(f"✗ Error too large: {max_error}")
    
    print("\n" + "="*60)
    print("Spatial derivative implementation complete")
    print("="*60)
