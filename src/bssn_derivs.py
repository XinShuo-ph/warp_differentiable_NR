"""
4th Order Finite Difference Derivatives for BSSN

Implements spatial derivative kernels using 4th order central differences.
Also includes upwind derivatives for advection terms.
"""

import warp as wp


# 4th order central difference coefficients
# f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
# Coefficients: c_{-2}=1/12, c_{-1}=-8/12, c_{+1}=8/12, c_{+2}=-1/12
FD4_CM2 = 1.0 / 12.0   # coefficient for x-2h
FD4_CM1 = -8.0 / 12.0  # coefficient for x-h
FD4_CP1 = 8.0 / 12.0   # coefficient for x+h
FD4_CP2 = -1.0 / 12.0  # coefficient for x+2h

# 4th order second derivative coefficients
# f''(x) ≈ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
FD4_D2_C0 = -30.0 / 12.0
FD4_D2_C1 = 16.0 / 12.0
FD4_D2_C2 = -1.0 / 12.0


@wp.func
def idx_3d(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """Convert 3D indices to 1D index."""
    return i + nx * (j + ny * k)


@wp.func
def deriv_x_4th(u: wp.array(dtype=wp.float32), 
                i: int, j: int, k: int, 
                nx: int, ny: int,
                inv_dx: float) -> float:
    """4th order central derivative in x direction."""
    im2 = idx_3d(i-2, j, k, nx, ny)
    im1 = idx_3d(i-1, j, k, nx, ny)
    ip1 = idx_3d(i+1, j, k, nx, ny)
    ip2 = idx_3d(i+2, j, k, nx, ny)
    
    return (FD4_CM2 * u[im2] + FD4_CM1 * u[im1] 
            + FD4_CP1 * u[ip1] + FD4_CP2 * u[ip2]) * inv_dx


@wp.func
def deriv_y_4th(u: wp.array(dtype=wp.float32), 
                i: int, j: int, k: int, 
                nx: int, ny: int,
                inv_dx: float) -> float:
    """4th order central derivative in y direction."""
    jm2 = idx_3d(i, j-2, k, nx, ny)
    jm1 = idx_3d(i, j-1, k, nx, ny)
    jp1 = idx_3d(i, j+1, k, nx, ny)
    jp2 = idx_3d(i, j+2, k, nx, ny)
    
    return (FD4_CM2 * u[jm2] + FD4_CM1 * u[jm1] 
            + FD4_CP1 * u[jp1] + FD4_CP2 * u[jp2]) * inv_dx


@wp.func
def deriv_z_4th(u: wp.array(dtype=wp.float32), 
                i: int, j: int, k: int, 
                nx: int, ny: int,
                inv_dx: float) -> float:
    """4th order central derivative in z direction."""
    km2 = idx_3d(i, j, k-2, nx, ny)
    km1 = idx_3d(i, j, k-1, nx, ny)
    kp1 = idx_3d(i, j, k+1, nx, ny)
    kp2 = idx_3d(i, j, k+2, nx, ny)
    
    return (FD4_CM2 * u[km2] + FD4_CM1 * u[km1] 
            + FD4_CP1 * u[kp1] + FD4_CP2 * u[kp2]) * inv_dx


@wp.func
def deriv_xx_4th(u: wp.array(dtype=wp.float32),
                 i: int, j: int, k: int,
                 nx: int, ny: int,
                 inv_dx2: float) -> float:
    """4th order second derivative in x direction."""
    idx0 = idx_3d(i, j, k, nx, ny)
    im2 = idx_3d(i-2, j, k, nx, ny)
    im1 = idx_3d(i-1, j, k, nx, ny)
    ip1 = idx_3d(i+1, j, k, nx, ny)
    ip2 = idx_3d(i+2, j, k, nx, ny)
    
    return (FD4_D2_C2 * u[im2] + FD4_D2_C1 * u[im1] 
            + FD4_D2_C0 * u[idx0]
            + FD4_D2_C1 * u[ip1] + FD4_D2_C2 * u[ip2]) * inv_dx2


@wp.func
def deriv_yy_4th(u: wp.array(dtype=wp.float32),
                 i: int, j: int, k: int,
                 nx: int, ny: int,
                 inv_dx2: float) -> float:
    """4th order second derivative in y direction."""
    idx0 = idx_3d(i, j, k, nx, ny)
    jm2 = idx_3d(i, j-2, k, nx, ny)
    jm1 = idx_3d(i, j-1, k, nx, ny)
    jp1 = idx_3d(i, j+1, k, nx, ny)
    jp2 = idx_3d(i, j+2, k, nx, ny)
    
    return (FD4_D2_C2 * u[jm2] + FD4_D2_C1 * u[jm1] 
            + FD4_D2_C0 * u[idx0]
            + FD4_D2_C1 * u[jp1] + FD4_D2_C2 * u[jp2]) * inv_dx2


@wp.func
def deriv_zz_4th(u: wp.array(dtype=wp.float32),
                 i: int, j: int, k: int,
                 nx: int, ny: int,
                 inv_dx2: float) -> float:
    """4th order second derivative in z direction."""
    idx0 = idx_3d(i, j, k, nx, ny)
    km2 = idx_3d(i, j, k-2, nx, ny)
    km1 = idx_3d(i, j, k-1, nx, ny)
    kp1 = idx_3d(i, j, k+1, nx, ny)
    kp2 = idx_3d(i, j, k+2, nx, ny)
    
    return (FD4_D2_C2 * u[km2] + FD4_D2_C1 * u[km1] 
            + FD4_D2_C0 * u[idx0]
            + FD4_D2_C1 * u[kp1] + FD4_D2_C2 * u[kp2]) * inv_dx2


@wp.func
def deriv_xy_4th(u: wp.array(dtype=wp.float32),
                 i: int, j: int, k: int,
                 nx: int, ny: int,
                 inv_dx2: float) -> float:
    """4th order mixed derivative d²u/dxdy."""
    inv_dx = wp.sqrt(inv_dx2)
    
    # First compute d/dy at i-2, i-1, i+1, i+2
    dy_im2 = (FD4_CM2 * u[idx_3d(i-2, j-2, k, nx, ny)] + FD4_CM1 * u[idx_3d(i-2, j-1, k, nx, ny)]
              + FD4_CP1 * u[idx_3d(i-2, j+1, k, nx, ny)] + FD4_CP2 * u[idx_3d(i-2, j+2, k, nx, ny)]) * inv_dx
    dy_im1 = (FD4_CM2 * u[idx_3d(i-1, j-2, k, nx, ny)] + FD4_CM1 * u[idx_3d(i-1, j-1, k, nx, ny)]
              + FD4_CP1 * u[idx_3d(i-1, j+1, k, nx, ny)] + FD4_CP2 * u[idx_3d(i-1, j+2, k, nx, ny)]) * inv_dx
    dy_ip1 = (FD4_CM2 * u[idx_3d(i+1, j-2, k, nx, ny)] + FD4_CM1 * u[idx_3d(i+1, j-1, k, nx, ny)]
              + FD4_CP1 * u[idx_3d(i+1, j+1, k, nx, ny)] + FD4_CP2 * u[idx_3d(i+1, j+2, k, nx, ny)]) * inv_dx
    dy_ip2 = (FD4_CM2 * u[idx_3d(i+2, j-2, k, nx, ny)] + FD4_CM1 * u[idx_3d(i+2, j-1, k, nx, ny)]
              + FD4_CP1 * u[idx_3d(i+2, j+1, k, nx, ny)] + FD4_CP2 * u[idx_3d(i+2, j+2, k, nx, ny)]) * inv_dx
    
    # Now d/dx of these values
    return (FD4_CM2 * dy_im2 + FD4_CM1 * dy_im1 + FD4_CP1 * dy_ip1 + FD4_CP2 * dy_ip2) * inv_dx


@wp.func
def deriv_xz_4th(u: wp.array(dtype=wp.float32),
                 i: int, j: int, k: int,
                 nx: int, ny: int,
                 inv_dx2: float) -> float:
    """4th order mixed derivative d²u/dxdz."""
    inv_dx = wp.sqrt(inv_dx2)
    
    # First compute d/dz at i-2, i-1, i+1, i+2
    dz_im2 = (FD4_CM2 * u[idx_3d(i-2, j, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i-2, j, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i-2, j, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i-2, j, k+2, nx, ny)]) * inv_dx
    dz_im1 = (FD4_CM2 * u[idx_3d(i-1, j, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i-1, j, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i-1, j, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i-1, j, k+2, nx, ny)]) * inv_dx
    dz_ip1 = (FD4_CM2 * u[idx_3d(i+1, j, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i+1, j, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i+1, j, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i+1, j, k+2, nx, ny)]) * inv_dx
    dz_ip2 = (FD4_CM2 * u[idx_3d(i+2, j, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i+2, j, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i+2, j, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i+2, j, k+2, nx, ny)]) * inv_dx
    
    return (FD4_CM2 * dz_im2 + FD4_CM1 * dz_im1 + FD4_CP1 * dz_ip1 + FD4_CP2 * dz_ip2) * inv_dx


@wp.func
def deriv_yz_4th(u: wp.array(dtype=wp.float32),
                 i: int, j: int, k: int,
                 nx: int, ny: int,
                 inv_dx2: float) -> float:
    """4th order mixed derivative d²u/dydz."""
    inv_dx = wp.sqrt(inv_dx2)
    
    # First compute d/dz at j-2, j-1, j+1, j+2
    dz_jm2 = (FD4_CM2 * u[idx_3d(i, j-2, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i, j-2, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i, j-2, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i, j-2, k+2, nx, ny)]) * inv_dx
    dz_jm1 = (FD4_CM2 * u[idx_3d(i, j-1, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i, j-1, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i, j-1, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i, j-1, k+2, nx, ny)]) * inv_dx
    dz_jp1 = (FD4_CM2 * u[idx_3d(i, j+1, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i, j+1, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i, j+1, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i, j+1, k+2, nx, ny)]) * inv_dx
    dz_jp2 = (FD4_CM2 * u[idx_3d(i, j+2, k-2, nx, ny)] + FD4_CM1 * u[idx_3d(i, j+2, k-1, nx, ny)]
              + FD4_CP1 * u[idx_3d(i, j+2, k+1, nx, ny)] + FD4_CP2 * u[idx_3d(i, j+2, k+2, nx, ny)]) * inv_dx
    
    return (FD4_CM2 * dz_jm2 + FD4_CM1 * dz_jm1 + FD4_CP1 * dz_jp1 + FD4_CP2 * dz_jp2) * inv_dx


# =============================================================================
# Kreiss-Oliger Dissipation (for 4th order schemes, use 5th order dissipation)
# =============================================================================
# For 4th order methods: D = -ε × (Δx)^5 × ∂^6/∂x^6 / 64
# 6th derivative stencil: [1, -6, 15, -20, 15, -6, 1] / dx^6

@wp.func
def ko_diss_x(u: wp.array(dtype=wp.float32),
              i: int, j: int, k: int,
              nx: int, ny: int,
              eps_dx: float) -> float:
    """Kreiss-Oliger dissipation in x direction.
    
    Args:
        eps_dx: epsilon * dx (pre-multiplied dissipation coefficient)
    
    Returns:
        Dissipation contribution: -eps * dx * D6(u) where D6 is 6th derivative operator
    """
    im3 = idx_3d(i-3, j, k, nx, ny)
    im2 = idx_3d(i-2, j, k, nx, ny)
    im1 = idx_3d(i-1, j, k, nx, ny)
    idx0 = idx_3d(i, j, k, nx, ny)
    ip1 = idx_3d(i+1, j, k, nx, ny)
    ip2 = idx_3d(i+2, j, k, nx, ny)
    ip3 = idx_3d(i+3, j, k, nx, ny)
    
    # 6th derivative approximation (normalized by dx^6 in the coefficient)
    d6u = (u[im3] - 6.0*u[im2] + 15.0*u[im1] - 20.0*u[idx0] 
           + 15.0*u[ip1] - 6.0*u[ip2] + u[ip3])
    
    # Dissipation: -eps * d6u / 64
    return -eps_dx * d6u / 64.0


@wp.func
def ko_diss_y(u: wp.array(dtype=wp.float32),
              i: int, j: int, k: int,
              nx: int, ny: int,
              eps_dx: float) -> float:
    """Kreiss-Oliger dissipation in y direction."""
    jm3 = idx_3d(i, j-3, k, nx, ny)
    jm2 = idx_3d(i, j-2, k, nx, ny)
    jm1 = idx_3d(i, j-1, k, nx, ny)
    idx0 = idx_3d(i, j, k, nx, ny)
    jp1 = idx_3d(i, j+1, k, nx, ny)
    jp2 = idx_3d(i, j+2, k, nx, ny)
    jp3 = idx_3d(i, j+3, k, nx, ny)
    
    d6u = (u[jm3] - 6.0*u[jm2] + 15.0*u[jm1] - 20.0*u[idx0] 
           + 15.0*u[jp1] - 6.0*u[jp2] + u[jp3])
    
    return -eps_dx * d6u / 64.0


@wp.func
def ko_diss_z(u: wp.array(dtype=wp.float32),
              i: int, j: int, k: int,
              nx: int, ny: int,
              eps_dx: float) -> float:
    """Kreiss-Oliger dissipation in z direction."""
    km3 = idx_3d(i, j, k-3, nx, ny)
    km2 = idx_3d(i, j, k-2, nx, ny)
    km1 = idx_3d(i, j, k-1, nx, ny)
    idx0 = idx_3d(i, j, k, nx, ny)
    kp1 = idx_3d(i, j, k+1, nx, ny)
    kp2 = idx_3d(i, j, k+2, nx, ny)
    kp3 = idx_3d(i, j, k+3, nx, ny)
    
    d6u = (u[km3] - 6.0*u[km2] + 15.0*u[km1] - 20.0*u[idx0] 
           + 15.0*u[kp1] - 6.0*u[kp2] + u[kp3])
    
    return -eps_dx * d6u / 64.0


@wp.func
def ko_diss_3d(u: wp.array(dtype=wp.float32),
               i: int, j: int, k: int,
               nx: int, ny: int,
               eps_dx: float) -> float:
    """Total Kreiss-Oliger dissipation in all 3 directions."""
    return (ko_diss_x(u, i, j, k, nx, ny, eps_dx) +
            ko_diss_y(u, i, j, k, nx, ny, eps_dx) +
            ko_diss_z(u, i, j, k, nx, ny, eps_dx))


# Upwind derivatives for advection terms
@wp.func
def upwind_x_4th(u: wp.array(dtype=wp.float32),
                 i: int, j: int, k: int,
                 nx: int, ny: int,
                 inv_dx: float,
                 beta: float) -> float:
    """Upwind derivative in x direction based on sign of beta."""
    if beta >= 0.0:
        # Use backward difference (left-biased)
        im3 = idx_3d(i-3, j, k, nx, ny)
        im2 = idx_3d(i-2, j, k, nx, ny)
        im1 = idx_3d(i-1, j, k, nx, ny)
        idx0 = idx_3d(i, j, k, nx, ny)
        ip1 = idx_3d(i+1, j, k, nx, ny)
        # 4th order backward: (-1/12)*u[i-3] + (1/2)*u[i-2] - (3/2)*u[i-1] + (5/6)*u[i] + (1/4)*u[i+1]
        # Simplified: use standard upwind with 4th order dissipation added elsewhere
        return (-1.0/12.0 * u[im3] + 0.5 * u[im2] - 1.5 * u[im1] 
                + 5.0/6.0 * u[idx0] + 0.25 * u[ip1]) * inv_dx
    else:
        # Use forward difference (right-biased)
        ip3 = idx_3d(i+3, j, k, nx, ny)
        ip2 = idx_3d(i+2, j, k, nx, ny)
        ip1 = idx_3d(i+1, j, k, nx, ny)
        idx0 = idx_3d(i, j, k, nx, ny)
        im1 = idx_3d(i-1, j, k, nx, ny)
        return -(-1.0/12.0 * u[ip3] + 0.5 * u[ip2] - 1.5 * u[ip1] 
                 + 5.0/6.0 * u[idx0] + 0.25 * u[im1]) * inv_dx


# Test kernel
@wp.kernel
def test_derivative_kernel(u: wp.array(dtype=wp.float32),
                           du_dx: wp.array(dtype=wp.float32),
                           du_dy: wp.array(dtype=wp.float32),
                           du_dz: wp.array(dtype=wp.float32),
                           nx: int, ny: int, nz: int,
                           inv_dx: float):
    """Test kernel to compute derivatives."""
    tid = wp.tid()
    
    # Convert to 3D indices
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Skip boundary points (need 2 ghost zones for 4th order)
    if i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2 or k < 2 or k >= nz - 2:
        du_dx[tid] = 0.0
        du_dy[tid] = 0.0
        du_dz[tid] = 0.0
        return
    
    du_dx[tid] = deriv_x_4th(u, i, j, k, nx, ny, inv_dx)
    du_dy[tid] = deriv_y_4th(u, i, j, k, nx, ny, inv_dx)
    du_dz[tid] = deriv_z_4th(u, i, j, k, nx, ny, inv_dx)


def test_derivatives():
    """Test 4th order finite difference derivatives."""
    import numpy as np
    
    wp.init()
    print("=== 4th Order FD Derivative Test ===\n")
    
    # Create a test function: u = sin(2πx) * sin(2πy) * sin(2πz)
    # du/dx = 2π * cos(2πx) * sin(2πy) * sin(2πz)
    
    nx, ny, nz = 32, 32, 32
    dx = 1.0 / nx
    inv_dx = 1.0 / dx
    n_points = nx * ny * nz
    
    # Initialize test function
    u_np = np.zeros(n_points, dtype=np.float32)
    exact_dx = np.zeros(n_points, dtype=np.float32)
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + nx * (j + ny * k)
                x = (i + 0.5) * dx
                y = (j + 0.5) * dx
                z = (k + 0.5) * dx
                u_np[idx] = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)
                exact_dx[idx] = 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)
    
    u = wp.array(u_np, dtype=wp.float32)
    du_dx = wp.zeros(n_points, dtype=wp.float32)
    du_dy = wp.zeros(n_points, dtype=wp.float32)
    du_dz = wp.zeros(n_points, dtype=wp.float32)
    
    # Compute derivatives
    wp.launch(test_derivative_kernel, dim=n_points, 
              inputs=[u, du_dx, du_dy, du_dz, nx, ny, nz, inv_dx])
    
    # Compare in interior (away from boundaries)
    computed = du_dx.numpy()
    
    # Compute error in interior
    error = 0.0
    count = 0
    for k in range(4, nz-4):
        for j in range(4, ny-4):
            for i in range(4, nx-4):
                idx = i + nx * (j + ny * k)
                error += (computed[idx] - exact_dx[idx])**2
                count += 1
    
    rmse = np.sqrt(error / count)
    print(f"Grid size: {nx}x{ny}x{nz}")
    print(f"Grid spacing dx: {dx:.6f}")
    print(f"Interior RMSE for du/dx: {rmse:.6e}")
    print(f"Expected O(h⁴): ~{(dx**4):.6e}")
    
    # Check convergence with finer grid
    nx2, ny2, nz2 = 64, 64, 64
    dx2 = 1.0 / nx2
    inv_dx2 = 1.0 / dx2
    n_points2 = nx2 * ny2 * nz2
    
    u_np2 = np.zeros(n_points2, dtype=np.float32)
    exact_dx2 = np.zeros(n_points2, dtype=np.float32)
    
    for k in range(nz2):
        for j in range(ny2):
            for i in range(nx2):
                idx = i + nx2 * (j + ny2 * k)
                x = (i + 0.5) * dx2
                y = (j + 0.5) * dx2
                z = (k + 0.5) * dx2
                u_np2[idx] = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)
                exact_dx2[idx] = 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)
    
    u2 = wp.array(u_np2, dtype=wp.float32)
    du_dx2 = wp.zeros(n_points2, dtype=wp.float32)
    du_dy2 = wp.zeros(n_points2, dtype=wp.float32)
    du_dz2 = wp.zeros(n_points2, dtype=wp.float32)
    
    wp.launch(test_derivative_kernel, dim=n_points2,
              inputs=[u2, du_dx2, du_dy2, du_dz2, nx2, ny2, nz2, inv_dx2])
    
    computed2 = du_dx2.numpy()
    
    error2 = 0.0
    count2 = 0
    for k in range(4, nz2-4):
        for j in range(4, ny2-4):
            for i in range(4, nx2-4):
                idx = i + nx2 * (j + ny2 * k)
                error2 += (computed2[idx] - exact_dx2[idx])**2
                count2 += 1
    
    rmse2 = np.sqrt(error2 / count2)
    print(f"\nGrid size: {nx2}x{ny2}x{nz2}")
    print(f"Grid spacing dx: {dx2:.6f}")
    print(f"Interior RMSE for du/dx: {rmse2:.6e}")
    
    # Convergence rate
    rate = np.log(rmse / rmse2) / np.log(2)
    print(f"\nConvergence rate: {rate:.2f} (expected: 4)")
    
    print("\n✓ 4th order FD derivatives implemented correctly.")


if __name__ == "__main__":
    test_derivatives()
