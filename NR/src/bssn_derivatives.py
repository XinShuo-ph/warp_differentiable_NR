"""
4th-order finite difference derivative kernels for BSSN.

Implements centered and upwind derivatives on a uniform 3D grid.
"""

import warp as wp


@wp.func
def deriv_x_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idx: float
) -> float:
    """
    4th order centered derivative in x-direction.
    
    Stencil: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)
    """
    return idx * (
        -f[i+2, j, k] + 8.0 * f[i+1, j, k] 
        - 8.0 * f[i-1, j, k] + f[i-2, j, k]
    ) / 12.0


@wp.func
def deriv_y_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idy: float
) -> float:
    """4th order centered derivative in y-direction"""
    return idy * (
        -f[i, j+2, k] + 8.0 * f[i, j+1, k]
        - 8.0 * f[i, j-1, k] + f[i, j-2, k]
    ) / 12.0


@wp.func
def deriv_z_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idz: float
) -> float:
    """4th order centered derivative in z-direction"""
    return idz * (
        -f[i, j, k+2] + 8.0 * f[i, j, k+1]
        - 8.0 * f[i, j, k-1] + f[i, j, k-2]
    ) / 12.0


@wp.func
def deriv2_x_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idx: float
) -> float:
    """
    4th order second derivative in x-direction.
    
    Stencil: (-f[i+2] + 16*f[i+1] - 30*f[i] + 16*f[i-1] - f[i-2]) / (12*dx^2)
    """
    return (idx * idx) * (
        -f[i+2, j, k] + 16.0 * f[i+1, j, k] - 30.0 * f[i, j, k]
        + 16.0 * f[i-1, j, k] - f[i-2, j, k]
    ) / 12.0


@wp.func
def deriv2_y_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idy: float
) -> float:
    """4th order second derivative in y-direction"""
    return (idy * idy) * (
        -f[i, j+2, k] + 16.0 * f[i, j+1, k] - 30.0 * f[i, j, k]
        + 16.0 * f[i, j-1, k] - f[i, j-2, k]
    ) / 12.0


@wp.func
def deriv2_z_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idz: float
) -> float:
    """4th order second derivative in z-direction"""
    return (idz * idz) * (
        -f[i, j, k+2] + 16.0 * f[i, j, k+1] - 30.0 * f[i, j, k]
        + 16.0 * f[i, j, k-1] - f[i, j, k-2]
    ) / 12.0


@wp.func
def deriv2_xy_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idx: float,
    idy: float
) -> float:
    """
    4th order mixed derivative d²f/dxdy.
    
    Apply first d/dx, then d/dy to the result.
    """
    # df/dx at j-2, j-1, j+1, j+2
    dfx_jm2 = idx * (-f[i+2, j-2, k] + 8.0 * f[i+1, j-2, k] - 8.0 * f[i-1, j-2, k] + f[i-2, j-2, k]) / 12.0
    dfx_jm1 = idx * (-f[i+2, j-1, k] + 8.0 * f[i+1, j-1, k] - 8.0 * f[i-1, j-1, k] + f[i-2, j-1, k]) / 12.0
    dfx_jp1 = idx * (-f[i+2, j+1, k] + 8.0 * f[i+1, j+1, k] - 8.0 * f[i-1, j+1, k] + f[i-2, j+1, k]) / 12.0
    dfx_jp2 = idx * (-f[i+2, j+2, k] + 8.0 * f[i+1, j+2, k] - 8.0 * f[i-1, j+2, k] + f[i-2, j+2, k]) / 12.0
    
    # Now d/dy of df/dx
    return idy * (-dfx_jp2 + 8.0 * dfx_jp1 - 8.0 * dfx_jm1 + dfx_jm2) / 12.0


@wp.func
def deriv2_xz_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idx: float,
    idz: float
) -> float:
    """4th order mixed derivative d²f/dxdz"""
    # df/dx at k-2, k-1, k+1, k+2
    dfx_km2 = idx * (-f[i+2, j, k-2] + 8.0 * f[i+1, j, k-2] - 8.0 * f[i-1, j, k-2] + f[i-2, j, k-2]) / 12.0
    dfx_km1 = idx * (-f[i+2, j, k-1] + 8.0 * f[i+1, j, k-1] - 8.0 * f[i-1, j, k-1] + f[i-2, j, k-1]) / 12.0
    dfx_kp1 = idx * (-f[i+2, j, k+1] + 8.0 * f[i+1, j, k+1] - 8.0 * f[i-1, j, k+1] + f[i-2, j, k+1]) / 12.0
    dfx_kp2 = idx * (-f[i+2, j, k+2] + 8.0 * f[i+1, j, k+2] - 8.0 * f[i-1, j, k+2] + f[i-2, j, k+2]) / 12.0
    
    return idz * (-dfx_kp2 + 8.0 * dfx_kp1 - 8.0 * dfx_km1 + dfx_km2) / 12.0


@wp.func
def deriv2_yz_4th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idy: float,
    idz: float
) -> float:
    """4th order mixed derivative d²f/dydz"""
    # df/dy at k-2, k-1, k+1, k+2
    dfy_km2 = idy * (-f[i, j+2, k-2] + 8.0 * f[i, j+1, k-2] - 8.0 * f[i, j-1, k-2] + f[i, j-2, k-2]) / 12.0
    dfy_km1 = idy * (-f[i, j+2, k-1] + 8.0 * f[i, j+1, k-1] - 8.0 * f[i, j-1, k-1] + f[i, j-2, k-1]) / 12.0
    dfy_kp1 = idy * (-f[i, j+2, k+1] + 8.0 * f[i, j+1, k+1] - 8.0 * f[i, j-1, k+1] + f[i, j-2, k+1]) / 12.0
    dfy_kp2 = idy * (-f[i, j+2, k+2] + 8.0 * f[i, j+1, k+2] - 8.0 * f[i, j-1, k+2] + f[i, j-2, k+2]) / 12.0
    
    return idz * (-dfy_kp2 + 8.0 * dfy_kp1 - 8.0 * dfy_km1 + dfy_km2) / 12.0


@wp.func
def upwind_x(
    f: wp.array3d(dtype=float),
    beta_x: float,
    i: int, j: int, k: int,
    idx: float
) -> float:
    """
    Upwind derivative in x-direction based on shift beta_x.
    
    If beta_x > 0: use backward difference
    If beta_x < 0: use forward difference
    """
    if beta_x > 0.0:
        # Backward: (3*f[i] - 4*f[i-1] + f[i-2]) / (2*dx)
        return beta_x * idx * (3.0 * f[i, j, k] - 4.0 * f[i-1, j, k] + f[i-2, j, k]) / 2.0
    else:
        # Forward: (-3*f[i] + 4*f[i+1] - f[i+2]) / (2*dx)
        return beta_x * idx * (-3.0 * f[i, j, k] + 4.0 * f[i+1, j, k] - f[i+2, j, k]) / 2.0


@wp.func
def upwind_y(
    f: wp.array3d(dtype=float),
    beta_y: float,
    i: int, j: int, k: int,
    idy: float
) -> float:
    """Upwind derivative in y-direction"""
    if beta_y > 0.0:
        return beta_y * idy * (3.0 * f[i, j, k] - 4.0 * f[i, j-1, k] + f[i, j-2, k]) / 2.0
    else:
        return beta_y * idy * (-3.0 * f[i, j, k] + 4.0 * f[i, j+1, k] - f[i, j+2, k]) / 2.0


@wp.func
def upwind_z(
    f: wp.array3d(dtype=float),
    beta_z: float,
    i: int, j: int, k: int,
    idz: float
) -> float:
    """Upwind derivative in z-direction"""
    if beta_z > 0.0:
        return beta_z * idz * (3.0 * f[i, j, k] - 4.0 * f[i, j, k-1] + f[i, j, k-2]) / 2.0
    else:
        return beta_z * idz * (-3.0 * f[i, j, k] + 4.0 * f[i, j, k+1] - f[i, j, k+2]) / 2.0


@wp.func
def advection(
    f: wp.array3d(dtype=float),
    beta_x: float,
    beta_y: float,
    beta_z: float,
    i: int, j: int, k: int,
    idx: float,
    idy: float,
    idz: float
) -> float:
    """
    Advection term: beta^i * d_i f
    Uses upwind differencing for stability.
    """
    return (upwind_x(f, beta_x, i, j, k, idx) +
            upwind_y(f, beta_y, i, j, k, idy) +
            upwind_z(f, beta_z, i, j, k, idz))


@wp.func
def dissipation_5th(
    f: wp.array3d(dtype=float),
    i: int, j: int, k: int,
    idx: float,
    idy: float,
    idz: float,
    eps: float
) -> float:
    """
    5th order Kreiss-Oliger dissipation.
    
    Dissipation = -eps * (dx^4) * d^6 f / dx^6 (approximately)
    Using 6-point stencil in each direction.
    """
    # 6th derivative approximation (simplified)
    diss_x = (idx ** 6.0) * (
        f[i+3, j, k] - 6.0 * f[i+2, j, k] + 15.0 * f[i+1, j, k]
        - 20.0 * f[i, j, k]
        + 15.0 * f[i-1, j, k] - 6.0 * f[i-2, j, k] + f[i-3, j, k]
    )
    
    diss_y = (idy ** 6.0) * (
        f[i, j+3, k] - 6.0 * f[i, j+2, k] + 15.0 * f[i, j+1, k]
        - 20.0 * f[i, j, k]
        + 15.0 * f[i, j-1, k] - 6.0 * f[i, j-2, k] + f[i, j-3, k]
    )
    
    diss_z = (idz ** 6.0) * (
        f[i, j, k+3] - 6.0 * f[i, j, k+2] + 15.0 * f[i, j, k+1]
        - 20.0 * f[i, j, k]
        + 15.0 * f[i, j, k-1] - 6.0 * f[i, j, k-2] + f[i, j, k-3]
    )
    
    return -eps * (diss_x + diss_y + diss_z) / 64.0
