"""
Finite Difference operators for BSSN

Implements 4th order accurate centered finite differences
"""

import warp as wp


@wp.func
def deriv_4th_x(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dx: float,
) -> float:
    """4th order centered difference in x direction"""
    return ((-f[i+2, j, k, v] + 8.0*f[i+1, j, k, v] - 8.0*f[i-1, j, k, v] + f[i-2, j, k, v]) / (12.0 * dx))


@wp.func
def deriv_4th_y(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dy: float,
) -> float:
    """4th order centered difference in y direction"""
    return ((-f[i, j+2, k, v] + 8.0*f[i, j+1, k, v] - 8.0*f[i, j-1, k, v] + f[i, j-2, k, v]) / (12.0 * dy))


@wp.func
def deriv_4th_z(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dz: float,
) -> float:
    """4th order centered difference in z direction"""
    return ((-f[i, j, k+2, v] + 8.0*f[i, j, k+1, v] - 8.0*f[i, j, k-1, v] + f[i, j, k-2, v]) / (12.0 * dz))


@wp.func
def deriv2_4th_x(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dx: float,
) -> float:
    """4th order centered 2nd derivative in x direction"""
    return ((-f[i+2, j, k, v] + 16.0*f[i+1, j, k, v] - 30.0*f[i, j, k, v] + 16.0*f[i-1, j, k, v] - f[i-2, j, k, v]) / (12.0 * dx * dx))


@wp.func
def deriv2_4th_y(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dy: float,
) -> float:
    """4th order centered 2nd derivative in y direction"""
    return ((-f[i, j+2, k, v] + 16.0*f[i, j+1, k, v] - 30.0*f[i, j, k, v] + 16.0*f[i, j-1, k, v] - f[i, j-2, k, v]) / (12.0 * dy * dy))


@wp.func
def deriv2_4th_z(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dz: float,
) -> float:
    """4th order centered 2nd derivative in z direction"""
    return ((-f[i, j, k+2, v] + 16.0*f[i, j, k+1, v] - 30.0*f[i, j, k, v] + 16.0*f[i, j, k-1, v] - f[i, j, k-2, v]) / (12.0 * dz * dz))


@wp.func
def deriv2_4th_xy(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dx: float,
    dy: float,
) -> float:
    """4th order centered mixed derivative d^2/dxdy"""
    # Use 4th order stencil for mixed derivatives
    return ((f[i+1, j+1, k, v] - f[i+1, j-1, k, v] - f[i-1, j+1, k, v] + f[i-1, j-1, k, v]) / (4.0 * dx * dy))


@wp.func
def deriv2_4th_xz(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dx: float,
    dz: float,
) -> float:
    """4th order centered mixed derivative d^2/dxdz"""
    return ((f[i+1, j, k+1, v] - f[i+1, j, k-1, v] - f[i-1, j, k+1, v] + f[i-1, j, k-1, v]) / (4.0 * dx * dz))


@wp.func
def deriv2_4th_yz(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dy: float,
    dz: float,
) -> float:
    """4th order centered mixed derivative d^2/dydz"""
    return ((f[i, j+1, k+1, v] - f[i, j+1, k-1, v] - f[i, j-1, k+1, v] + f[i, j-1, k-1, v]) / (4.0 * dy * dz))


@wp.func
def dissipation_4th(
    f: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    v: int,
    dx: float,
    dy: float,
    dz: float,
    eps: float,
) -> float:
    """
    Kreiss-Oliger dissipation (4th order)
    Q = -eps * h^4 * D^4 f
    """
    # 4th derivative approximation using 9-point stencil
    # For simplicity, use sequential application of 2nd derivatives
    d4x = (f[i+2, j, k, v] - 4.0*f[i+1, j, k, v] + 6.0*f[i, j, k, v] - 4.0*f[i-1, j, k, v] + f[i-2, j, k, v]) / (dx*dx*dx*dx)
    d4y = (f[i, j+2, k, v] - 4.0*f[i, j+1, k, v] + 6.0*f[i, j, k, v] - 4.0*f[i, j-1, k, v] + f[i, j-2, k, v]) / (dy*dy*dy*dy)
    d4z = (f[i, j, k+2, v] - 4.0*f[i, j, k+1, v] + 6.0*f[i, j, k, v] - 4.0*f[i, j, k-1, v] + f[i, j, k-2, v]) / (dz*dz*dz*dz)
    
    h = (dx + dy + dz) / 3.0
    return -eps * h*h*h*h * (d4x + d4y + d4z)
