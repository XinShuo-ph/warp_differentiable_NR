"""
Spatial derivatives for BSSN evolution

Implements 4th order centered finite differences for computing spatial
derivatives of BSSN variables on a uniform Cartesian grid.
"""

import warp as wp

@wp.func
def deriv_x_4th(
    f: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    idx: float,
    nx: int
) -> float:
    """
    4th order centered finite difference in x-direction
    
    Args:
        f: scalar field
        i,j,k: grid indices
        idx: 1/dx
        nx: grid size in x
    
    Returns:
        df/dx at (i,j,k)
    """
    # Boundary handling: use one-sided stencils near boundaries
    if i < 2 or i >= nx - 2:
        # 2nd order one-sided
        if i == 0:
            return idx * (-1.5 * f[i,j,k] + 2.0 * f[i+1,j,k] - 0.5 * f[i+2,j,k])
        elif i == 1:
            return idx * 0.5 * (f[i+1,j,k] - f[i-1,j,k])
        elif i == nx-2:
            return idx * 0.5 * (f[i+1,j,k] - f[i-1,j,k])
        else:  # i == nx-1
            return idx * (1.5 * f[i,j,k] - 2.0 * f[i-1,j,k] + 0.5 * f[i-2,j,k])
    
    # 4th order centered
    return idx * (1.0/12.0 * (-f[i+2,j,k] + 8.0*f[i+1,j,k] - 8.0*f[i-1,j,k] + f[i-2,j,k]))

@wp.func
def deriv_y_4th(
    f: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    idy: float,
    ny: int
) -> float:
    """4th order centered finite difference in y-direction"""
    if j < 2 or j >= ny - 2:
        if j == 0:
            return idy * (-1.5 * f[i,j,k] + 2.0 * f[i,j+1,k] - 0.5 * f[i,j+2,k])
        elif j == 1:
            return idy * 0.5 * (f[i,j+1,k] - f[i,j-1,k])
        elif j == ny-2:
            return idy * 0.5 * (f[i,j+1,k] - f[i,j-1,k])
        else:
            return idy * (1.5 * f[i,j,k] - 2.0 * f[i,j-1,k] + 0.5 * f[i,j-2,k])
    
    return idy * (1.0/12.0 * (-f[i,j+2,k] + 8.0*f[i,j+1,k] - 8.0*f[i,j-1,k] + f[i,j-2,k]))

@wp.func
def deriv_z_4th(
    f: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    idz: float,
    nz: int
) -> float:
    """4th order centered finite difference in z-direction"""
    if k < 2 or k >= nz - 2:
        if k == 0:
            return idz * (-1.5 * f[i,j,k] + 2.0 * f[i,j,k+1] - 0.5 * f[i,j,k+2])
        elif k == 1:
            return idz * 0.5 * (f[i,j,k+1] - f[i,j,k-1])
        elif k == nz-2:
            return idz * 0.5 * (f[i,j,k+1] - f[i,j,k-1])
        else:
            return idz * (1.5 * f[i,j,k] - 2.0 * f[i,j,k-1] + 0.5 * f[i,j,k-2])
    
    return idz * (1.0/12.0 * (-f[i,j,k+2] + 8.0*f[i,j,k+1] - 8.0*f[i,j,k-1] + f[i,j,k-2]))

@wp.func
def deriv_xx_4th(
    f: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    idx2: float,
    nx: int
) -> float:
    """
    4th order centered 2nd derivative in x-direction
    
    Args:
        f: scalar field
        i,j,k: grid indices
        idx2: 1/dx^2
        nx: grid size in x
    
    Returns:
        d^2f/dx^2 at (i,j,k)
    """
    if i < 2 or i >= nx - 2:
        if i <= 1 or i >= nx-2:
            # 2nd order
            return idx2 * (f[i+1,j,k] - 2.0*f[i,j,k] + f[i-1,j,k])
        return idx2 * (f[i+1,j,k] - 2.0*f[i,j,k] + f[i-1,j,k])
    
    # 4th order centered
    return idx2 * (1.0/12.0 * (-f[i+2,j,k] + 16.0*f[i+1,j,k] - 30.0*f[i,j,k] + 16.0*f[i-1,j,k] - f[i-2,j,k]))

@wp.func
def deriv_yy_4th(
    f: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    idy2: float,
    ny: int
) -> float:
    """4th order centered 2nd derivative in y-direction"""
    if j < 2 or j >= ny - 2:
        if j <= 1 or j >= ny-2:
            return idy2 * (f[i,j+1,k] - 2.0*f[i,j,k] + f[i,j-1,k])
        return idy2 * (f[i,j+1,k] - 2.0*f[i,j,k] + f[i,j-1,k])
    
    return idy2 * (1.0/12.0 * (-f[i,j+2,k] + 16.0*f[i,j+1,k] - 30.0*f[i,j,k] + 16.0*f[i,j-1,k] - f[i,j-2,k]))

@wp.func
def deriv_zz_4th(
    f: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    idz2: float,
    nz: int
) -> float:
    """4th order centered 2nd derivative in z-direction"""
    if k < 2 or k >= nz - 2:
        if k <= 1 or k >= nz-2:
            return idz2 * (f[i,j,k+1] - 2.0*f[i,j,k] + f[i,j,k-1])
        return idz2 * (f[i,j,k+1] - 2.0*f[i,j,k] + f[i,j,k-1])
    
    return idz2 * (1.0/12.0 * (-f[i,j,k+2] + 16.0*f[i,j,k+1] - 30.0*f[i,j,k] + 16.0*f[i,j,k-1] - f[i,j,k-2]))

@wp.func
def deriv_xy_4th(
    f: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
    idx: float,
    idy: float,
    nx: int,
    ny: int
) -> float:
    """
    4th order centered mixed derivative d^2f/dxdy
    
    Uses d/dx(df/dy)
    """
    # Compute df/dy at adjacent x points
    dfdy_xp2 = deriv_y_4th(f, i+2, j, k, idy, ny)
    dfdy_xp1 = deriv_y_4th(f, i+1, j, k, idy, ny)
    dfdy_xm1 = deriv_y_4th(f, i-1, j, k, idy, ny)
    dfdy_xm2 = deriv_y_4th(f, i-2, j, k, idy, ny)
    
    # Apply x-derivative stencil
    if i < 2 or i >= nx - 2:
        if i <= 1:
            return 0.5 * idx * (dfdy_xp1 - dfdy_xm1)
        else:
            return 0.5 * idx * (dfdy_xp1 - dfdy_xm1)
    
    return idx * (1.0/12.0 * (-dfdy_xp2 + 8.0*dfdy_xp1 - 8.0*dfdy_xm1 + dfdy_xm2))
