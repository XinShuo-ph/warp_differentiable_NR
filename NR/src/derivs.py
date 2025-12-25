import warp as wp

# Finite Difference Kernels (4th Order)

@wp.func
def d_dx(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float):
    return ( -f[i+2,j,k] + 8.0*f[i+1,j,k] - 8.0*f[i-1,j,k] + f[i-2,j,k] ) / (12.0 * dx)

@wp.func
def d_dy(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dy: float):
    return ( -f[i,j+2,k] + 8.0*f[i,j+1,k] - 8.0*f[i,j-1,k] + f[i,j-2,k] ) / (12.0 * dy)

@wp.func
def d_dz(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dz: float):
    return ( -f[i,j,k+2] + 8.0*f[i,j,k+1] - 8.0*f[i,j,k-1] + f[i,j,k-2] ) / (12.0 * dz)

@wp.func
def d2_dx2(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float):
    return ( -f[i+2,j,k] + 16.0*f[i+1,j,k] - 30.0*f[i,j,k] + 16.0*f[i-1,j,k] - f[i-2,j,k] ) / (12.0 * dx * dx)

@wp.func
def d2_dy2(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dy: float):
    return ( -f[i,j+2,k] + 16.0*f[i,j+1,k] - 30.0*f[i,j,k] + 16.0*f[i,j-1,k] - f[i,j-2,k] ) / (12.0 * dy * dy)

@wp.func
def d2_dz2(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dz: float):
    return ( -f[i,j,k+2] + 16.0*f[i,j,k+1] - 30.0*f[i,j,k] + 16.0*f[i,j,k-1] - f[i,j,k-2] ) / (12.0 * dz * dz)

@wp.func
def d2_dxdy(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float, dy: float):
    # 4th order mixed derivative? 
    # Usually standard centered difference of first derivatives.
    # d/dx (d/dy f)
    # 2nd order mixed is (f_i+1,j+1 - f_i+1,j-1 - f_i-1,j+1 + f_i-1,j-1) / (4 dx dy)
    # 4th order is more complex stencil.
    # For now, apply d_dx to d_dy stencil? Or use 2nd order for mixed?
    # Let's use 2nd order for simplicity or implement full 4th order if needed.
    # Standard 4th order cross derivative:
    # 1/(144 h^2) [ 8(f_i+1,j-2 - f_i+2,j-1 - f_i-2,j+1 + f_i-1,j+2) + ... ] 
    # It's huge. 
    # Let's use applying d_dx on d_dy results (iterated).
    # d/dx (d/dy) ~ D_x(4) D_y(4).
    # But inside a kernel we don't have intermediate array.
    # We need to sample multiple points.
    
    # Let's stick to 2nd order for mixed terms for now, or use a simplified 4th order.
    # (f_i+1,j+1 - f_i+1,j-1 - f_i-1,j+1 + f_i-1,j-1) / (4*dx*dy)
    val = (f[i+1,j+1,k] - f[i+1,j-1,k] - f[i-1,j+1,k] + f[i-1,j-1,k]) / (4.0 * dx * dy)
    return val

# Dissipation (KO)
@wp.func
def kreiss_oliger_dissipation(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, sigma: float):
    # 5th order KO dissipation term for 4th order FD
    val_x = -(f[i-2,j,k] - 4.0*f[i-1,j,k] + 6.0*f[i,j,k] - 4.0*f[i+1,j,k] + f[i+2,j,k])
    val_y = -(f[i,j-2,k] - 4.0*f[i,j-1,k] + 6.0*f[i,j,k] - 4.0*f[i,j+1,k] + f[i,j+2,k])
    val_z = -(f[i,j,k-2] - 4.0*f[i,j,k-1] + 6.0*f[i,j,k] - 4.0*f[i,j,k+1] + f[i,j,k+2])
    return sigma * (val_x + val_y + val_z)

@wp.func
def kreiss_oliger_dissipation_comp(f: wp.array(dtype=float, ndim=4), i: int, j: int, k: int, c: int, sigma: float):
    val_x = -(f[i-2,j,k,c] - 4.0*f[i-1,j,k,c] + 6.0*f[i,j,k,c] - 4.0*f[i+1,j,k,c] + f[i+2,j,k,c])
    val_y = -(f[i,j-2,k,c] - 4.0*f[i,j-1,k,c] + 6.0*f[i,j,k,c] - 4.0*f[i,j+1,k,c] + f[i,j+2,k,c])
    val_z = -(f[i,j,k-2,c] - 4.0*f[i,j,k-1,c] + 6.0*f[i,j,k,c] - 4.0*f[i,j,k+1,c] + f[i,j,k+2,c])
    return sigma * (val_x + val_y + val_z)

@wp.func
def d_dx_comp(f: wp.array(dtype=float, ndim=4), i: int, j: int, k: int, c: int, dx: float):
    return ( -f[i+2,j,k,c] + 8.0*f[i+1,j,k,c] - 8.0*f[i-1,j,k,c] + f[i-2,j,k,c] ) / (12.0 * dx)

@wp.func
def d_dy_comp(f: wp.array(dtype=float, ndim=4), i: int, j: int, k: int, c: int, dy: float):
    return ( -f[i,j+2,k,c] + 8.0*f[i,j+1,k,c] - 8.0*f[i,j-1,k,c] + f[i,j-2,k,c] ) / (12.0 * dy)

@wp.func
def d_dz_comp(f: wp.array(dtype=float, ndim=4), i: int, j: int, k: int, c: int, dz: float):
    return ( -f[i,j,k+2,c] + 8.0*f[i,j,k+1,c] - 8.0*f[i,j,k-1,c] + f[i,j,k-2,c] ) / (12.0 * dz)


