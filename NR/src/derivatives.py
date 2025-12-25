import warp as wp

@wp.func
def idx_periodic(i: int, N: int):
    # Helper for periodic boundary conditions
    res = i % N
    if res < 0:
        res = res + N
    return res

@wp.func
def D_x(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float):
    Nx = f.shape[0]
    i_m2 = idx_periodic(i - 2, Nx)
    i_m1 = idx_periodic(i - 1, Nx)
    i_p1 = idx_periodic(i + 1, Nx)
    i_p2 = idx_periodic(i + 2, Nx)
    val = -f[i_p2, j, k] + 8.0 * f[i_p1, j, k] - 8.0 * f[i_m1, j, k] + f[i_m2, j, k]
    return val / (12.0 * dx)

@wp.func
def D_y(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dy: float):
    Ny = f.shape[1]
    j_m2 = idx_periodic(j - 2, Ny)
    j_m1 = idx_periodic(j - 1, Ny)
    j_p1 = idx_periodic(j + 1, Ny)
    j_p2 = idx_periodic(j + 2, Ny)
    val = -f[i, j_p2, k] + 8.0 * f[i, j_p1, k] - 8.0 * f[i, j_m1, k] + f[i, j_m2, k]
    return val / (12.0 * dy)

@wp.func
def D_z(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dz: float):
    Nz = f.shape[2]
    k_m2 = idx_periodic(k - 2, Nz)
    k_m1 = idx_periodic(k - 1, Nz)
    k_p1 = idx_periodic(k + 1, Nz)
    k_p2 = idx_periodic(k + 2, Nz)
    val = -f[i, j, k_p2] + 8.0 * f[i, j, k_p1] - 8.0 * f[i, j, k_m1] + f[i, j, k_m2]
    return val / (12.0 * dz)

@wp.func
def D2_x(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float):
    Nx = f.shape[0]
    i_m2 = idx_periodic(i - 2, Nx)
    i_m1 = idx_periodic(i - 1, Nx)
    i_p1 = idx_periodic(i + 1, Nx)
    i_p2 = idx_periodic(i + 2, Nx)
    val = -f[i_p2, j, k] + 16.0 * f[i_p1, j, k] - 30.0 * f[i, j, k] + 16.0 * f[i_m1, j, k] - f[i_m2, j, k]
    return val / (12.0 * dx * dx)

@wp.func
def D2_y(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dy: float):
    Ny = f.shape[1]
    j_m2 = idx_periodic(j - 2, Ny)
    j_m1 = idx_periodic(j - 1, Ny)
    j_p1 = idx_periodic(j + 1, Ny)
    j_p2 = idx_periodic(j + 2, Ny)
    val = -f[i, j_p2, k] + 16.0 * f[i, j_p1, k] - 30.0 * f[i, j, k] + 16.0 * f[i, j_m1, k] - f[i, j_m2, k]
    return val / (12.0 * dy * dy)

@wp.func
def D2_z(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dz: float):
    Nz = f.shape[2]
    k_m2 = idx_periodic(k - 2, Nz)
    k_m1 = idx_periodic(k - 1, Nz)
    k_p1 = idx_periodic(k + 1, Nz)
    k_p2 = idx_periodic(k + 2, Nz)
    val = -f[i, j, k_p2] + 16.0 * f[i, j, k_p1] - 30.0 * f[i, j, k_m1] - f[i, j, k_m2]
    return val / (12.0 * dz * dz)

# Vector derivatives
@wp.func
def D_x_vec3(f: wp.array(dtype=wp.vec3, ndim=3), i: int, j: int, k: int, dx: float):
    Nx = f.shape[0]
    i_m2 = idx_periodic(i - 2, Nx)
    i_m1 = idx_periodic(i - 1, Nx)
    i_p1 = idx_periodic(i + 1, Nx)
    i_p2 = idx_periodic(i + 2, Nx)
    val = -f[i_p2, j, k] + 8.0 * f[i_p1, j, k] - 8.0 * f[i_m1, j, k] + f[i_m2, j, k]
    return val / (12.0 * dx)

@wp.func
def D_y_vec3(f: wp.array(dtype=wp.vec3, ndim=3), i: int, j: int, k: int, dy: float):
    Ny = f.shape[1]
    j_m2 = idx_periodic(j - 2, Ny)
    j_m1 = idx_periodic(j - 1, Ny)
    j_p1 = idx_periodic(j + 1, Ny)
    j_p2 = idx_periodic(j + 2, Ny)
    val = -f[i, j_p2, k] + 8.0 * f[i, j_p1, k] - 8.0 * f[i, j_m1, k] + f[i, j_m2, k]
    return val / (12.0 * dy)

@wp.func
def D_z_vec3(f: wp.array(dtype=wp.vec3, ndim=3), i: int, j: int, k: int, dz: float):
    Nz = f.shape[2]
    k_m2 = idx_periodic(k - 2, Nz)
    k_m1 = idx_periodic(k - 1, Nz)
    k_p1 = idx_periodic(k + 1, Nz)
    k_p2 = idx_periodic(k + 2, Nz)
    val = -f[i, j, k_p2] + 8.0 * f[i, j, k_p1] - 8.0 * f[i, j, k_m1] + f[i, j, k_m2]
    return val / (12.0 * dz)

# Tensor derivatives
@wp.func
def D_x_mat33(f: wp.array(dtype=wp.mat33, ndim=3), i: int, j: int, k: int, dx: float):
    Nx = f.shape[0]
    i_m2 = idx_periodic(i - 2, Nx)
    i_m1 = idx_periodic(i - 1, Nx)
    i_p1 = idx_periodic(i + 1, Nx)
    i_p2 = idx_periodic(i + 2, Nx)
    val = -f[i_p2, j, k] + 8.0 * f[i_p1, j, k] - 8.0 * f[i_m1, j, k] + f[i_m2, j, k]
    return val / (12.0 * dx)

@wp.func
def D_y_mat33(f: wp.array(dtype=wp.mat33, ndim=3), i: int, j: int, k: int, dy: float):
    Ny = f.shape[1]
    j_m2 = idx_periodic(j - 2, Ny)
    j_m1 = idx_periodic(j - 1, Ny)
    j_p1 = idx_periodic(j + 1, Ny)
    j_p2 = idx_periodic(j + 2, Ny)
    val = -f[i, j_p2, k] + 8.0 * f[i, j_p1, k] - 8.0 * f[i, j_m1, k] + f[i, j_m2, k]
    return val / (12.0 * dy)

@wp.func
def D_z_mat33(f: wp.array(dtype=wp.mat33, ndim=3), i: int, j: int, k: int, dz: float):
    Nz = f.shape[2]
    k_m2 = idx_periodic(k - 2, Nz)
    k_m1 = idx_periodic(k - 1, Nz)
    k_p1 = idx_periodic(k + 1, Nz)
    k_p2 = idx_periodic(k + 2, Nz)
    val = -f[i, j, k_p2] + 8.0 * f[i, j, k_p1] - 8.0 * f[i, j, k_m1] + f[i, j, k_m2]
    return val / (12.0 * dz)

# Dissipation
@wp.func
def ko_dissipation_scalar(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, eps: float):
    # 4th order dissipation stencil (sum of x, y, z)
    # eps * (D4x + D4y + D4z)
    # Stencil: 1, -4, 6, -4, 1
    # We subtract it to dampen? D4 is positive definite? k^4 > 0.
    # So we want -epsilon * D4.
    
    Nx = f.shape[0]
    Ny = f.shape[1]
    Nz = f.shape[2]
    
    # X
    i_m2 = idx_periodic(i - 2, Nx)
    i_m1 = idx_periodic(i - 1, Nx)
    i_p1 = idx_periodic(i + 1, Nx)
    i_p2 = idx_periodic(i + 2, Nx)
    term_x = f[i_m2, j, k] - 4.0 * f[i_m1, j, k] + 6.0 * f[i, j, k] - 4.0 * f[i_p1, j, k] + f[i_p2, j, k]
    
    # Y
    j_m2 = idx_periodic(j - 2, Ny)
    j_m1 = idx_periodic(j - 1, Ny)
    j_p1 = idx_periodic(j + 1, Ny)
    j_p2 = idx_periodic(j + 2, Ny)
    term_y = f[i, j_m2, k] - 4.0 * f[i, j_m1, k] + 6.0 * f[i, j, k] - 4.0 * f[i, j_p1, k] + f[i, j_p2, k]
    
    # Z
    k_m2 = idx_periodic(k - 2, Nz)
    k_m1 = idx_periodic(k - 1, Nz)
    k_p1 = idx_periodic(k + 1, Nz)
    k_p2 = idx_periodic(k + 2, Nz)
    term_z = f[i, j, k_m2] - 4.0 * f[i, j, k_m1] + 6.0 * f[i, j, k] - 4.0 * f[i, j, k_p1] + f[i, j, k_p2]
    
    return -eps * (term_x + term_y + term_z)

@wp.func
def ko_dissipation_vec3(f: wp.array(dtype=wp.vec3, ndim=3), i: int, j: int, k: int, eps: float):
    # Same as scalar but for vec3
    Nx = f.shape[0]
    Ny = f.shape[1]
    Nz = f.shape[2]
    
    i_m2 = idx_periodic(i - 2, Nx); i_m1 = idx_periodic(i - 1, Nx); i_p1 = idx_periodic(i + 1, Nx); i_p2 = idx_periodic(i + 2, Nx)
    term_x = f[i_m2, j, k] - 4.0 * f[i_m1, j, k] + 6.0 * f[i, j, k] - 4.0 * f[i_p1, j, k] + f[i_p2, j, k]
    
    j_m2 = idx_periodic(j - 2, Ny); j_m1 = idx_periodic(j - 1, Ny); j_p1 = idx_periodic(j + 1, Ny); j_p2 = idx_periodic(j + 2, Ny)
    term_y = f[i, j_m2, k] - 4.0 * f[i, j_m1, k] + 6.0 * f[i, j, k] - 4.0 * f[i, j_p1, k] + f[i, j_p2, k]
    
    k_m2 = idx_periodic(k - 2, Nz); k_m1 = idx_periodic(k - 1, Nz); k_p1 = idx_periodic(k + 1, Nz); k_p2 = idx_periodic(k + 2, Nz)
    term_z = f[i, j, k_m2] - 4.0 * f[i, j, k_m1] + 6.0 * f[i, j, k] - 4.0 * f[i, j, k_p1] + f[i, j, k_p2]
    
    return -eps * (term_x + term_y + term_z)

@wp.func
def ko_dissipation_mat33(f: wp.array(dtype=wp.mat33, ndim=3), i: int, j: int, k: int, eps: float):
    # Same as scalar but for mat33
    Nx = f.shape[0]
    Ny = f.shape[1]
    Nz = f.shape[2]
    
    i_m2 = idx_periodic(i - 2, Nx); i_m1 = idx_periodic(i - 1, Nx); i_p1 = idx_periodic(i + 1, Nx); i_p2 = idx_periodic(i + 2, Nx)
    term_x = f[i_m2, j, k] - 4.0 * f[i_m1, j, k] + 6.0 * f[i, j, k] - 4.0 * f[i_p1, j, k] + f[i_p2, j, k]
    
    j_m2 = idx_periodic(j - 2, Ny); j_m1 = idx_periodic(j - 1, Ny); j_p1 = idx_periodic(j + 1, Ny); j_p2 = idx_periodic(j + 2, Ny)
    term_y = f[i, j_m2, k] - 4.0 * f[i, j_m1, k] + 6.0 * f[i, j, k] - 4.0 * f[i, j_p1, k] + f[i, j_p2, k]
    
    k_m2 = idx_periodic(k - 2, Nz); k_m1 = idx_periodic(k - 1, Nz); k_p1 = idx_periodic(k + 1, Nz); k_p2 = idx_periodic(k + 2, Nz)
    term_z = f[i, j, k_m2] - 4.0 * f[i, j, k_m1] + 6.0 * f[i, j, k] - 4.0 * f[i, j, k_p1] + f[i, j, k_p2]
    
    return -eps * (term_x + term_y + term_z)
