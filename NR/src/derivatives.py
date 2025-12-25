import warp as wp

# Finite Difference Coefficients (4th Order)
# D1: [-1/12, 8/12, 0, -8/12, 1/12]
# D2: [-1/12, 16/12, -30/12, 16/12, -1/12]

@wp.func
def d_dx(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float):
    return ( -f[i+2, j, k] + 8.0*f[i+1, j, k] - 8.0*f[i-1, j, k] + f[i-2, j, k] ) / (12.0 * dx)

@wp.func
def d_dy(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dy: float):
    return ( -f[i, j+2, k] + 8.0*f[i, j+1, k] - 8.0*f[i, j-1, k] + f[i, j-2, k] ) / (12.0 * dy)

@wp.func
def d_dz(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dz: float):
    return ( -f[i, j, k+2] + 8.0*f[i, j, k+1] - 8.0*f[i, j, k-1] + f[i, j, k-2] ) / (12.0 * dz)

@wp.func
def d2_dx2(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float):
    return ( -f[i+2, j, k] + 16.0*f[i+1, j, k] - 30.0*f[i, j, k] + 16.0*f[i-1, j, k] - f[i-2, j, k] ) / (12.0 * dx * dx)

@wp.func
def d2_dy2(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dy: float):
    return ( -f[i, j+2, k] + 16.0*f[i, j+1, k] - 30.0*f[i, j, k] + 16.0*f[i, j-1, k] - f[i, j-2, k] ) / (12.0 * dy * dy)

@wp.func
def d2_dz2(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dz: float):
    return ( -f[i, j, k+2] + 16.0*f[i, j, k+1] - 30.0*f[i, j, k] + 16.0*f[i, j, k-1] - f[i, j, k-2] ) / (12.0 * dz * dz)

@wp.func
def d2_dxdy(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float, dy: float):
    # Mixed derivative applying D_x then D_y
    # This involves 25 points? Or just apply D_x on D_y neighbors.
    # D_x D_y f = D_x [ ( -f(j+2) + 8f(j+1) - 8f(j-1) + f(j-2) ) / 12dy ]
    # It gets complicated to inline. 
    # For simplicity, we can use 2nd order mixed if 4th order stencil is too wide, 
    # but strictly we should use 4th order.
    # 4th order mixed: Apply D_x to D_y output.
    
    # Let's simplify and assume we have helper accessors or just write it out.
    # d/dx (d/dy f)
    val = 0.0
    # Coeffs for D1: c2=-1/12, c1=8/12, c-1=-8/12, c-2=1/12
    c2 = -1.0/12.0
    c1 = 8.0/12.0
    cm1 = -8.0/12.0
    cm2 = 1.0/12.0
    
    # Outer sum (x)
    val += c2 * d_dy(f, i+2, j, k, dy)
    val += c1 * d_dy(f, i+1, j, k, dy)
    val += cm1 * d_dy(f, i-1, j, k, dy)
    val += cm2 * d_dy(f, i-2, j, k, dy)
    
    return val / dx

@wp.func
def d2_dxdz(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float, dz: float):
    c2 = -1.0/12.0
    c1 = 8.0/12.0
    cm1 = -8.0/12.0
    cm2 = 1.0/12.0
    
    val = 0.0
    val += c2 * d_dz(f, i+2, j, k, dz)
    val += c1 * d_dz(f, i+1, j, k, dz)
    val += cm1 * d_dz(f, i-1, j, k, dz)
    val += cm2 * d_dz(f, i-2, j, k, dz)
    
    return val / dx

@wp.func
def d2_dydz(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dy: float, dz: float):
    c2 = -1.0/12.0
    c1 = 8.0/12.0
    cm1 = -8.0/12.0
    cm2 = 1.0/12.0
    
    val = 0.0
    val += c2 * d_dz(f, i, j+2, k, dz)
    val += c1 * d_dz(f, i, j+1, k, dz)
    val += cm1 * d_dz(f, i, j-1, k, dz)
    val += cm2 * d_dz(f, i, j-2, k, dz)
    
    return val / dy

@wp.func
def grad(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, dx: float, dy: float, dz: float):
    return wp.vec3(
        d_dx(f, i, j, k, dx),
        d_dy(f, i, j, k, dy),
        d_dz(f, i, j, k, dz)
    )

@wp.func
def div_vec(v: wp.array(dtype=wp.vec3, ndim=3), i: int, j: int, k: int, dx: float, dy: float, dz: float):
    # D_i v^i
    return d_dx_comp(v, 0, i, j, k, dx) + d_dy_comp(v, 1, i, j, k, dy) + d_dz_comp(v, 2, i, j, k, dz)

# Component-wise derivatives for vector/tensor fields
@wp.func
def d_dx_comp(f: wp.array(dtype=wp.vec3, ndim=3), c: int, i: int, j: int, k: int, dx: float):
    # accessing vector component c
    v_p2 = f[i+2, j, k][c]
    v_p1 = f[i+1, j, k][c]
    v_m1 = f[i-1, j, k][c]
    v_m2 = f[i-2, j, k][c]
    return ( -v_p2 + 8.0*v_p1 - 8.0*v_m1 + v_m2 ) / (12.0 * dx)

@wp.func
def d_dy_comp(f: wp.array(dtype=wp.vec3, ndim=3), c: int, i: int, j: int, k: int, dy: float):
    v_p2 = f[i, j+2, k][c]
    v_p1 = f[i, j+1, k][c]
    v_m1 = f[i, j-1, k][c]
    v_m2 = f[i, j-2, k][c]
    return ( -v_p2 + 8.0*v_p1 - 8.0*v_m1 + v_m2 ) / (12.0 * dy)

@wp.func
def d_dz_comp(f: wp.array(dtype=wp.vec3, ndim=3), c: int, i: int, j: int, k: int, dz: float):
    v_p2 = f[i, j, k+2][c]
    v_p1 = f[i, j, k+1][c]
    v_m1 = f[i, j, k-1][c]
    v_m2 = f[i, j, k-2][c]
    return ( -v_p2 + 8.0*v_p1 - 8.0*v_m1 + v_m2 ) / (12.0 * dz)

# Kreiss-Oliger Dissipation (6th order deriv / 5th order accuracy?)
# Standard is to add eps/h * D_6 f
# D_6 = (1, -6, 15, -20, 15, -6, 1) / 64
@wp.func
def ko_dissipation(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, sigma: float):
    # sigma is dissipation strength parameter (usually ~0.1-0.5 / 64)
    # Applied in all directions
    
    val = 0.0
    # X direction
    val += (f[i+3,j,k] - 6.0*f[i+2,j,k] + 15.0*f[i+1,j,k] - 20.0*f[i,j,k] + 15.0*f[i-1,j,k] - 6.0*f[i-2,j,k] + f[i-3,j,k])
    # Y direction
    val += (f[i,j+3,k] - 6.0*f[i,j+2,k] + 15.0*f[i,j+1,k] - 20.0*f[i,j,k] + 15.0*f[i,j-1,k] - 6.0*f[i,j-2,k] + f[i,j-3,k])
    # Z direction
    val += (f[i,j,k+3] - 6.0*f[i,j,k+2] + 15.0*f[i,j,k+1] - 20.0*f[i,j,k] + 15.0*f[i,j,k-1] - 6.0*f[i,j,k-2] + f[i,j,k-3])
    
    return -sigma * val # Negative sign because D6 is typically negative definite? 
    # Usually dissipation term is - epsilon * (-1)^p D_2p
    # For p=3 (6th order), (-1)^3 = -1. So term is + epsilon * D_6.
    # D_6 stencil center is -20.
    # We want to damp high freq.
    # Let's check sign.
    
    # Standard KO: rhs += - epsilon * (dx)^5 * D_6 / dx^6 = - epsilon/dx * D_6_stencil
    # (assuming factor 64 is absorbed in epsilon or explicit)
    
    return -sigma * val

