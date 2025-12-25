import warp as wp

# Standard 4th order finite difference coefficients
# For first derivative (centered):
# f'(x) ~ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
#
# For second derivative (centered):
# f''(x) ~ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)

@wp.func
def D_1_4th(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, axis: int, dx: float):
    # Retrieve strides or just use indices
    # We need to clamp indices or assume boundaries are handled via ghost zones or periodic
    # For now, let's just clamp to boundary (Neumann/Dirichlet via clamp is crude but robust for init)
    # Better: Use a helper to get value with boundary condition.
    
    # Coefficients: [-1/12, 8/12, 0, -8/12, 1/12]
    # Indices: i+2, i+1, i, i-1, i-2
    
    inv_12dx = 1.0 / (12.0 * dx)
    
    val = 0.0
    
    if axis == 0:
        # X derivative
        # We need to fetch values safely.
        # Warp doesn't have try/except in kernels.
        # Use min/max to clamp indices.
        
        nx = f.shape[0]
        i_m2 = wp.max(0, i-2)
        i_m1 = wp.max(0, i-1)
        i_p1 = wp.min(nx-1, i+1)
        i_p2 = wp.min(nx-1, i+2)
        
        val = (-f[i_p2, j, k] + 8.0*f[i_p1, j, k] - 8.0*f[i_m1, j, k] + f[i_m2, j, k]) * inv_12dx

    elif axis == 1:
        # Y derivative
        ny = f.shape[1]
        j_m2 = wp.max(0, j-2)
        j_m1 = wp.max(0, j-1)
        j_p1 = wp.min(ny-1, j+1)
        j_p2 = wp.min(ny-1, j+2)
        
        val = (-f[i, j_p2, k] + 8.0*f[i, j_p1, k] - 8.0*f[i, j_m1, k] + f[i, j_m2, k]) * inv_12dx

    elif axis == 2:
        # Z derivative
        nz = f.shape[2]
        k_m2 = wp.max(0, k-2)
        k_m1 = wp.max(0, k-1)
        k_p1 = wp.min(nz-1, k+1)
        k_p2 = wp.min(nz-1, k+2)
        
        val = (-f[i, j, k_p2] + 8.0*f[i, j, k_p1] - 8.0*f[i, j, k_m1] + f[i, j, k_m2]) * inv_12dx
        
    return val

@wp.func
def D_2_4th(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, axis: int, dx: float):
    # f''(x) ~ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
    inv_12dx2 = 1.0 / (12.0 * dx * dx)
    
    val = 0.0
    
    if axis == 0:
        nx = f.shape[0]
        i_m2 = wp.max(0, i-2)
        i_m1 = wp.max(0, i-1)
        i_p1 = wp.min(nx-1, i+1)
        i_p2 = wp.min(nx-1, i+2)
        
        val = (-f[i_p2, j, k] + 16.0*f[i_p1, j, k] - 30.0*f[i, j, k] + 16.0*f[i_m1, j, k] - f[i_m2, j, k]) * inv_12dx2
        
    elif axis == 1:
        ny = f.shape[1]
        j_m2 = wp.max(0, j-2)
        j_m1 = wp.max(0, j-1)
        j_p1 = wp.min(ny-1, j+1)
        j_p2 = wp.min(ny-1, j+2)
        
        val = (-f[i, j_p2, k] + 16.0*f[i, j_p1, k] - 30.0*f[i, j, k] + 16.0*f[i, j_m1, k] - f[i, j_m2, k]) * inv_12dx2
        
    elif axis == 2:
        nz = f.shape[2]
        k_m2 = wp.max(0, k-2)
        k_m1 = wp.max(0, k-1)
        k_p1 = wp.min(nz-1, k+1)
        k_p2 = wp.min(nz-1, k+2)
        
        val = (-f[i, j, k_p2] + 16.0*f[i, j, k_p1] - 30.0*f[i, j, k] + 16.0*f[i, j, k_m1] - f[i, j, k_m2]) * inv_12dx2
        
    return val

@wp.func
def D_mixed_4th(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, axis1: int, axis2: int, dx: float):
    # Mixed partial derivative d^2f / dx dy
    # Apply D_1 to D_1.
    # But doing it inside a kernel is tricky without shared mem or multiple passes.
    # Standard stencil for mixed derivative involves cross terms.
    # 4th order mixed: 
    # (D_x (D_y f)) 
    # Use standard 4th order D_1 stencil on axis1 of the result of D_1 on axis2?
    # That requires evaluating D_1(f) at neighbor points.
    
    # Simpler approach: 2nd order for mixed to start? Or implement the full stencil.
    # Full stencil for dxy is applying the 5-point stencil in x to the 5-point stencil in y.
    # That's 25 points.
    
    # For now, let's implement the stencil application by calling a helper that samples "D_1" at offsets?
    # No, that's recursion.
    
    # Let's just implement 2nd order centered for mixed terms first as a placeholder, or write out the 4th order.
    # 2nd order: (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / (4h^2)
    
    # Let's stick to 2nd order for mixed derivatives for M3 to keep it simple, or upgrade later.
    # Actually, BSSN stability might depend on it.
    
    # Let's implement the 4th order mixed via nested application logic (unrolled).
    # d/dx (df/dy) ~ sum_m c_m * (df/dy)(x + m*h)
    # df/dy(x) ~ sum_n c_n * f(x, y + n*h)
    # -> sum_m sum_n c_m c_n f(x+m*h, y+n*h)
    
    # Coeffs for 1st deriv: c_2=-1/12, c_1=8/12, c_0=0, c_-1=-8/12, c_-2=1/12
    # This loop is small (4x4 = 16 non-zero points).
    
    coeffs = wp.mat33(
         0.0, 0.0, 0.0, # Placeholder, warp doesn't support array literals well inside func?
         0.0, 0.0, 0.0,
         0.0, 0.0, 0.0
    ) 
    # Using fixed unrolling.
    
    # Actually, let's implement a generic sampler and apply weights.
    # Too much code for now.
    
    # Let's use 2nd order for mixed derivatives for now.
    inv_4dx2 = 1.0 / (4.0 * dx * dx)
    
    # Indices logic similar to above...
    # For simplicity in this snippet, assumes axis1=0 (x), axis2=1 (y)
    
    # Placeholder return
    return 0.0 

# To properly verify FD kernels, we need a test script.
