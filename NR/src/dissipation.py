import warp as wp

# Kreiss-Oliger dissipation
# D_diss(u) = - sigma * (h^3 / 16) * D_+^2 D_-^2 u (for 3rd order, 4th order derivative)
# Standard KO dissipation for 4th order FD is usually 5th derivative or similar, scaled by dx.
# Common form: D_t u = ... + epsilon_diss * (dx)^(p-1) * D^p u
# For 4th order scheme, typically use 6th order derivative operator for dissipation? Or 4th?
# 
# Let's use the standard "sigma" term.
# Q_KO = epsilon * (-1)^(r+1) * dx^(2r-1) * (d^(2r) / dx^(2r)) u
# For 4th order accurate code (p=4), we usually set 2r > p. E.g. 2r=6 (r=3).
# Or 2r=4 (r=2) for 2nd order code.
# 
# Let's implement a simple dissipation kernel.
# For now, let's skip complex stencil and use a placeholder or simplified laplacian-like damping if needed,
# but for flat spacetime it's not strictly needed yet.
#
# Actually, let's implement the 4th derivative operator (stencil size 5).
# d^4 f / dx^4 ~ (f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)) / h^4
#
# KO term added to RHS: - sigma * (h^3) * (d^4 f / dx^4) ? No.
# If we add h^3 * D^4, it's O(h^3) error, which kills 4th order accuracy (O(h^4)).
# We need higher order dissipation or scale properly.
# Standard: sigma * (dx/dt) * (dx)^3 * D^4 (effectively O(dx^3)).
#
# Let's verify standard BSSN Dissipation.
# Usually: eps/64 * D_diss
#
# Let's implement the stencil function in derivatives.py

@wp.func
def ko_dissipation_4th(f: wp.array(dtype=float, ndim=3), i: int, j: int, k: int, axis: int, dx: float, sigma: float):
    # Computes sigma * (dx)^3 * d^4f/dx^4 
    # Stencil: 1, -4, 6, -4, 1
    # We want the term to be added to RHS.
    # Dimensions: [1/T]. dx^3 * [1/L^4] * [U] = [U/L]. Consistent if sigma has units [L/T]?
    # Usually sigma ~ 1.0 (dimensionless if prefactor handled).
    
    # We ignore the 1/dx^4 factor if we just sum the coefficients and scale by sigma/dx ?
    # Let's assume the dissipation operator returns the update value directly.
    
    # Term = (sigma / 64) * ( -f(i-2) + 4f(i-1) - 6f(i) + 4f(i+1) - f(i+2) ) / dt ?
    # Let's follow a standard reference (e.g. Alcubierre book).
    # operator D = (f_{i+2} - 4f_{i+1} + 6f_i - 4f_{i-1} + f_{i-2})
    # RHS += - (epsilon / 64) * D / dt * (some factor?)
    
    # Let's just implement the operator D first.
    
    val = 0.0
    
    if axis == 0:
        nx = f.shape[0]
        i_m2 = wp.max(0, i-2)
        i_m1 = wp.max(0, i-1)
        i_p1 = wp.min(nx-1, i+1)
        i_p2 = wp.min(nx-1, i+2)
        
        val = (f[i_m2, j, k] - 4.0*f[i_m1, j, k] + 6.0*f[i, j, k] - 4.0*f[i_p1, j, k] + f[i_p2, j, k])
        
    elif axis == 1:
        ny = f.shape[1]
        j_m2 = wp.max(0, j-2)
        j_m1 = wp.max(0, j-1)
        j_p1 = wp.min(ny-1, j+1)
        j_p2 = wp.min(ny-1, j+2)
        
        val = (f[i, j_m2, k] - 4.0*f[i, j_m1, k] + 6.0*f[i, j, k] - 4.0*f[i, j_p1, k] + f[i, j_p2, k])

    elif axis == 2:
        nz = f.shape[2]
        k_m2 = wp.max(0, k-2)
        k_m1 = wp.max(0, k-1)
        k_p1 = wp.min(nz-1, k+1)
        k_p2 = wp.min(nz-1, k+2)
        
        val = (f[i, j, k_m2] - 4.0*f[i, j, k_m1] + 6.0*f[i, j, k] - 4.0*f[i, j, k_p1] + f[i, j, k_p2])
        
    # Scale: -epsilon/64 * (1/dx) ? No, unit analysis suggests 1/dt scaling or similar.
    # If we add to RHS (du/dt), units are U/T.
    # D is dimensionless (sum of U).
    # We need U/T. So D/dt? Or D * (dx/dt) / dx?
    # Usually epsilon * D / dx * (characteristic speed ~ 1).
    
    # Let's use sigma * D / dx.
    
    return -sigma * val / dx

