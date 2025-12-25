import warp as wp
from derivs import d_dx, d_dy, d_dz, d2_dx2, d2_dy2, d2_dz2, d_dx_comp, d_dy_comp, d_dz_comp, kreiss_oliger_dissipation, kreiss_oliger_dissipation_comp
from bssn import IDX_XX, IDX_XY, IDX_XZ, IDX_YY, IDX_YZ, IDX_ZZ

@wp.kernel
def bssn_rhs_kernel(
    # State variables
    phi: wp.array(dtype=float, ndim=3),
    gamma_tilde: wp.array(dtype=float, ndim=4),
    K: wp.array(dtype=float, ndim=3),
    A_tilde: wp.array(dtype=float, ndim=4),
    Gam_tilde: wp.array(dtype=float, ndim=4),
    alpha: wp.array(dtype=float, ndim=3),
    beta: wp.array(dtype=float, ndim=4),
    B: wp.array(dtype=float, ndim=4),
    # RHS variables (output)
    rhs_phi: wp.array(dtype=float, ndim=3),
    rhs_gamma_tilde: wp.array(dtype=float, ndim=4),
    rhs_K: wp.array(dtype=float, ndim=3),
    rhs_A_tilde: wp.array(dtype=float, ndim=4),
    rhs_Gam_tilde: wp.array(dtype=float, ndim=4),
    rhs_alpha: wp.array(dtype=float, ndim=3),
    rhs_beta: wp.array(dtype=float, ndim=4),
    rhs_B: wp.array(dtype=float, ndim=4),
    # Grid params
    dx: float,
    dy: float,
    dz: float
):
    i, j, k = wp.tid()
    
    # Boundary check (simple 0 for boundaries for now)
    dim_x = phi.shape[0]
    dim_y = phi.shape[1]
    dim_z = phi.shape[2]
    if i < 2 or i >= dim_x - 2 or j < 2 or j >= dim_y - 2 or k < 2 or k >= dim_z - 2:
        return

    # Extract local values
    alp = alpha[i, j, k]
    trK = K[i, j, k]
    # ... more extractions ...
    
    # Calculate RHS (Simplified for flat spacetime / linear wave first)
    
    # 1. dt(phi) = -1/6 * alpha * K + 1/6 * div(beta)
    # div(beta) = d_x beta^x + ...
    div_beta = d_dx_comp(beta, i, j, k, 0, dx) + d_dy_comp(beta, i, j, k, 1, dy) + d_dz_comp(beta, i, j, k, 2, dz)
    
    rhs_phi[i, j, k] = -1.0/6.0 * alp * trK + 1.0/6.0 * div_beta
    
    # 2. dt(gamma_tilde_ij) = -2 alpha A_tilde_ij
    # Ignoring advection and shift terms for flat start
    for c in range(6):
        rhs_gamma_tilde[i, j, k, c] = -2.0 * alp * A_tilde[i, j, k, c]
        
    # 3. dt(K) = -D^2 alpha + ...
    # D^2 alpha = laplacian(alpha) in flat space
    lap_alpha = d2_dx2(alpha, i, j, k, dx) + d2_dy2(alpha, i, j, k, dy) + d2_dz2(alpha, i, j, k, dz)
    rhs_K[i, j, k] = -lap_alpha
    
    # 4. dt(A_tilde_ij) = ...
    # ... complex terms ...
    
    # 5. Gauge evolution
    # dt(alpha) = -2 alpha K
    rhs_alpha[i, j, k] = -2.0 * alp * trK
    
    # dt(beta) = 3/4 B
    for c in range(3):
        rhs_beta[i, j, k, c] = 0.75 * B[i, j, k, c]
        rhs_B[i, j, k, c] = 0.0 # simplified

    # Add Dissipation
    sigma = 0.01 # Dissipation strength
    rhs_phi[i, j, k] += kreiss_oliger_dissipation(phi, i, j, k, sigma)
    rhs_K[i, j, k] += kreiss_oliger_dissipation(K, i, j, k, sigma)
    rhs_alpha[i, j, k] += kreiss_oliger_dissipation(alpha, i, j, k, sigma)
    
    for c in range(6):
        rhs_gamma_tilde[i, j, k, c] += kreiss_oliger_dissipation_comp(gamma_tilde, i, j, k, c, sigma)
        rhs_A_tilde[i, j, k, c] += kreiss_oliger_dissipation_comp(A_tilde, i, j, k, c, sigma)
        
    for c in range(3):
        rhs_Gam_tilde[i, j, k, c] += kreiss_oliger_dissipation_comp(Gam_tilde, i, j, k, c, sigma)
        rhs_beta[i, j, k, c] += kreiss_oliger_dissipation_comp(beta, i, j, k, c, sigma)
        rhs_B[i, j, k, c] += kreiss_oliger_dissipation_comp(B, i, j, k, c, sigma)
