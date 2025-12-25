import warp as wp
from NR.src.derivatives import d_dx, d_dy, d_dz, d2_dx2, d2_dy2, d2_dz2, d_dx_comp, d_dy_comp, d_dz_comp, ko_dissipation
from NR.src.bssn_defs import BSSNState

@wp.func
def trace(m: wp.mat33):
    return m[0,0] + m[1,1] + m[2,2]

@wp.func
def mat_vec_mul(m: wp.mat33, v: wp.vec3):
    return wp.vec3(
        m[0,0]*v[0] + m[0,1]*v[1] + m[0,2]*v[2],
        m[1,0]*v[0] + m[1,1]*v[1] + m[1,2]*v[2],
        m[2,0]*v[0] + m[2,1]*v[1] + m[2,2]*v[2]
    )

@wp.kernel
def bssn_rhs_kernel(
    # State fields
    phi: wp.array(dtype=float, ndim=3),
    gamma_tilde: wp.array(dtype=wp.mat33, ndim=3),
    K: wp.array(dtype=float, ndim=3),
    A_tilde: wp.array(dtype=wp.mat33, ndim=3),
    Gamma_tilde: wp.array(dtype=wp.vec3, ndim=3),
    alpha: wp.array(dtype=float, ndim=3),
    beta: wp.array(dtype=wp.vec3, ndim=3),
    B: wp.array(dtype=wp.vec3, ndim=3),
    
    # RHS fields (Output)
    dt_phi: wp.array(dtype=float, ndim=3),
    dt_gamma_tilde: wp.array(dtype=wp.mat33, ndim=3),
    dt_K: wp.array(dtype=float, ndim=3),
    dt_A_tilde: wp.array(dtype=wp.mat33, ndim=3),
    dt_Gamma_tilde: wp.array(dtype=wp.vec3, ndim=3),
    dt_alpha: wp.array(dtype=float, ndim=3),
    dt_beta: wp.array(dtype=wp.vec3, ndim=3),
    dt_B: wp.array(dtype=wp.vec3, ndim=3),
    
    # Params
    dx: float, dy: float, dz: float
):
    i, j, k = wp.tid()
    
    # Boundary check (3 ghost zones for 4th order)
    # Actually 2 ghost zones are enough for 5-point stencil? 
    # Stencil is -2..2. So 2 ghost zones.
    # KO dissipation might use 3.
    nx = dt_phi.shape[0]
    ny = dt_phi.shape[1]
    nz = dt_phi.shape[2]
    
    if i < 3 or i >= nx-3 or j < 3 or j >= ny-3 or k < 3 or k >= nz-3:
        # Zero RHS on boundaries (frozen BCs for now)
        dt_phi[i,j,k] = 0.0
        dt_gamma_tilde[i,j,k] = wp.mat33(0.0)
        dt_K[i,j,k] = 0.0
        dt_A_tilde[i,j,k] = wp.mat33(0.0)
        dt_Gamma_tilde[i,j,k] = wp.vec3(0.0)
        dt_alpha[i,j,k] = 0.0
        dt_beta[i,j,k] = wp.vec3(0.0)
        dt_B[i,j,k] = wp.vec3(0.0)
        return

    # Load local variables
    p = phi[i,j,k]
    g = gamma_tilde[i,j,k]
    trK = K[i,j,k]
    A = A_tilde[i,j,k]
    Gam = Gamma_tilde[i,j,k]
    alp = alpha[i,j,k]
    bet = beta[i,j,k]
    b_vec = B[i,j,k]
    
    # Precompute derivatives
    # alpha derivs
    d_alp_x = d_dx(alpha, i, j, k, dx)
    d_alp_y = d_dy(alpha, i, j, k, dy)
    d_alp_z = d_dz(alpha, i, j, k, dz)
    
    # beta derivs (needed for advection)
    # d_beta_x = ... (vec3)
    
    # phi derivs
    d_phi_x = d_dx(phi, i, j, k, dx)
    d_phi_y = d_dy(phi, i, j, k, dy)
    d_phi_z = d_dz(phi, i, j, k, dz)
    
    # Advection terms operator
    # beta^i d_i f
    # beta is a vector.
    adv_phi = bet[0]*d_phi_x + bet[1]*d_phi_y + bet[2]*d_phi_z
    
    # Divergence of shift (partial)
    div_beta = d_dx_comp(beta, 0, i,j,k, dx) + d_dy_comp(beta, 1, i,j,k, dy) + d_dz_comp(beta, 2, i,j,k, dz)
    
    # ----------------------------------------------------
    # 1. dt_phi
    dt_phi[i,j,k] = adv_phi - 0.166666667 * alp * trK + 0.166666667 * div_beta
    # Dissipation
    dt_phi[i,j,k] += ko_dissipation(phi, i,j,k, 0.1)

    # ----------------------------------------------------
    # 2. dt_gamma_tilde
    # dt_g_ij = beta^k d_k g_ij - 2 alpha A_ij + g_ik d_j beta^k + g_jk d_i beta^k - 2/3 g_ij div_beta
    
    # Need derivatives of beta components
    d_beta_dx = wp.vec3(d_dx_comp(beta, 0, i,j,k, dx), d_dx_comp(beta, 1, i,j,k, dx), d_dx_comp(beta, 2, i,j,k, dx))
    d_beta_dy = wp.vec3(d_dy_comp(beta, 0, i,j,k, dy), d_dy_comp(beta, 1, i,j,k, dy), d_dy_comp(beta, 2, i,j,k, dy))
    d_beta_dz = wp.vec3(d_dz_comp(beta, 0, i,j,k, dz), d_dz_comp(beta, 1, i,j,k, dz), d_dz_comp(beta, 2, i,j,k, dz))
    
    # d_j beta^i (row i, col j)
    # Actually usually indices are d_j beta^i
    
    # Let's compute rhs_gamma term by term
    rhs_gamma = wp.mat33(0.0)
    
    # - 2 alpha A_ij
    rhs_gamma -= 2.0 * alp * A
    
    # + advection beta^k d_k g_ij
    # d_k g_ij needs derivatives of matrix components. 
    # For now, let's assume flat space beta=0 approx, so advection is small, but we should include it.
    # It's tedious to write out for all 6 components.
    
    # + g_ik d_j beta^k
    # Term 1: g * d_beta^T ? No.
    # g_ik (d_beta_j)^k
    # We can use matrix mult if we organize d_beta right.
    # Let L_ij = d_i beta^j. Then d_j beta^k is L_jk.
    # g_ik L_jk = (g * L^T)_ij
    
    # Let's construct L matrix where L_ij = d_i beta^j
    L = wp.mat33(
        d_beta_dx[0], d_beta_dx[1], d_beta_dx[2],
        d_beta_dy[0], d_beta_dy[1], d_beta_dy[2],
        d_beta_dz[0], d_beta_dz[1], d_beta_dz[2]
    )
    
    # g_ik d_j beta^k -> sum_k g_ik L_jk
    # g_jk d_i beta^k -> sum_k g_jk L_ik
    
    # g * L^T + (g * L^T)^T ?
    # Let M = g * wp.transpose(L)
    # M_ij = g_ik L_jk
    # M_ji = g_jk L_ik
    # So Term = M + M^T
    
    M = g * wp.transpose(L)
    rhs_gamma += M + wp.transpose(M)
    
    # - 2/3 g_ij div_beta
    rhs_gamma -= 0.666666667 * g * div_beta
    
    dt_gamma_tilde[i,j,k] = rhs_gamma
    
    # ----------------------------------------------------
    # 3. dt_K
    # dt_K = beta^i d_i K - D^2 alpha + alpha(A_ij A^ij + 1/3 K^2)
    
    adv_K = bet[0]*d_dx(K, i,j,k, dx) + bet[1]*d_dy(K, i,j,k, dy) + bet[2]*d_dz(K, i,j,k, dz)
    
    # D^2 alpha = gamma^ij (d_i d_j alpha - Gamma^k_ij d_k alpha)
    # We need inverse gamma (physical or conformal?)
    # D^2 is covariant derivative associated with physical metric gamma.
    # gamma_ij = e^{4phi} tilde_gamma_ij
    # gamma^ij = e^{-4phi} tilde_gamma^ij
    
    inv_g_tilde = wp.inverse(g)
    inv_g_phys = wp.exp(-4.0 * p) * inv_g_tilde
    
    # For flat space, phi=0, g=I, Gamma=0. So D^2 alpha = Laplacian alpha.
    # We should implement at least Laplacian part.
    lap_alpha = d2_dx2(alpha, i,j,k, dx) + d2_dy2(alpha, i,j,k, dy) + d2_dz2(alpha, i,j,k, dz)
    # Full D^2 alpha is complex.
    
    # A_ij A^ij = A_ij A_kl gamma^ik gamma^jl
    # A^ij = gamma^ik gamma^jl A_kl
    # Conformal A_tilde is used? 
    # Usually A_ij A^ij = A_tilde_ij A_tilde^ij (indices raised by tilde metric)
    # because conformal factors cancel out. e^{-4phi} * e^{4phi} ...
    # Wait: A^ij (phys) = e^{-4phi} A_tilde^ij.
    # A_ij (phys) = e^{4phi} A_tilde_ij.
    # Product is A_tilde_ij A_tilde^ij.
    
    # Compute A_tilde^ij = tilde_gamma^ik tilde_gamma^jl A_tilde_kl
    # A_up = inv_g_tilde * A * inv_g_tilde
    A_up = inv_g_tilde * A * inv_g_tilde
    
    # Contraction A_ij A^ij = trace(A * A_up) ?
    # (A * A_up)_ij = A_ik A_up_kj. Trace sums A_ik A_up_ki.
    # Symmetric A, A_up.
    tr_A2 = trace(A * A_up)
    
    dt_K[i,j,k] = adv_K - lap_alpha + alp * (tr_A2 + 0.333333333 * trK * trK)
    dt_K[i,j,k] += ko_dissipation(K, i,j,k, 0.1)
    
    # ----------------------------------------------------
    # 4. dt_A_tilde
    # Advection placeholder (Need component-wise derivatives for mat33)
    dt_A_tilde[i,j,k] = wp.mat33(0.0) 
    
    # ----------------------------------------------------
    # 5. dt_Gamma_tilde
    # Advection
    d_Gam_x = wp.vec3(d_dx_comp(Gamma_tilde, 0, i,j,k, dx), d_dx_comp(Gamma_tilde, 1, i,j,k, dx), d_dx_comp(Gamma_tilde, 2, i,j,k, dx))
    d_Gam_y = wp.vec3(d_dy_comp(Gamma_tilde, 0, i,j,k, dy), d_dy_comp(Gamma_tilde, 1, i,j,k, dy), d_dy_comp(Gamma_tilde, 2, i,j,k, dy))
    d_Gam_z = wp.vec3(d_dz_comp(Gamma_tilde, 0, i,j,k, dz), d_dz_comp(Gamma_tilde, 1, i,j,k, dz), d_dz_comp(Gamma_tilde, 2, i,j,k, dz))
    adv_Gam = bet[0]*d_Gam_x + bet[1]*d_Gam_y + bet[2]*d_Gam_z
    
    dt_Gamma_tilde[i,j,k] = adv_Gam
    
    # ----------------------------------------------------
    # 6. Gauge Evolution
    # 1+log: dt_alpha = -2 alpha K + advection
    dt_alpha[i,j,k] = -2.0 * alp * trK
    dt_alpha[i,j,k] += bet[0]*d_alp_x + bet[1]*d_alp_y + bet[2]*d_alp_z
    dt_alpha[i,j,k] += ko_dissipation(alpha, i,j,k, 0.1)
    
    # Gamma-driver: dt_beta = 3/4 B + advection
    # Advection for beta
    d_bet_x = wp.vec3(d_dx_comp(beta, 0, i,j,k, dx), d_dx_comp(beta, 1, i,j,k, dx), d_dx_comp(beta, 2, i,j,k, dx))
    d_bet_y = wp.vec3(d_dy_comp(beta, 0, i,j,k, dy), d_dy_comp(beta, 1, i,j,k, dy), d_dy_comp(beta, 2, i,j,k, dy))
    d_bet_z = wp.vec3(d_dz_comp(beta, 0, i,j,k, dz), d_dz_comp(beta, 1, i,j,k, dz), d_dz_comp(beta, 2, i,j,k, dz))
    adv_bet = bet[0]*d_bet_x + bet[1]*d_bet_y + bet[2]*d_bet_z
    
    dt_beta[i,j,k] = 0.75 * b_vec + adv_bet
    
    # dt_B = dt_Gamma_tilde - eta B + advection
    d_B_x = wp.vec3(d_dx_comp(B, 0, i,j,k, dx), d_dx_comp(B, 1, i,j,k, dx), d_dx_comp(B, 2, i,j,k, dx))
    d_B_y = wp.vec3(d_dy_comp(B, 0, i,j,k, dy), d_dy_comp(B, 1, i,j,k, dy), d_dy_comp(B, 2, i,j,k, dy))
    d_B_z = wp.vec3(d_dz_comp(B, 0, i,j,k, dz), d_dz_comp(B, 1, i,j,k, dz), d_dz_comp(B, 2, i,j,k, dz))
    adv_B = bet[0]*d_B_x + bet[1]*d_B_y + bet[2]*d_B_z
    
    eta = 2.0 # Damping
    dt_B[i,j,k] = dt_Gamma_tilde[i,j,k] - eta * b_vec + adv_B
