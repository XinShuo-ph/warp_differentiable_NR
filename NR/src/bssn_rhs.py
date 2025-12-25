import warp as wp
from NR.src.derivatives import (
    D_x, D_y, D_z, 
    D_x_vec3, D_y_vec3, D_z_vec3,
    D_x_mat33, D_y_mat33, D_z_mat33,
    ko_dissipation_scalar, ko_dissipation_vec3, ko_dissipation_mat33
)

@wp.func
def trace(m: wp.mat33):
    return m[0,0] + m[1,1] + m[2,2]

@wp.func
def calc_div_beta(beta: wp.array(dtype=wp.vec3, ndim=3), i: int, j: int, k: int, dx: float, dy: float, dz: float):
    # div beta = d_i beta^i
    return (D_x_vec3(beta, i, j, k, dx)[0] + 
            D_y_vec3(beta, i, j, k, dy)[1] + 
            D_z_vec3(beta, i, j, k, dz)[2])

@wp.func
def calc_lie_beta_gt(gt: wp.array(dtype=wp.mat33, ndim=3), beta: wp.array(dtype=wp.vec3, ndim=3), i: int, j: int, k: int, dx: float, dy: float, dz: float):
    # beta^k d_k gt_ij + gt_ik d_j beta^k + gt_kj d_i beta^k
    
    b = beta[i,j,k]
    
    d_x_gt = D_x_mat33(gt, i, j, k, dx)
    d_y_gt = D_y_mat33(gt, i, j, k, dy)
    d_z_gt = D_z_mat33(gt, i, j, k, dz)
    
    advect = b[0] * d_x_gt + b[1] * d_y_gt + b[2] * d_z_gt
    
    d_x_beta = D_x_vec3(beta, i, j, k, dx) # d_x beta^0, d_x beta^1, d_x beta^2
    d_y_beta = D_y_vec3(beta, i, j, k, dy)
    d_z_beta = D_z_vec3(beta, i, j, k, dz)
    
    l_gt = gt[i,j,k]
    
    # Result accumulator
    res = advect
    
    # Add terms: gt_ik * d_j beta^k + gt_kj * d_i beta^k
    # i, j here are tensor indices, not grid indices
    
    # We iterate over matrix components u, v (0..2)
    # res[u, v] += sum_w (gt[u, w] * d_v beta^w) + sum_w (gt[w, v] * d_u beta^w)
    
    # Row 0
    # res[0,0] += (gt[0,0]*d_x_beta[0] + gt[0,1]*d_x_beta[1] + gt[0,2]*d_x_beta[2]) + (gt[0,0]*d_x_beta[0] + gt[1,0]*d_x_beta[1] + gt[2,0]*d_x_beta[2])
    # ...
    
    # d_v beta^w corresponds to:
    # v=0 (x): d_x_beta[w]
    # v=1 (y): d_y_beta[w]
    # v=2 (z): d_z_beta[w]
    
    # Let's unroll for performance and clarity
    
    # Helper to get d_v beta^w
    # dbeta[v][w]
    dbeta_0 = d_x_beta
    dbeta_1 = d_y_beta
    dbeta_2 = d_z_beta
    
    # Manual loop unroll (Warp loops inside kernels are fine too)
    for u in range(3):
        for v in range(3):
            # Term 2: sum_w gt[u, w] * d_v beta^w
            term2 = 0.0
            term3 = 0.0
            for w in range(3):
                # Select d_v beta^w
                dv_bw = 0.0
                if v == 0: dv_bw = dbeta_0[w]
                elif v == 1: dv_bw = dbeta_1[w]
                else: dv_bw = dbeta_2[w]
                
                term2 += l_gt[u, w] * dv_bw
                
                # Term 3: sum_w gt[w, v] * d_u beta^w
                du_bw = 0.0
                if u == 0: du_bw = dbeta_0[w]
                elif u == 1: du_bw = dbeta_1[w]
                else: du_bw = dbeta_2[w]
                
                term3 += l_gt[w, v] * du_bw
            
            res[u, v] += term2 + term3
            
    return res

@wp.kernel
def bssn_rhs_kernel(
    phi: wp.array(dtype=float, ndim=3),
    gt: wp.array(dtype=wp.mat33, ndim=3),
    K: wp.array(dtype=float, ndim=3),
    At: wp.array(dtype=wp.mat33, ndim=3),
    Xt: wp.array(dtype=wp.vec3, ndim=3),
    alpha: wp.array(dtype=float, ndim=3),
    beta: wp.array(dtype=wp.vec3, ndim=3),
    B: wp.array(dtype=wp.vec3, ndim=3),
    # RHS outputs
    dt_phi: wp.array(dtype=float, ndim=3),
    dt_gt: wp.array(dtype=wp.mat33, ndim=3),
    dt_K: wp.array(dtype=float, ndim=3),
    dt_At: wp.array(dtype=wp.mat33, ndim=3),
    dt_Xt: wp.array(dtype=wp.vec3, ndim=3),
    dt_alpha: wp.array(dtype=float, ndim=3),
    dt_beta: wp.array(dtype=wp.vec3, ndim=3),
    dt_B: wp.array(dtype=wp.vec3, ndim=3),
    # Parameters
    dx: float, dy: float, dz: float
):
    i, j, k = wp.tid()
    
    # Local vars
    l_alpha = alpha[i,j,k]
    l_K = K[i,j,k]
    l_beta = beta[i,j,k]
    l_At = At[i,j,k]
    l_gt = gt[i,j,k]
    
    # 1. dt_phi
    div_beta = calc_div_beta(beta, i, j, k, dx, dy, dz)
    
    grad_phi_x = D_x(phi, i, j, k, dx)
    grad_phi_y = D_y(phi, i, j, k, dy)
    grad_phi_z = D_z(phi, i, j, k, dz)
    
    advect_phi = l_beta[0] * grad_phi_x + l_beta[1] * grad_phi_y + l_beta[2] * grad_phi_z
    
    dt_phi[i,j,k] = -0.166666667 * l_alpha * l_K + advect_phi + 0.166666667 * div_beta + ko_dissipation_scalar(phi, i, j, k, 0.1)
    
    # 2. dt_gt
    # -2 alpha At + Lie_beta gt - 2/3 gt div_beta
    lie_gt = calc_lie_beta_gt(gt, beta, i, j, k, dx, dy, dz)
    
    dt_gt[i,j,k] = -2.0 * l_alpha * l_At + lie_gt - 0.666666667 * l_gt * div_beta + ko_dissipation_mat33(gt, i, j, k, 0.1)
    
    # 3. dt_alpha (1+log)
    # -2 alpha K + beta^i d_i alpha
    grad_alpha_x = D_x(alpha, i, j, k, dx)
    grad_alpha_y = D_y(alpha, i, j, k, dy)
    grad_alpha_z = D_z(alpha, i, j, k, dz)
    advect_alpha = l_beta[0] * grad_alpha_x + l_beta[1] * grad_alpha_y + l_beta[2] * grad_alpha_z
    
    dt_alpha[i,j,k] = -2.0 * l_alpha * l_K + advect_alpha + ko_dissipation_scalar(alpha, i, j, k, 0.1)
    
    # 4. dt_beta (Gamma-driver)
    # 3/4 B + beta^j d_j beta^i
    l_B = B[i,j,k]
    
    d_x_beta = D_x_vec3(beta, i, j, k, dx)
    d_y_beta = D_y_vec3(beta, i, j, k, dy)
    d_z_beta = D_z_vec3(beta, i, j, k, dz)
    
    advect_beta = l_beta[0] * d_x_beta + l_beta[1] * d_y_beta + l_beta[2] * d_z_beta
    
    dt_beta[i,j,k] = 0.75 * l_B + advect_beta + ko_dissipation_vec3(beta, i, j, k, 0.1)
    
    # 5. dt_K (Simplified for flat space verification: -D^2 alpha term is 0 if alpha=1)
    # dt_K = -D^2 alpha + alpha(A_ij A^ij + 1/3 K^2) + beta^i d_i K
    # For initial implementation, we assume alpha is constant 1, so D^2 alpha = 0.
    # We should implement D^2 alpha properly later.
    
    sq_At = wp.mat33(0.0) # A_ij A^ij needs metric inverse to raise indices?
    # A^ij = g^ik g^jl A_kl
    # For now, assume flat metric for tensor contractions if needed or implement inverse.
    # But wait, we have dynamic metric.
    # Inverse of 3x3 matrix:
    det_gt = wp.determinant(l_gt)
    inv_gt = wp.inverse(l_gt) # Warp has inverse for mat33? Yes.
    
    # Contraction A_ij A^ij = A_ij g^ik g^jl A_kl
    # This is expensive.
    tr_A2 = 0.0
    for u in range(3):
        for v in range(3):
            # A^uv
            A_upper_uv = 0.0
            for m in range(3):
                for n in range(3):
                    A_upper_uv += inv_gt[u, m] * inv_gt[v, n] * l_At[m, n]
            
            tr_A2 += l_At[u, v] * A_upper_uv
            
    grad_K_x = D_x(K, i, j, k, dx)
    grad_K_y = D_y(K, i, j, k, dy)
    grad_K_z = D_z(K, i, j, k, dz)
    advect_K = l_beta[0] * grad_K_x + l_beta[1] * grad_K_y + l_beta[2] * grad_K_z
    
    # Laplacian D^2 alpha = g^ij D_i D_j alpha = g^ij (d_i d_j alpha - Gamma^k_ij d_k alpha)
    # Since alpha is spatially constant (initially), D^2 alpha = 0.
    # Implement full Laplacian later.
    laplace_alpha = 0.0 
    
    dt_K[i,j,k] = -laplace_alpha + l_alpha * (tr_A2 + 0.333333333 * l_K * l_K) + advect_K + ko_dissipation_scalar(K, i, j, k, 0.1)

    # 6. dt_At, dt_Xt, dt_B set to 0 for now (or simple advection)
    # to verify basic stability of flat space first.
    dt_At[i,j,k] = wp.mat33(0.0) + ko_dissipation_mat33(At, i, j, k, 0.1)
    dt_Xt[i,j,k] = wp.vec3(0.0) + ko_dissipation_vec3(Xt, i, j, k, 0.1)
    dt_B[i,j,k] = wp.vec3(0.0) + ko_dissipation_vec3(B, i, j, k, 0.1)
