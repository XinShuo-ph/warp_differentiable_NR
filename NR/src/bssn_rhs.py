"""
BSSN RHS computation - simplified version for flat spacetime testing.

Starting with flat spacetime (Minkowski) and simple gauge conditions.
"""

import warp as wp
from bssn_derivatives import (
    deriv_x_4th, deriv_y_4th, deriv_z_4th,
    deriv2_x_4th, deriv2_y_4th, deriv2_z_4th,
    deriv2_xy_4th, deriv2_xz_4th, deriv2_yz_4th,
    advection, dissipation_5th
)


@wp.kernel
def compute_bssn_rhs_simple(
    # Input: current state
    phi: wp.array3d(dtype=float),
    gt_xx: wp.array3d(dtype=float),
    gt_xy: wp.array3d(dtype=float),
    gt_xz: wp.array3d(dtype=float),
    gt_yy: wp.array3d(dtype=float),
    gt_yz: wp.array3d(dtype=float),
    gt_zz: wp.array3d(dtype=float),
    At_xx: wp.array3d(dtype=float),
    At_xy: wp.array3d(dtype=float),
    At_xz: wp.array3d(dtype=float),
    At_yy: wp.array3d(dtype=float),
    At_yz: wp.array3d(dtype=float),
    At_zz: wp.array3d(dtype=float),
    Gamma_x: wp.array3d(dtype=float),
    Gamma_y: wp.array3d(dtype=float),
    Gamma_z: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    beta_x: wp.array3d(dtype=float),
    beta_y: wp.array3d(dtype=float),
    beta_z: wp.array3d(dtype=float),
    # Output: RHS
    phi_rhs: wp.array3d(dtype=float),
    gt_xx_rhs: wp.array3d(dtype=float),
    gt_xy_rhs: wp.array3d(dtype=float),
    gt_xz_rhs: wp.array3d(dtype=float),
    gt_yy_rhs: wp.array3d(dtype=float),
    gt_yz_rhs: wp.array3d(dtype=float),
    gt_zz_rhs: wp.array3d(dtype=float),
    At_xx_rhs: wp.array3d(dtype=float),
    At_xy_rhs: wp.array3d(dtype=float),
    At_xz_rhs: wp.array3d(dtype=float),
    At_yy_rhs: wp.array3d(dtype=float),
    At_yz_rhs: wp.array3d(dtype=float),
    At_zz_rhs: wp.array3d(dtype=float),
    Gamma_x_rhs: wp.array3d(dtype=float),
    Gamma_y_rhs: wp.array3d(dtype=float),
    Gamma_z_rhs: wp.array3d(dtype=float),
    K_rhs: wp.array3d(dtype=float),
    alpha_rhs: wp.array3d(dtype=float),
    beta_x_rhs: wp.array3d(dtype=float),
    beta_y_rhs: wp.array3d(dtype=float),
    beta_z_rhs: wp.array3d(dtype=float),
    # Grid parameters
    idx: float,
    idy: float,
    idz: float,
    eps_diss: float
):
    """
    Compute RHS of BSSN evolution equations.
    
    Simplified version for testing:
    - Flat spacetime initial data
    - 1+log slicing: d_t alpha = -2 alpha K
    - Gamma driver shift: d_t beta^i = B^i = 3/4 Gamma^i
    - No matter sources
    """
    i, j, k = wp.tid()
    
    # Skip boundaries (need 3 points for dissipation)
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    
    if i < 3 or i >= nx - 3 or j < 3 or j >= ny - 3 or k < 3 or k >= nz - 3:
        return
    
    # Read current values at this point
    phi_val = phi[i, j, k]
    K_val = K[i, j, k]
    alpha_val = alpha[i, j, k]
    beta_x_val = beta_x[i, j, k]
    beta_y_val = beta_y[i, j, k]
    beta_z_val = beta_z[i, j, k]
    
    # Conformal metric components
    gxx = gt_xx[i, j, k]
    gxy = gt_xy[i, j, k]
    gxz = gt_xz[i, j, k]
    gyy = gt_yy[i, j, k]
    gyz = gt_yz[i, j, k]
    gzz = gt_zz[i, j, k]
    
    # Traceless extrinsic curvature components
    Axx = At_xx[i, j, k]
    Axy = At_xy[i, j, k]
    Axz = At_xz[i, j, k]
    Ayy = At_yy[i, j, k]
    Ayz = At_yz[i, j, k]
    Azz = At_zz[i, j, k]
    
    # Christoffel symbols
    Gx = Gamma_x[i, j, k]
    Gy = Gamma_y[i, j, k]
    Gz = Gamma_z[i, j, k]
    
    # Compute derivatives needed for RHS
    # Derivatives of beta
    dbeta_x_dx = deriv_x_4th(beta_x, i, j, k, idx)
    dbeta_x_dy = deriv_y_4th(beta_x, i, j, k, idy)
    dbeta_x_dz = deriv_z_4th(beta_x, i, j, k, idz)
    
    dbeta_y_dx = deriv_x_4th(beta_y, i, j, k, idx)
    dbeta_y_dy = deriv_y_4th(beta_y, i, j, k, idy)
    dbeta_y_dz = deriv_z_4th(beta_y, i, j, k, idz)
    
    dbeta_z_dx = deriv_x_4th(beta_z, i, j, k, idx)
    dbeta_z_dy = deriv_y_4th(beta_z, i, j, k, idy)
    dbeta_z_dz = deriv_z_4th(beta_z, i, j, k, idz)
    
    # Divergence of shift
    div_beta = dbeta_x_dx + dbeta_y_dy + dbeta_z_dz
    
    # ===== PHI EVOLUTION =====
    # d_t phi = -1/6 alpha K + 1/6 div(beta) + beta^i d_i phi + dissipation
    phi_rhs[i, j, k] = (
        -alpha_val * K_val / 6.0
        + div_beta / 6.0
        + advection(phi, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(phi, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # ===== CONFORMAL METRIC EVOLUTION =====
    # d_t g_ij = -2 alpha A_ij + g_ik d_j beta^k + g_jk d_i beta^k - 2/3 g_ij d_k beta^k
    #          + beta^k d_k g_ij + dissipation
    
    # xx component
    gt_xx_rhs[i, j, k] = (
        -2.0 * alpha_val * Axx
        + 2.0 * gxx * dbeta_x_dx
        + 2.0 * gxy * dbeta_x_dy
        + 2.0 * gxz * dbeta_x_dz
        - 2.0 / 3.0 * gxx * div_beta
        + advection(gt_xx, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(gt_xx, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # xy component
    gt_xy_rhs[i, j, k] = (
        -2.0 * alpha_val * Axy
        + gxx * dbeta_y_dx + gxy * dbeta_x_dx
        + gxy * dbeta_y_dy + gyy * dbeta_x_dy
        + gxz * dbeta_y_dz + gyz * dbeta_x_dz
        - 2.0 / 3.0 * gxy * div_beta
        + advection(gt_xy, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(gt_xy, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # xz component  
    gt_xz_rhs[i, j, k] = (
        -2.0 * alpha_val * Axz
        + gxx * dbeta_z_dx + gxz * dbeta_x_dx
        + gxy * dbeta_z_dy + gyz * dbeta_x_dy
        + gxz * dbeta_z_dz + gzz * dbeta_x_dz
        - 2.0 / 3.0 * gxz * div_beta
        + advection(gt_xz, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(gt_xz, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # yy component
    gt_yy_rhs[i, j, k] = (
        -2.0 * alpha_val * Ayy
        + 2.0 * gxy * dbeta_y_dx
        + 2.0 * gyy * dbeta_y_dy
        + 2.0 * gyz * dbeta_y_dz
        - 2.0 / 3.0 * gyy * div_beta
        + advection(gt_yy, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(gt_yy, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # yz component
    gt_yz_rhs[i, j, k] = (
        -2.0 * alpha_val * Ayz
        + gxy * dbeta_z_dx + gxz * dbeta_y_dx
        + gyy * dbeta_z_dy + gyz * dbeta_y_dy
        + gyz * dbeta_z_dz + gzz * dbeta_y_dz
        - 2.0 / 3.0 * gyz * div_beta
        + advection(gt_yz, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(gt_yz, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # zz component
    gt_zz_rhs[i, j, k] = (
        -2.0 * alpha_val * Azz
        + 2.0 * gxz * dbeta_z_dx
        + 2.0 * gyz * dbeta_z_dy
        + 2.0 * gzz * dbeta_z_dz
        - 2.0 / 3.0 * gzz * div_beta
        + advection(gt_zz, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(gt_zz, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # ===== EXTRINSIC CURVATURE (simplified, flat spacetime) =====
    # For flat spacetime test, A_ij and K should remain zero
    # Set RHS to zero for now (will add proper terms later)
    At_xx_rhs[i, j, k] = dissipation_5th(At_xx, i, j, k, idx, idy, idz, eps_diss)
    At_xy_rhs[i, j, k] = dissipation_5th(At_xy, i, j, k, idx, idy, idz, eps_diss)
    At_xz_rhs[i, j, k] = dissipation_5th(At_xz, i, j, k, idx, idy, idz, eps_diss)
    At_yy_rhs[i, j, k] = dissipation_5th(At_yy, i, j, k, idx, idy, idz, eps_diss)
    At_yz_rhs[i, j, k] = dissipation_5th(At_yz, i, j, k, idx, idy, idz, eps_diss)
    At_zz_rhs[i, j, k] = dissipation_5th(At_zz, i, j, k, idx, idy, idz, eps_diss)
    
    # ===== TRACE K (simplified) =====
    # For flat spacetime, K should remain zero
    K_rhs[i, j, k] = dissipation_5th(K, i, j, k, idx, idy, idz, eps_diss)
    
    # ===== GAMMA (contracted Christoffel, simplified) =====
    # For flat spacetime, Gamma should remain zero
    Gamma_x_rhs[i, j, k] = dissipation_5th(Gamma_x, i, j, k, idx, idy, idz, eps_diss)
    Gamma_y_rhs[i, j, k] = dissipation_5th(Gamma_y, i, j, k, idx, idy, idz, eps_diss)
    Gamma_z_rhs[i, j, k] = dissipation_5th(Gamma_z, i, j, k, idx, idy, idz, eps_diss)
    
    # ===== LAPSE (1+log slicing) =====
    # d_t alpha = -2 alpha K + beta^i d_i alpha + dissipation
    alpha_rhs[i, j, k] = (
        -2.0 * alpha_val * K_val
        + advection(alpha, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(alpha, i, j, k, idx, idy, idz, eps_diss)
    )
    
    # ===== SHIFT (Gamma driver, simplified) =====
    # d_t beta^i = 3/4 Gamma^i + beta^j d_j beta^i + dissipation
    beta_x_rhs[i, j, k] = (
        0.75 * Gx
        + advection(beta_x, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(beta_x, i, j, k, idx, idy, idz, eps_diss)
    )
    
    beta_y_rhs[i, j, k] = (
        0.75 * Gy
        + advection(beta_y, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(beta_y, i, j, k, idx, idy, idz, eps_diss)
    )
    
    beta_z_rhs[i, j, k] = (
        0.75 * Gz
        + advection(beta_z, beta_x_val, beta_y_val, beta_z_val, i, j, k, idx, idy, idz)
        + dissipation_5th(beta_z, i, j, k, idx, idy, idz, eps_diss)
    )
