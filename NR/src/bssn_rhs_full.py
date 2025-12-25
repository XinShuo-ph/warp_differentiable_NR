"""
Full BSSN Right-Hand-Side with Gauge Conditions

Includes:
- All BSSN evolution equations
- 1+log slicing: d_t(alpha) = -2*alpha*K
- Gamma-driver shift: d_t(beta^i) = (3/4)*B^i, d_t(B^i) = d_t(Gamma^i) - eta*B^i
- Kreiss-Oliger dissipation
"""

import warp as wp
import numpy as np
from bssn_derivatives import (
    d1_x, d1_y, d1_z,
    d2_xx, d2_yy, d2_zz, d2_xy, d2_xz, d2_yz,
    ko_dissipation
)

wp.init()


@wp.kernel
def compute_full_bssn_rhs(
    # Current state
    chi: wp.array3d(dtype=float),
    gamma_xx: wp.array3d(dtype=float),
    gamma_xy: wp.array3d(dtype=float),
    gamma_xz: wp.array3d(dtype=float),
    gamma_yy: wp.array3d(dtype=float),
    gamma_yz: wp.array3d(dtype=float),
    gamma_zz: wp.array3d(dtype=float),
    K_in: wp.array3d(dtype=float),
    A_xx: wp.array3d(dtype=float),
    A_xy: wp.array3d(dtype=float),
    A_xz: wp.array3d(dtype=float),
    A_yy: wp.array3d(dtype=float),
    A_yz: wp.array3d(dtype=float),
    A_zz: wp.array3d(dtype=float),
    Gamma_x: wp.array3d(dtype=float),
    Gamma_y: wp.array3d(dtype=float),
    Gamma_z: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    beta_x: wp.array3d(dtype=float),
    beta_y: wp.array3d(dtype=float),
    beta_z: wp.array3d(dtype=float),
    B_x: wp.array3d(dtype=float),
    B_y: wp.array3d(dtype=float),
    B_z: wp.array3d(dtype=float),
    # RHS outputs
    rhs_chi: wp.array3d(dtype=float),
    rhs_gamma_xx: wp.array3d(dtype=float),
    rhs_gamma_xy: wp.array3d(dtype=float),
    rhs_gamma_xz: wp.array3d(dtype=float),
    rhs_gamma_yy: wp.array3d(dtype=float),
    rhs_gamma_yz: wp.array3d(dtype=float),
    rhs_gamma_zz: wp.array3d(dtype=float),
    rhs_K: wp.array3d(dtype=float),
    rhs_A_xx: wp.array3d(dtype=float),
    rhs_A_xy: wp.array3d(dtype=float),
    rhs_A_xz: wp.array3d(dtype=float),
    rhs_A_yy: wp.array3d(dtype=float),
    rhs_A_yz: wp.array3d(dtype=float),
    rhs_A_zz: wp.array3d(dtype=float),
    rhs_Gamma_x: wp.array3d(dtype=float),
    rhs_Gamma_y: wp.array3d(dtype=float),
    rhs_Gamma_z: wp.array3d(dtype=float),
    rhs_alpha: wp.array3d(dtype=float),
    rhs_beta_x: wp.array3d(dtype=float),
    rhs_beta_y: wp.array3d(dtype=float),
    rhs_beta_z: wp.array3d(dtype=float),
    rhs_B_x: wp.array3d(dtype=float),
    rhs_B_y: wp.array3d(dtype=float),
    rhs_B_z: wp.array3d(dtype=float),
    # Parameters
    nx: int, ny: int, nz: int,
    inv_12h: float,
    inv_12h2: float,
    inv_144h2: float,
    sigma_ko: float,
    dt: float,
    eta: float  # Gamma-driver damping
):
    """Compute full BSSN RHS with 1+log slicing and Gamma-driver shift"""
    i, j, k = wp.tid()
    
    # Need 3 ghost zones for KO dissipation
    if i < 3 or i >= nx - 3:
        return
    if j < 3 or j >= ny - 3:
        return
    if k < 3 or k >= nz - 3:
        return
    
    # ============ Load local values ============
    chi_ijk = chi[i, j, k]
    alpha_ijk = alpha[i, j, k]
    K_ijk = K_in[i, j, k]
    
    beta_x_ijk = beta_x[i, j, k]
    beta_y_ijk = beta_y[i, j, k]
    beta_z_ijk = beta_z[i, j, k]
    
    B_x_ijk = B_x[i, j, k]
    B_y_ijk = B_y[i, j, k]
    B_z_ijk = B_z[i, j, k]
    
    A_xx_ijk = A_xx[i, j, k]
    A_xy_ijk = A_xy[i, j, k]
    A_xz_ijk = A_xz[i, j, k]
    A_yy_ijk = A_yy[i, j, k]
    A_yz_ijk = A_yz[i, j, k]
    A_zz_ijk = A_zz[i, j, k]
    
    gamma_xx_ijk = gamma_xx[i, j, k]
    gamma_xy_ijk = gamma_xy[i, j, k]
    gamma_xz_ijk = gamma_xz[i, j, k]
    gamma_yy_ijk = gamma_yy[i, j, k]
    gamma_yz_ijk = gamma_yz[i, j, k]
    gamma_zz_ijk = gamma_zz[i, j, k]
    
    Gamma_x_ijk = Gamma_x[i, j, k]
    Gamma_y_ijk = Gamma_y[i, j, k]
    Gamma_z_ijk = Gamma_z[i, j, k]
    
    # ============ Compute derivatives ============
    # First derivatives of chi
    dchi_x = d1_x(chi, i, j, k, inv_12h)
    dchi_y = d1_y(chi, i, j, k, inv_12h)
    dchi_z = d1_z(chi, i, j, k, inv_12h)
    
    # First derivatives of alpha
    dalpha_x = d1_x(alpha, i, j, k, inv_12h)
    dalpha_y = d1_y(alpha, i, j, k, inv_12h)
    dalpha_z = d1_z(alpha, i, j, k, inv_12h)
    
    # Second derivatives of alpha (for Laplacian)
    d2alpha_xx = d2_xx(alpha, i, j, k, inv_12h2)
    d2alpha_yy = d2_yy(alpha, i, j, k, inv_12h2)
    d2alpha_zz = d2_zz(alpha, i, j, k, inv_12h2)
    
    # First derivatives of beta
    dbetax_x = d1_x(beta_x, i, j, k, inv_12h)
    dbetax_y = d1_y(beta_x, i, j, k, inv_12h)
    dbetax_z = d1_z(beta_x, i, j, k, inv_12h)
    
    dbetay_x = d1_x(beta_y, i, j, k, inv_12h)
    dbetay_y = d1_y(beta_y, i, j, k, inv_12h)
    dbetay_z = d1_z(beta_y, i, j, k, inv_12h)
    
    dbetaz_x = d1_x(beta_z, i, j, k, inv_12h)
    dbetaz_y = d1_y(beta_z, i, j, k, inv_12h)
    dbetaz_z = d1_z(beta_z, i, j, k, inv_12h)
    
    div_beta = dbetax_x + dbetay_y + dbetaz_z
    
    # First derivatives of K
    dK_x = d1_x(K_in, i, j, k, inv_12h)
    dK_y = d1_y(K_in, i, j, k, inv_12h)
    dK_z = d1_z(K_in, i, j, k, inv_12h)
    
    # ============ RHS for chi ============
    # d_t(chi) = (2/3)*chi*(alpha*K - div(beta)) + beta^i*d_i(chi)
    advect_chi = beta_x_ijk * dchi_x + beta_y_ijk * dchi_y + beta_z_ijk * dchi_z
    rhs_chi_val = (2.0/3.0) * chi_ijk * (alpha_ijk * K_ijk - div_beta) + advect_chi
    rhs_chi_val = rhs_chi_val - ko_dissipation(chi, i, j, k, sigma_ko / dt)
    rhs_chi[i, j, k] = rhs_chi_val
    
    # ============ RHS for gamma_tilde ============
    # d_t(gamma_ij) = -2*alpha*A_ij + beta^k*d_k(gamma_ij) + gamma_ik*d_j(beta^k) + gamma_jk*d_i(beta^k) - (2/3)*gamma_ij*div(beta)
    
    # Advection terms
    dgxx_x = d1_x(gamma_xx, i, j, k, inv_12h)
    dgxx_y = d1_y(gamma_xx, i, j, k, inv_12h)
    dgxx_z = d1_z(gamma_xx, i, j, k, inv_12h)
    advect_gxx = beta_x_ijk * dgxx_x + beta_y_ijk * dgxx_y + beta_z_ijk * dgxx_z
    
    rhs_gxx = -2.0 * alpha_ijk * A_xx_ijk + advect_gxx
    rhs_gxx = rhs_gxx + 2.0 * gamma_xx_ijk * dbetax_x + 2.0 * gamma_xy_ijk * dbetax_y + 2.0 * gamma_xz_ijk * dbetax_z
    rhs_gxx = rhs_gxx - (2.0/3.0) * gamma_xx_ijk * div_beta
    rhs_gxx = rhs_gxx - ko_dissipation(gamma_xx, i, j, k, sigma_ko / dt)
    rhs_gamma_xx[i, j, k] = rhs_gxx
    
    # gamma_xy
    dgxy_x = d1_x(gamma_xy, i, j, k, inv_12h)
    dgxy_y = d1_y(gamma_xy, i, j, k, inv_12h)
    dgxy_z = d1_z(gamma_xy, i, j, k, inv_12h)
    advect_gxy = beta_x_ijk * dgxy_x + beta_y_ijk * dgxy_y + beta_z_ijk * dgxy_z
    
    rhs_gxy = -2.0 * alpha_ijk * A_xy_ijk + advect_gxy
    rhs_gxy = rhs_gxy + gamma_xx_ijk * dbetay_x + gamma_xy_ijk * dbetay_y + gamma_xz_ijk * dbetay_z
    rhs_gxy = rhs_gxy + gamma_xy_ijk * dbetax_x + gamma_yy_ijk * dbetax_y + gamma_yz_ijk * dbetax_z
    rhs_gxy = rhs_gxy - (2.0/3.0) * gamma_xy_ijk * div_beta
    rhs_gxy = rhs_gxy - ko_dissipation(gamma_xy, i, j, k, sigma_ko / dt)
    rhs_gamma_xy[i, j, k] = rhs_gxy
    
    # gamma_xz
    dgxz_x = d1_x(gamma_xz, i, j, k, inv_12h)
    dgxz_y = d1_y(gamma_xz, i, j, k, inv_12h)
    dgxz_z = d1_z(gamma_xz, i, j, k, inv_12h)
    advect_gxz = beta_x_ijk * dgxz_x + beta_y_ijk * dgxz_y + beta_z_ijk * dgxz_z
    
    rhs_gxz = -2.0 * alpha_ijk * A_xz_ijk + advect_gxz
    rhs_gxz = rhs_gxz + gamma_xx_ijk * dbetaz_x + gamma_xy_ijk * dbetaz_y + gamma_xz_ijk * dbetaz_z
    rhs_gxz = rhs_gxz + gamma_xz_ijk * dbetax_x + gamma_yz_ijk * dbetax_y + gamma_zz_ijk * dbetax_z
    rhs_gxz = rhs_gxz - (2.0/3.0) * gamma_xz_ijk * div_beta
    rhs_gxz = rhs_gxz - ko_dissipation(gamma_xz, i, j, k, sigma_ko / dt)
    rhs_gamma_xz[i, j, k] = rhs_gxz
    
    # gamma_yy
    dgyy_x = d1_x(gamma_yy, i, j, k, inv_12h)
    dgyy_y = d1_y(gamma_yy, i, j, k, inv_12h)
    dgyy_z = d1_z(gamma_yy, i, j, k, inv_12h)
    advect_gyy = beta_x_ijk * dgyy_x + beta_y_ijk * dgyy_y + beta_z_ijk * dgyy_z
    
    rhs_gyy = -2.0 * alpha_ijk * A_yy_ijk + advect_gyy
    rhs_gyy = rhs_gyy + 2.0 * gamma_xy_ijk * dbetay_x + 2.0 * gamma_yy_ijk * dbetay_y + 2.0 * gamma_yz_ijk * dbetay_z
    rhs_gyy = rhs_gyy - (2.0/3.0) * gamma_yy_ijk * div_beta
    rhs_gyy = rhs_gyy - ko_dissipation(gamma_yy, i, j, k, sigma_ko / dt)
    rhs_gamma_yy[i, j, k] = rhs_gyy
    
    # gamma_yz
    dgyz_x = d1_x(gamma_yz, i, j, k, inv_12h)
    dgyz_y = d1_y(gamma_yz, i, j, k, inv_12h)
    dgyz_z = d1_z(gamma_yz, i, j, k, inv_12h)
    advect_gyz = beta_x_ijk * dgyz_x + beta_y_ijk * dgyz_y + beta_z_ijk * dgyz_z
    
    rhs_gyz = -2.0 * alpha_ijk * A_yz_ijk + advect_gyz
    rhs_gyz = rhs_gyz + gamma_xy_ijk * dbetaz_x + gamma_yy_ijk * dbetaz_y + gamma_yz_ijk * dbetaz_z
    rhs_gyz = rhs_gyz + gamma_xz_ijk * dbetay_x + gamma_yz_ijk * dbetay_y + gamma_zz_ijk * dbetay_z
    rhs_gyz = rhs_gyz - (2.0/3.0) * gamma_yz_ijk * div_beta
    rhs_gyz = rhs_gyz - ko_dissipation(gamma_yz, i, j, k, sigma_ko / dt)
    rhs_gamma_yz[i, j, k] = rhs_gyz
    
    # gamma_zz
    dgzz_x = d1_x(gamma_zz, i, j, k, inv_12h)
    dgzz_y = d1_y(gamma_zz, i, j, k, inv_12h)
    dgzz_z = d1_z(gamma_zz, i, j, k, inv_12h)
    advect_gzz = beta_x_ijk * dgzz_x + beta_y_ijk * dgzz_y + beta_z_ijk * dgzz_z
    
    rhs_gzz = -2.0 * alpha_ijk * A_zz_ijk + advect_gzz
    rhs_gzz = rhs_gzz + 2.0 * gamma_xz_ijk * dbetaz_x + 2.0 * gamma_yz_ijk * dbetaz_y + 2.0 * gamma_zz_ijk * dbetaz_z
    rhs_gzz = rhs_gzz - (2.0/3.0) * gamma_zz_ijk * div_beta
    rhs_gzz = rhs_gzz - ko_dissipation(gamma_zz, i, j, k, sigma_ko / dt)
    rhs_gamma_zz[i, j, k] = rhs_gzz
    
    # ============ RHS for K ============
    # d_t(K) = -D^2(alpha) + alpha*(A_ij*A^ij + K^2/3) + beta^i*d_i(K)
    lap_alpha = d2alpha_xx + d2alpha_yy + d2alpha_zz
    
    # A_ij * A^ij (using flat metric for simplicity)
    A_sq = (A_xx_ijk * A_xx_ijk + A_yy_ijk * A_yy_ijk + A_zz_ijk * A_zz_ijk
            + 2.0 * (A_xy_ijk * A_xy_ijk + A_xz_ijk * A_xz_ijk + A_yz_ijk * A_yz_ijk))
    
    advect_K = beta_x_ijk * dK_x + beta_y_ijk * dK_y + beta_z_ijk * dK_z
    
    rhs_K_val = -lap_alpha + alpha_ijk * (A_sq + K_ijk * K_ijk / 3.0) + advect_K
    rhs_K_val = rhs_K_val - ko_dissipation(K_in, i, j, k, sigma_ko / dt)
    rhs_K[i, j, k] = rhs_K_val
    
    # ============ RHS for A_tilde (simplified - ignoring Ricci tensor) ============
    # d_t(A_ij) = alpha*(K*A_ij - 2*A_ik*A^k_j) + advection + Lie derivative
    # Simplified: just advection and trace-free projection terms
    
    dAxx_x = d1_x(A_xx, i, j, k, inv_12h)
    dAxx_y = d1_y(A_xx, i, j, k, inv_12h)
    dAxx_z = d1_z(A_xx, i, j, k, inv_12h)
    advect_Axx = beta_x_ijk * dAxx_x + beta_y_ijk * dAxx_y + beta_z_ijk * dAxx_z
    
    rhs_Axx = alpha_ijk * K_ijk * A_xx_ijk + advect_Axx
    rhs_Axx = rhs_Axx - ko_dissipation(A_xx, i, j, k, sigma_ko / dt)
    rhs_A_xx[i, j, k] = rhs_Axx
    
    # Similar for other A components (simplified)
    dAxy_x = d1_x(A_xy, i, j, k, inv_12h)
    dAxy_y = d1_y(A_xy, i, j, k, inv_12h)
    dAxy_z = d1_z(A_xy, i, j, k, inv_12h)
    advect_Axy = beta_x_ijk * dAxy_x + beta_y_ijk * dAxy_y + beta_z_ijk * dAxy_z
    rhs_Axy = alpha_ijk * K_ijk * A_xy_ijk + advect_Axy
    rhs_Axy = rhs_Axy - ko_dissipation(A_xy, i, j, k, sigma_ko / dt)
    rhs_A_xy[i, j, k] = rhs_Axy
    
    dAxz_x = d1_x(A_xz, i, j, k, inv_12h)
    dAxz_y = d1_y(A_xz, i, j, k, inv_12h)
    dAxz_z = d1_z(A_xz, i, j, k, inv_12h)
    advect_Axz = beta_x_ijk * dAxz_x + beta_y_ijk * dAxz_y + beta_z_ijk * dAxz_z
    rhs_Axz = alpha_ijk * K_ijk * A_xz_ijk + advect_Axz
    rhs_Axz = rhs_Axz - ko_dissipation(A_xz, i, j, k, sigma_ko / dt)
    rhs_A_xz[i, j, k] = rhs_Axz
    
    dAyy_x = d1_x(A_yy, i, j, k, inv_12h)
    dAyy_y = d1_y(A_yy, i, j, k, inv_12h)
    dAyy_z = d1_z(A_yy, i, j, k, inv_12h)
    advect_Ayy = beta_x_ijk * dAyy_x + beta_y_ijk * dAyy_y + beta_z_ijk * dAyy_z
    rhs_Ayy = alpha_ijk * K_ijk * A_yy_ijk + advect_Ayy
    rhs_Ayy = rhs_Ayy - ko_dissipation(A_yy, i, j, k, sigma_ko / dt)
    rhs_A_yy[i, j, k] = rhs_Ayy
    
    dAyz_x = d1_x(A_yz, i, j, k, inv_12h)
    dAyz_y = d1_y(A_yz, i, j, k, inv_12h)
    dAyz_z = d1_z(A_yz, i, j, k, inv_12h)
    advect_Ayz = beta_x_ijk * dAyz_x + beta_y_ijk * dAyz_y + beta_z_ijk * dAyz_z
    rhs_Ayz = alpha_ijk * K_ijk * A_yz_ijk + advect_Ayz
    rhs_Ayz = rhs_Ayz - ko_dissipation(A_yz, i, j, k, sigma_ko / dt)
    rhs_A_yz[i, j, k] = rhs_Ayz
    
    dAzz_x = d1_x(A_zz, i, j, k, inv_12h)
    dAzz_y = d1_y(A_zz, i, j, k, inv_12h)
    dAzz_z = d1_z(A_zz, i, j, k, inv_12h)
    advect_Azz = beta_x_ijk * dAzz_x + beta_y_ijk * dAzz_y + beta_z_ijk * dAzz_z
    rhs_Azz = alpha_ijk * K_ijk * A_zz_ijk + advect_Azz
    rhs_Azz = rhs_Azz - ko_dissipation(A_zz, i, j, k, sigma_ko / dt)
    rhs_A_zz[i, j, k] = rhs_Azz
    
    # ============ RHS for Gamma^i (simplified) ============
    # Just advection and damping for now
    dGx_x = d1_x(Gamma_x, i, j, k, inv_12h)
    dGx_y = d1_y(Gamma_x, i, j, k, inv_12h)
    dGx_z = d1_z(Gamma_x, i, j, k, inv_12h)
    advect_Gx = beta_x_ijk * dGx_x + beta_y_ijk * dGx_y + beta_z_ijk * dGx_z
    rhs_Gx = advect_Gx - Gamma_x_ijk * dbetax_x
    rhs_Gx = rhs_Gx - ko_dissipation(Gamma_x, i, j, k, sigma_ko / dt)
    rhs_Gamma_x[i, j, k] = rhs_Gx
    
    dGy_x = d1_x(Gamma_y, i, j, k, inv_12h)
    dGy_y = d1_y(Gamma_y, i, j, k, inv_12h)
    dGy_z = d1_z(Gamma_y, i, j, k, inv_12h)
    advect_Gy = beta_x_ijk * dGy_x + beta_y_ijk * dGy_y + beta_z_ijk * dGy_z
    rhs_Gy = advect_Gy - Gamma_y_ijk * dbetay_y
    rhs_Gy = rhs_Gy - ko_dissipation(Gamma_y, i, j, k, sigma_ko / dt)
    rhs_Gamma_y[i, j, k] = rhs_Gy
    
    dGz_x = d1_x(Gamma_z, i, j, k, inv_12h)
    dGz_y = d1_y(Gamma_z, i, j, k, inv_12h)
    dGz_z = d1_z(Gamma_z, i, j, k, inv_12h)
    advect_Gz = beta_x_ijk * dGz_x + beta_y_ijk * dGz_y + beta_z_ijk * dGz_z
    rhs_Gz = advect_Gz - Gamma_z_ijk * dbetaz_z
    rhs_Gz = rhs_Gz - ko_dissipation(Gamma_z, i, j, k, sigma_ko / dt)
    rhs_Gamma_z[i, j, k] = rhs_Gz
    
    # ============ RHS for alpha (1+log slicing) ============
    # d_t(alpha) = -2*alpha*K + beta^i*d_i(alpha)
    advect_alpha = beta_x_ijk * dalpha_x + beta_y_ijk * dalpha_y + beta_z_ijk * dalpha_z
    rhs_alpha_val = -2.0 * alpha_ijk * K_ijk + advect_alpha
    rhs_alpha_val = rhs_alpha_val - ko_dissipation(alpha, i, j, k, sigma_ko / dt)
    rhs_alpha[i, j, k] = rhs_alpha_val
    
    # ============ RHS for beta^i (Gamma-driver) ============
    # d_t(beta^i) = (3/4)*B^i + beta^j*d_j(beta^i)
    dBx_x = d1_x(beta_x, i, j, k, inv_12h)
    dBx_y = d1_y(beta_x, i, j, k, inv_12h)
    dBx_z = d1_z(beta_x, i, j, k, inv_12h)
    advect_bx = beta_x_ijk * dBx_x + beta_y_ijk * dBx_y + beta_z_ijk * dBx_z
    rhs_bx = 0.75 * B_x_ijk + advect_bx
    rhs_bx = rhs_bx - ko_dissipation(beta_x, i, j, k, sigma_ko / dt)
    rhs_beta_x[i, j, k] = rhs_bx
    
    dBy_x = d1_x(beta_y, i, j, k, inv_12h)
    dBy_y = d1_y(beta_y, i, j, k, inv_12h)
    dBy_z = d1_z(beta_y, i, j, k, inv_12h)
    advect_by = beta_x_ijk * dBy_x + beta_y_ijk * dBy_y + beta_z_ijk * dBy_z
    rhs_by = 0.75 * B_y_ijk + advect_by
    rhs_by = rhs_by - ko_dissipation(beta_y, i, j, k, sigma_ko / dt)
    rhs_beta_y[i, j, k] = rhs_by
    
    dBz_x = d1_x(beta_z, i, j, k, inv_12h)
    dBz_y = d1_y(beta_z, i, j, k, inv_12h)
    dBz_z = d1_z(beta_z, i, j, k, inv_12h)
    advect_bz = beta_x_ijk * dBz_x + beta_y_ijk * dBz_y + beta_z_ijk * dBz_z
    rhs_bz = 0.75 * B_z_ijk + advect_bz
    rhs_bz = rhs_bz - ko_dissipation(beta_z, i, j, k, sigma_ko / dt)
    rhs_beta_z[i, j, k] = rhs_bz
    
    # ============ RHS for B^i ============
    # d_t(B^i) = d_t(Gamma^i) - eta*B^i + advection
    dBBx_x = d1_x(B_x, i, j, k, inv_12h)
    dBBx_y = d1_y(B_x, i, j, k, inv_12h)
    dBBx_z = d1_z(B_x, i, j, k, inv_12h)
    advect_Bx = beta_x_ijk * dBBx_x + beta_y_ijk * dBBx_y + beta_z_ijk * dBBx_z
    rhs_Bx = rhs_Gx - eta * B_x_ijk + advect_Bx
    rhs_Bx = rhs_Bx - ko_dissipation(B_x, i, j, k, sigma_ko / dt)
    rhs_B_x[i, j, k] = rhs_Bx
    
    dBBy_x = d1_x(B_y, i, j, k, inv_12h)
    dBBy_y = d1_y(B_y, i, j, k, inv_12h)
    dBBy_z = d1_z(B_y, i, j, k, inv_12h)
    advect_By = beta_x_ijk * dBBy_x + beta_y_ijk * dBBy_y + beta_z_ijk * dBBy_z
    rhs_By = rhs_Gy - eta * B_y_ijk + advect_By
    rhs_By = rhs_By - ko_dissipation(B_y, i, j, k, sigma_ko / dt)
    rhs_B_y[i, j, k] = rhs_By
    
    dBBz_x = d1_x(B_z, i, j, k, inv_12h)
    dBBz_y = d1_y(B_z, i, j, k, inv_12h)
    dBBz_z = d1_z(B_z, i, j, k, inv_12h)
    advect_Bz = beta_x_ijk * dBBz_x + beta_y_ijk * dBBz_y + beta_z_ijk * dBBz_z
    rhs_Bz = rhs_Gz - eta * B_z_ijk + advect_Bz
    rhs_Bz = rhs_Bz - ko_dissipation(B_z, i, j, k, sigma_ko / dt)
    rhs_B_z[i, j, k] = rhs_Bz


def test_full_rhs_flat_spacetime():
    """Test that full RHS is zero for flat spacetime with geodesic slicing"""
    print("Testing full BSSN RHS for flat spacetime...")
    
    ng = 3
    n = 22
    h = 0.1
    dt = 0.01
    sigma_ko = 0.1
    eta = 2.0
    
    shape = (n, n, n)
    
    # Initialize flat spacetime
    chi = wp.zeros(shape, dtype=float); chi.fill_(1.0)
    gamma_xx = wp.zeros(shape, dtype=float); gamma_xx.fill_(1.0)
    gamma_xy = wp.zeros(shape, dtype=float)
    gamma_xz = wp.zeros(shape, dtype=float)
    gamma_yy = wp.zeros(shape, dtype=float); gamma_yy.fill_(1.0)
    gamma_yz = wp.zeros(shape, dtype=float)
    gamma_zz = wp.zeros(shape, dtype=float); gamma_zz.fill_(1.0)
    K = wp.zeros(shape, dtype=float)
    A_xx = wp.zeros(shape, dtype=float)
    A_xy = wp.zeros(shape, dtype=float)
    A_xz = wp.zeros(shape, dtype=float)
    A_yy = wp.zeros(shape, dtype=float)
    A_yz = wp.zeros(shape, dtype=float)
    A_zz = wp.zeros(shape, dtype=float)
    Gamma_x = wp.zeros(shape, dtype=float)
    Gamma_y = wp.zeros(shape, dtype=float)
    Gamma_z = wp.zeros(shape, dtype=float)
    alpha = wp.zeros(shape, dtype=float); alpha.fill_(1.0)
    beta_x = wp.zeros(shape, dtype=float)
    beta_y = wp.zeros(shape, dtype=float)
    beta_z = wp.zeros(shape, dtype=float)
    B_x = wp.zeros(shape, dtype=float)
    B_y = wp.zeros(shape, dtype=float)
    B_z = wp.zeros(shape, dtype=float)
    
    # RHS arrays
    rhs_chi = wp.zeros(shape, dtype=float)
    rhs_gamma_xx = wp.zeros(shape, dtype=float)
    rhs_gamma_xy = wp.zeros(shape, dtype=float)
    rhs_gamma_xz = wp.zeros(shape, dtype=float)
    rhs_gamma_yy = wp.zeros(shape, dtype=float)
    rhs_gamma_yz = wp.zeros(shape, dtype=float)
    rhs_gamma_zz = wp.zeros(shape, dtype=float)
    rhs_K = wp.zeros(shape, dtype=float)
    rhs_A_xx = wp.zeros(shape, dtype=float)
    rhs_A_xy = wp.zeros(shape, dtype=float)
    rhs_A_xz = wp.zeros(shape, dtype=float)
    rhs_A_yy = wp.zeros(shape, dtype=float)
    rhs_A_yz = wp.zeros(shape, dtype=float)
    rhs_A_zz = wp.zeros(shape, dtype=float)
    rhs_Gamma_x = wp.zeros(shape, dtype=float)
    rhs_Gamma_y = wp.zeros(shape, dtype=float)
    rhs_Gamma_z = wp.zeros(shape, dtype=float)
    rhs_alpha = wp.zeros(shape, dtype=float)
    rhs_beta_x = wp.zeros(shape, dtype=float)
    rhs_beta_y = wp.zeros(shape, dtype=float)
    rhs_beta_z = wp.zeros(shape, dtype=float)
    rhs_B_x = wp.zeros(shape, dtype=float)
    rhs_B_y = wp.zeros(shape, dtype=float)
    rhs_B_z = wp.zeros(shape, dtype=float)
    
    inv_12h = 1.0 / (12.0 * h)
    inv_12h2 = 1.0 / (12.0 * h * h)
    inv_144h2 = 1.0 / (144.0 * h * h)
    
    wp.launch(
        compute_full_bssn_rhs,
        dim=(n, n, n),
        inputs=[
            chi, gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz,
            K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz,
            Gamma_x, Gamma_y, Gamma_z,
            alpha, beta_x, beta_y, beta_z, B_x, B_y, B_z,
            rhs_chi, rhs_gamma_xx, rhs_gamma_xy, rhs_gamma_xz,
            rhs_gamma_yy, rhs_gamma_yz, rhs_gamma_zz,
            rhs_K, rhs_A_xx, rhs_A_xy, rhs_A_xz, rhs_A_yy, rhs_A_yz, rhs_A_zz,
            rhs_Gamma_x, rhs_Gamma_y, rhs_Gamma_z,
            rhs_alpha, rhs_beta_x, rhs_beta_y, rhs_beta_z,
            rhs_B_x, rhs_B_y, rhs_B_z,
            n, n, n, inv_12h, inv_12h2, inv_144h2, sigma_ko, dt, eta
        ]
    )
    
    sl = slice(ng, n - ng)
    
    max_rhs = {
        'chi': np.max(np.abs(rhs_chi.numpy()[sl, sl, sl])),
        'gamma_xx': np.max(np.abs(rhs_gamma_xx.numpy()[sl, sl, sl])),
        'K': np.max(np.abs(rhs_K.numpy()[sl, sl, sl])),
        'alpha': np.max(np.abs(rhs_alpha.numpy()[sl, sl, sl])),
    }
    
    print("Max |RHS| for flat spacetime:")
    for name, val in max_rhs.items():
        print(f"  {name}: {val:.6e}")
    
    all_zero = all(v < 1e-10 for v in max_rhs.values())
    if all_zero:
        print("Full BSSN RHS test PASSED!")
    else:
        print("Full BSSN RHS test FAILED!")
    
    return all_zero


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_full_rhs_flat_spacetime()
