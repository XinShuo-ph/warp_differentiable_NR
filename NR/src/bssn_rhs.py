"""
BSSN Right-Hand-Side Evolution Equations

For flat spacetime with geodesic slicing (alpha=1, beta=0):
All RHS terms should be zero (stationary solution).

This module implements the full BSSN RHS which reduces to zero for flat space.
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
def compute_bssn_rhs_kernel(
    # Current state fields
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
    # RHS output fields (simplified subset for now)
    rhs_chi: wp.array3d(dtype=float),
    rhs_gamma_xx: wp.array3d(dtype=float),
    rhs_K: wp.array3d(dtype=float),
    # Grid parameters
    nx: int, ny: int, nz: int,
    inv_12h: float,
    inv_12h2: float,
    sigma_ko: float,
    dt: float
):
    """
    Compute BSSN RHS for chi, gamma_tilde, K (simplified version)
    
    Full BSSN evolution:
    partial_t(chi) = (2/3)*chi*(alpha*K - D_i(beta^i)) + beta^i*D_i(chi)
    partial_t(gamma_tilde_ij) = -2*alpha*A_tilde_ij + ... (advection + Lie derivative terms)
    partial_t(K) = -D^i D_i(alpha) + alpha*(A_ij*A^ij + K^2/3) + ... (matter terms)
    """
    i, j, k = wp.tid()
    
    # Need 3 ghost zones for KO dissipation
    if i < 3 or i >= nx - 3:
        return
    if j < 3 or j >= ny - 3:
        return
    if k < 3 or k >= nz - 3:
        return
    
    # Local values
    chi_ijk = chi[i, j, k]
    alpha_ijk = alpha[i, j, k]
    K_ijk = K_in[i, j, k]
    
    # ============ RHS for chi ============
    # partial_t(chi) = (2/3)*chi*(alpha*K - div(beta)) + beta^i * d_i(chi)
    
    # Derivatives of chi
    dchi_x = d1_x(chi, i, j, k, inv_12h)
    dchi_y = d1_y(chi, i, j, k, inv_12h)
    dchi_z = d1_z(chi, i, j, k, inv_12h)
    
    # Derivatives of beta (for div(beta))
    dbetax_x = d1_x(beta_x, i, j, k, inv_12h)
    dbetay_y = d1_y(beta_y, i, j, k, inv_12h)
    dbetaz_z = d1_z(beta_z, i, j, k, inv_12h)
    div_beta = dbetax_x + dbetay_y + dbetaz_z
    
    # Advection term
    beta_x_ijk = beta_x[i, j, k]
    beta_y_ijk = beta_y[i, j, k]
    beta_z_ijk = beta_z[i, j, k]
    advect_chi = beta_x_ijk * dchi_x + beta_y_ijk * dchi_y + beta_z_ijk * dchi_z
    
    rhs_chi_val = (2.0/3.0) * chi_ijk * (alpha_ijk * K_ijk - div_beta) + advect_chi
    
    # Add KO dissipation
    rhs_chi_val = rhs_chi_val - ko_dissipation(chi, i, j, k, sigma_ko / dt)
    
    rhs_chi[i, j, k] = rhs_chi_val
    
    # ============ RHS for gamma_tilde_xx (example component) ============
    # partial_t(gamma_tilde_ij) = -2*alpha*A_tilde_ij + beta^k*d_k(gamma_ij) + ...
    
    A_xx_ijk = A_xx[i, j, k]
    gamma_xx_ijk = gamma_xx[i, j, k]
    
    # Advection of gamma
    dgammaxx_x = d1_x(gamma_xx, i, j, k, inv_12h)
    dgammaxx_y = d1_y(gamma_xx, i, j, k, inv_12h)
    dgammaxx_z = d1_z(gamma_xx, i, j, k, inv_12h)
    advect_gamma = beta_x_ijk * dgammaxx_x + beta_y_ijk * dgammaxx_y + beta_z_ijk * dgammaxx_z
    
    # Lie derivative terms (partial_j(beta^k) contributions)
    dbetax_x = d1_x(beta_x, i, j, k, inv_12h)
    
    # Full gamma_xx evolution (simplified)
    rhs_gamma_xx_val = (
        -2.0 * alpha_ijk * A_xx_ijk 
        + advect_gamma 
        + 2.0 * gamma_xx_ijk * dbetax_x  # Lie derivative (diagonal term)
        - (2.0/3.0) * gamma_xx_ijk * div_beta
    )
    
    rhs_gamma_xx_val = rhs_gamma_xx_val - ko_dissipation(gamma_xx, i, j, k, sigma_ko / dt)
    rhs_gamma_xx[i, j, k] = rhs_gamma_xx_val
    
    # ============ RHS for K ============
    # partial_t(K) = -D^2(alpha) + alpha*(A_ij*A^ij + K^2/3) + beta^i*d_i(K)
    
    # Laplacian of alpha (flat space approximation: D^2 = d^2)
    d2alpha_xx = d2_xx(alpha, i, j, k, inv_12h2)
    d2alpha_yy = d2_yy(alpha, i, j, k, inv_12h2)
    d2alpha_zz = d2_zz(alpha, i, j, k, inv_12h2)
    lap_alpha = d2alpha_xx + d2alpha_yy + d2alpha_zz
    
    # A_ij * A^ij (using flat metric for raised indices in first approximation)
    A_xx_ijk = A_xx[i, j, k]
    A_xy_ijk = A_xy[i, j, k]
    A_xz_ijk = A_xz[i, j, k]
    A_yy_ijk = A_yy[i, j, k]
    A_yz_ijk = A_yz[i, j, k]
    A_zz_ijk = A_zz[i, j, k]
    
    A_sq = (
        A_xx_ijk * A_xx_ijk + A_yy_ijk * A_yy_ijk + A_zz_ijk * A_zz_ijk
        + 2.0 * (A_xy_ijk * A_xy_ijk + A_xz_ijk * A_xz_ijk + A_yz_ijk * A_yz_ijk)
    )
    
    # Advection
    dK_x = d1_x(K_in, i, j, k, inv_12h)
    dK_y = d1_y(K_in, i, j, k, inv_12h)
    dK_z = d1_z(K_in, i, j, k, inv_12h)
    advect_K = beta_x_ijk * dK_x + beta_y_ijk * dK_y + beta_z_ijk * dK_z
    
    rhs_K_val = (
        -lap_alpha 
        + alpha_ijk * (A_sq + (1.0/3.0) * K_ijk * K_ijk)
        + advect_K
    )
    
    rhs_K_val = rhs_K_val - ko_dissipation(K_in, i, j, k, sigma_ko / dt)
    rhs_K[i, j, k] = rhs_K_val


def test_flat_spacetime_rhs():
    """Test that RHS is zero for flat spacetime"""
    print("Testing BSSN RHS for flat spacetime...")
    
    # Grid parameters
    ng = 3  # ghost zones
    n_interior = 16
    n = n_interior + 2 * ng
    h = 0.1
    dt = 0.01
    sigma_ko = 0.1
    
    # Create arrays and initialize to flat spacetime
    shape = (n, n, n)
    
    chi = wp.zeros(shape, dtype=float)
    chi.fill_(1.0)
    
    gamma_xx = wp.zeros(shape, dtype=float)
    gamma_xx.fill_(1.0)
    gamma_xy = wp.zeros(shape, dtype=float)
    gamma_xz = wp.zeros(shape, dtype=float)
    gamma_yy = wp.zeros(shape, dtype=float)
    gamma_yy.fill_(1.0)
    gamma_yz = wp.zeros(shape, dtype=float)
    gamma_zz = wp.zeros(shape, dtype=float)
    gamma_zz.fill_(1.0)
    
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
    
    alpha = wp.zeros(shape, dtype=float)
    alpha.fill_(1.0)
    
    beta_x = wp.zeros(shape, dtype=float)
    beta_y = wp.zeros(shape, dtype=float)
    beta_z = wp.zeros(shape, dtype=float)
    
    # RHS arrays
    rhs_chi = wp.zeros(shape, dtype=float)
    rhs_gamma_xx = wp.zeros(shape, dtype=float)
    rhs_K = wp.zeros(shape, dtype=float)
    
    inv_12h = 1.0 / (12.0 * h)
    inv_12h2 = 1.0 / (12.0 * h * h)
    
    # Compute RHS
    wp.launch(
        compute_bssn_rhs_kernel,
        dim=(n, n, n),
        inputs=[
            chi, gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz,
            K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz,
            Gamma_x, Gamma_y, Gamma_z,
            alpha, beta_x, beta_y, beta_z,
            rhs_chi, rhs_gamma_xx, rhs_K,
            n, n, n, inv_12h, inv_12h2, sigma_ko, dt
        ]
    )
    
    # Check RHS values (should be zero for flat spacetime)
    sl = slice(ng, n - ng)
    
    rhs_chi_np = rhs_chi.numpy()[sl, sl, sl]
    rhs_gamma_xx_np = rhs_gamma_xx.numpy()[sl, sl, sl]
    rhs_K_np = rhs_K.numpy()[sl, sl, sl]
    
    max_rhs_chi = np.max(np.abs(rhs_chi_np))
    max_rhs_gamma = np.max(np.abs(rhs_gamma_xx_np))
    max_rhs_K = np.max(np.abs(rhs_K_np))
    
    print(f"Max |RHS_chi|: {max_rhs_chi:.6e}")
    print(f"Max |RHS_gamma_xx|: {max_rhs_gamma:.6e}")
    print(f"Max |RHS_K|: {max_rhs_K:.6e}")
    
    # For flat spacetime, all RHS should be zero (up to floating point)
    tol = 1e-10
    if max_rhs_chi < tol and max_rhs_gamma < tol and max_rhs_K < tol:
        print("Flat spacetime RHS test PASSED!")
    else:
        print("Flat spacetime RHS test FAILED - nonzero RHS values")
    
    return max_rhs_chi, max_rhs_gamma, max_rhs_K


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_flat_spacetime_rhs()
