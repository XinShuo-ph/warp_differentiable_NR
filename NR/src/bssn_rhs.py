"""
BSSN RHS Computation

Implements the right-hand sides of the BSSN evolution equations.
Starts with a simplified version for flat spacetime testing.
"""

import warp as wp
from bssn_derivs import (
    idx_3d, 
    deriv_x_4th, deriv_y_4th, deriv_z_4th,
    deriv_xx_4th, deriv_yy_4th, deriv_zz_4th,
    ko_diss_3d
)


# Gauge parameters
HARMONIC_F = 2.0  # For 1+log slicing
ETA_SHIFT = 2.0   # Damping for Gamma-driver shift


@wp.func
def compute_inverse_metric(gt11: float, gt12: float, gt13: float,
                            gt22: float, gt23: float, gt33: float
                            ) -> wp.vec3:
    """
    Compute inverse conformal metric (upper indices).
    Returns gtu11, gtu22, gtu33 (diagonal only for now, assuming near-diagonal).
    Full inverse needs all 6 components.
    """
    # Determinant
    det = (gt11 * (gt22 * gt33 - gt23 * gt23)
           - gt12 * (gt12 * gt33 - gt23 * gt13)
           + gt13 * (gt12 * gt23 - gt22 * gt13))
    
    inv_det = 1.0 / det
    
    # Inverse matrix elements (full symmetric inverse)
    gtu11 = (gt22 * gt33 - gt23 * gt23) * inv_det
    gtu22 = (gt11 * gt33 - gt13 * gt13) * inv_det
    gtu33 = (gt11 * gt22 - gt12 * gt12) * inv_det
    
    return wp.vec3(gtu11, gtu22, gtu33)


@wp.func
def compute_full_inverse_metric(gt11: float, gt12: float, gt13: float,
                                  gt22: float, gt23: float, gt33: float):
    """
    Compute full inverse conformal metric.
    Returns tuple (gtu11, gtu12, gtu13, gtu22, gtu23, gtu33)
    """
    # Determinant
    det = (gt11 * (gt22 * gt33 - gt23 * gt23)
           - gt12 * (gt12 * gt33 - gt23 * gt13)
           + gt13 * (gt12 * gt23 - gt22 * gt13))
    
    inv_det = 1.0 / det
    
    gtu11 = (gt22 * gt33 - gt23 * gt23) * inv_det
    gtu12 = (gt13 * gt23 - gt12 * gt33) * inv_det
    gtu13 = (gt12 * gt23 - gt13 * gt22) * inv_det
    gtu22 = (gt11 * gt33 - gt13 * gt13) * inv_det
    gtu23 = (gt12 * gt13 - gt11 * gt23) * inv_det
    gtu33 = (gt11 * gt22 - gt12 * gt12) * inv_det
    
    return gtu11, gtu12, gtu13, gtu22, gtu23, gtu33


@wp.kernel
def compute_bssn_rhs_kernel(
    # Evolved variables
    phi: wp.array(dtype=wp.float32),
    gt11: wp.array(dtype=wp.float32),
    gt12: wp.array(dtype=wp.float32),
    gt13: wp.array(dtype=wp.float32),
    gt22: wp.array(dtype=wp.float32),
    gt23: wp.array(dtype=wp.float32),
    gt33: wp.array(dtype=wp.float32),
    trK: wp.array(dtype=wp.float32),
    At11: wp.array(dtype=wp.float32),
    At12: wp.array(dtype=wp.float32),
    At13: wp.array(dtype=wp.float32),
    At22: wp.array(dtype=wp.float32),
    At23: wp.array(dtype=wp.float32),
    At33: wp.array(dtype=wp.float32),
    Xt1: wp.array(dtype=wp.float32),
    Xt2: wp.array(dtype=wp.float32),
    Xt3: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    beta1: wp.array(dtype=wp.float32),
    beta2: wp.array(dtype=wp.float32),
    beta3: wp.array(dtype=wp.float32),
    # RHS output
    phi_rhs: wp.array(dtype=wp.float32),
    gt11_rhs: wp.array(dtype=wp.float32),
    gt12_rhs: wp.array(dtype=wp.float32),
    gt13_rhs: wp.array(dtype=wp.float32),
    gt22_rhs: wp.array(dtype=wp.float32),
    gt23_rhs: wp.array(dtype=wp.float32),
    gt33_rhs: wp.array(dtype=wp.float32),
    trK_rhs: wp.array(dtype=wp.float32),
    At11_rhs: wp.array(dtype=wp.float32),
    At12_rhs: wp.array(dtype=wp.float32),
    At13_rhs: wp.array(dtype=wp.float32),
    At22_rhs: wp.array(dtype=wp.float32),
    At23_rhs: wp.array(dtype=wp.float32),
    At33_rhs: wp.array(dtype=wp.float32),
    Xt1_rhs: wp.array(dtype=wp.float32),
    Xt2_rhs: wp.array(dtype=wp.float32),
    Xt3_rhs: wp.array(dtype=wp.float32),
    alpha_rhs: wp.array(dtype=wp.float32),
    beta1_rhs: wp.array(dtype=wp.float32),
    beta2_rhs: wp.array(dtype=wp.float32),
    beta3_rhs: wp.array(dtype=wp.float32),
    # Grid parameters
    nx: int, ny: int, nz: int,
    inv_dx: float,
    eps_diss: float  # Kreiss-Oliger dissipation coefficient (eps * dx)
):
    """Compute BSSN RHS at interior points."""
    tid = wp.tid()
    
    # Convert to 3D indices
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Skip boundary points (need 3 ghost zones for 4th order + dissipation)
    if i < 3 or i >= nx - 3 or j < 3 or j >= ny - 3 or k < 3 or k >= nz - 3:
        phi_rhs[tid] = 0.0
        gt11_rhs[tid] = 0.0
        gt12_rhs[tid] = 0.0
        gt13_rhs[tid] = 0.0
        gt22_rhs[tid] = 0.0
        gt23_rhs[tid] = 0.0
        gt33_rhs[tid] = 0.0
        trK_rhs[tid] = 0.0
        At11_rhs[tid] = 0.0
        At12_rhs[tid] = 0.0
        At13_rhs[tid] = 0.0
        At22_rhs[tid] = 0.0
        At23_rhs[tid] = 0.0
        At33_rhs[tid] = 0.0
        Xt1_rhs[tid] = 0.0
        Xt2_rhs[tid] = 0.0
        Xt3_rhs[tid] = 0.0
        alpha_rhs[tid] = 0.0
        beta1_rhs[tid] = 0.0
        beta2_rhs[tid] = 0.0
        beta3_rhs[tid] = 0.0
        return
    
    # Load local values
    phi_L = phi[tid]
    gt11_L = gt11[tid]
    gt12_L = gt12[tid]
    gt13_L = gt13[tid]
    gt22_L = gt22[tid]
    gt23_L = gt23[tid]
    gt33_L = gt33[tid]
    trK_L = trK[tid]
    At11_L = At11[tid]
    At12_L = At12[tid]
    At13_L = At13[tid]
    At22_L = At22[tid]
    At23_L = At23[tid]
    At33_L = At33[tid]
    Xt1_L = Xt1[tid]
    Xt2_L = Xt2[tid]
    Xt3_L = Xt3[tid]
    alpha_L = alpha[tid]
    beta1_L = beta1[tid]
    beta2_L = beta2[tid]
    beta3_L = beta3[tid]
    
    # Compute derivatives of shift
    dbeta1_dx = deriv_x_4th(beta1, i, j, k, nx, ny, inv_dx)
    dbeta1_dy = deriv_y_4th(beta1, i, j, k, nx, ny, inv_dx)
    dbeta1_dz = deriv_z_4th(beta1, i, j, k, nx, ny, inv_dx)
    dbeta2_dx = deriv_x_4th(beta2, i, j, k, nx, ny, inv_dx)
    dbeta2_dy = deriv_y_4th(beta2, i, j, k, nx, ny, inv_dx)
    dbeta2_dz = deriv_z_4th(beta2, i, j, k, nx, ny, inv_dx)
    dbeta3_dx = deriv_x_4th(beta3, i, j, k, nx, ny, inv_dx)
    dbeta3_dy = deriv_y_4th(beta3, i, j, k, nx, ny, inv_dx)
    dbeta3_dz = deriv_z_4th(beta3, i, j, k, nx, ny, inv_dx)
    
    # Divergence of shift
    div_beta = dbeta1_dx + dbeta2_dy + dbeta3_dz
    
    # ===== phi RHS =====
    # Using W = e^{-2φ} formulation
    # ∂ₜφ = (1/3)(αK - ∂ᵢβⁱ) + βⁱ∂ᵢφ
    dphi_dx = deriv_x_4th(phi, i, j, k, nx, ny, inv_dx)
    dphi_dy = deriv_y_4th(phi, i, j, k, nx, ny, inv_dx)
    dphi_dz = deriv_z_4th(phi, i, j, k, nx, ny, inv_dx)
    
    advect_phi = beta1_L * dphi_dx + beta2_L * dphi_dy + beta3_L * dphi_dz
    diss_phi = ko_diss_3d(phi, i, j, k, nx, ny, eps_diss)
    phi_rhs[tid] = (1.0/3.0) * (alpha_L * trK_L - div_beta) + advect_phi + diss_phi
    
    # ===== Conformal metric RHS =====
    # ∂ₜγ̃ᵢⱼ = -2αÃᵢⱼ + γ̃ᵢₖ∂ⱼβᵏ + γ̃ⱼₖ∂ᵢβᵏ - (2/3)γ̃ᵢⱼ∂ₖβᵏ + βᵏ∂ₖγ̃ᵢⱼ
    
    # Advection terms
    dgt11_dx = deriv_x_4th(gt11, i, j, k, nx, ny, inv_dx)
    dgt11_dy = deriv_y_4th(gt11, i, j, k, nx, ny, inv_dx)
    dgt11_dz = deriv_z_4th(gt11, i, j, k, nx, ny, inv_dx)
    advect_gt11 = beta1_L * dgt11_dx + beta2_L * dgt11_dy + beta3_L * dgt11_dz
    
    dgt12_dx = deriv_x_4th(gt12, i, j, k, nx, ny, inv_dx)
    dgt12_dy = deriv_y_4th(gt12, i, j, k, nx, ny, inv_dx)
    dgt12_dz = deriv_z_4th(gt12, i, j, k, nx, ny, inv_dx)
    advect_gt12 = beta1_L * dgt12_dx + beta2_L * dgt12_dy + beta3_L * dgt12_dz
    
    dgt13_dx = deriv_x_4th(gt13, i, j, k, nx, ny, inv_dx)
    dgt13_dy = deriv_y_4th(gt13, i, j, k, nx, ny, inv_dx)
    dgt13_dz = deriv_z_4th(gt13, i, j, k, nx, ny, inv_dx)
    advect_gt13 = beta1_L * dgt13_dx + beta2_L * dgt13_dy + beta3_L * dgt13_dz
    
    dgt22_dx = deriv_x_4th(gt22, i, j, k, nx, ny, inv_dx)
    dgt22_dy = deriv_y_4th(gt22, i, j, k, nx, ny, inv_dx)
    dgt22_dz = deriv_z_4th(gt22, i, j, k, nx, ny, inv_dx)
    advect_gt22 = beta1_L * dgt22_dx + beta2_L * dgt22_dy + beta3_L * dgt22_dz
    
    dgt23_dx = deriv_x_4th(gt23, i, j, k, nx, ny, inv_dx)
    dgt23_dy = deriv_y_4th(gt23, i, j, k, nx, ny, inv_dx)
    dgt23_dz = deriv_z_4th(gt23, i, j, k, nx, ny, inv_dx)
    advect_gt23 = beta1_L * dgt23_dx + beta2_L * dgt23_dy + beta3_L * dgt23_dz
    
    dgt33_dx = deriv_x_4th(gt33, i, j, k, nx, ny, inv_dx)
    dgt33_dy = deriv_y_4th(gt33, i, j, k, nx, ny, inv_dx)
    dgt33_dz = deriv_z_4th(gt33, i, j, k, nx, ny, inv_dx)
    advect_gt33 = beta1_L * dgt33_dx + beta2_L * dgt33_dy + beta3_L * dgt33_dz
    
    # Shift derivative terms: γ̃ᵢₖ∂ⱼβᵏ + γ̃ⱼₖ∂ᵢβᵏ
    # For gt11: γ̃₁ₖ∂₁βᵏ + γ̃₁ₖ∂₁βᵏ = 2(γ̃₁₁∂₁β¹ + γ̃₁₂∂₁β² + γ̃₁₃∂₁β³)
    shift_gt11 = 2.0 * (gt11_L * dbeta1_dx + gt12_L * dbeta2_dx + gt13_L * dbeta3_dx)
    shift_gt12 = (gt11_L * dbeta1_dy + gt12_L * dbeta2_dy + gt13_L * dbeta3_dy +
                  gt12_L * dbeta1_dx + gt22_L * dbeta2_dx + gt23_L * dbeta3_dx)
    shift_gt13 = (gt11_L * dbeta1_dz + gt12_L * dbeta2_dz + gt13_L * dbeta3_dz +
                  gt13_L * dbeta1_dx + gt23_L * dbeta2_dx + gt33_L * dbeta3_dx)
    shift_gt22 = 2.0 * (gt12_L * dbeta1_dy + gt22_L * dbeta2_dy + gt23_L * dbeta3_dy)
    shift_gt23 = (gt12_L * dbeta1_dz + gt22_L * dbeta2_dz + gt23_L * dbeta3_dz +
                  gt13_L * dbeta1_dy + gt23_L * dbeta2_dy + gt33_L * dbeta3_dy)
    shift_gt33 = 2.0 * (gt13_L * dbeta1_dz + gt23_L * dbeta2_dz + gt33_L * dbeta3_dz)
    
    diss_gt11 = ko_diss_3d(gt11, i, j, k, nx, ny, eps_diss)
    diss_gt12 = ko_diss_3d(gt12, i, j, k, nx, ny, eps_diss)
    diss_gt13 = ko_diss_3d(gt13, i, j, k, nx, ny, eps_diss)
    diss_gt22 = ko_diss_3d(gt22, i, j, k, nx, ny, eps_diss)
    diss_gt23 = ko_diss_3d(gt23, i, j, k, nx, ny, eps_diss)
    diss_gt33 = ko_diss_3d(gt33, i, j, k, nx, ny, eps_diss)
    
    gt11_rhs[tid] = -2.0 * alpha_L * At11_L + shift_gt11 - (2.0/3.0) * gt11_L * div_beta + advect_gt11 + diss_gt11
    gt12_rhs[tid] = -2.0 * alpha_L * At12_L + shift_gt12 - (2.0/3.0) * gt12_L * div_beta + advect_gt12 + diss_gt12
    gt13_rhs[tid] = -2.0 * alpha_L * At13_L + shift_gt13 - (2.0/3.0) * gt13_L * div_beta + advect_gt13 + diss_gt13
    gt22_rhs[tid] = -2.0 * alpha_L * At22_L + shift_gt22 - (2.0/3.0) * gt22_L * div_beta + advect_gt22 + diss_gt22
    gt23_rhs[tid] = -2.0 * alpha_L * At23_L + shift_gt23 - (2.0/3.0) * gt23_L * div_beta + advect_gt23 + diss_gt23
    gt33_rhs[tid] = -2.0 * alpha_L * At33_L + shift_gt33 - (2.0/3.0) * gt33_L * div_beta + advect_gt33 + diss_gt33
    
    # ===== trK RHS (simplified for flat spacetime) =====
    # Full equation: ∂ₜK = -∇²α + α(AᵢⱼAⁱʲ + K²/3) + βⁱ∂ᵢK
    dtrK_dx = deriv_x_4th(trK, i, j, k, nx, ny, inv_dx)
    dtrK_dy = deriv_y_4th(trK, i, j, k, nx, ny, inv_dx)
    dtrK_dz = deriv_z_4th(trK, i, j, k, nx, ny, inv_dx)
    advect_trK = beta1_L * dtrK_dx + beta2_L * dtrK_dy + beta3_L * dtrK_dz
    
    inv_dx2 = inv_dx * inv_dx
    d2alpha_xx = deriv_xx_4th(alpha, i, j, k, nx, ny, inv_dx2)
    d2alpha_yy = deriv_yy_4th(alpha, i, j, k, nx, ny, inv_dx2)
    d2alpha_zz = deriv_zz_4th(alpha, i, j, k, nx, ny, inv_dx2)
    
    # Compute A_ij A^ij using inverse metric
    gtu11, gtu12, gtu13, gtu22, gtu23, gtu33 = compute_full_inverse_metric(
        gt11_L, gt12_L, gt13_L, gt22_L, gt23_L, gt33_L)
    
    # A^ij = gtu^ik gtu^jl At_kl
    # A_ij A^ij = At_ij gtu^ik gtu^jl At_kl = ...
    # For simplicity: At_ij At^ij where At^ij = gtu^ik gtu^jl At_kl
    At_sq = (At11_L * At11_L * gtu11 * gtu11 + At22_L * At22_L * gtu22 * gtu22 + 
             At33_L * At33_L * gtu33 * gtu33 +
             2.0 * At12_L * At12_L * gtu11 * gtu22 +
             2.0 * At13_L * At13_L * gtu11 * gtu33 +
             2.0 * At23_L * At23_L * gtu22 * gtu33)
    
    # Simplified Laplacian (ignoring conformal factor for now)
    lap_alpha = d2alpha_xx + d2alpha_yy + d2alpha_zz
    
    diss_trK = ko_diss_3d(trK, i, j, k, nx, ny, eps_diss)
    trK_rhs[tid] = -lap_alpha + alpha_L * (At_sq + trK_L * trK_L / 3.0) + advect_trK + diss_trK
    
    # ===== At_ij RHS (simplified - just advection for flat spacetime test) =====
    dAt11_dx = deriv_x_4th(At11, i, j, k, nx, ny, inv_dx)
    dAt11_dy = deriv_y_4th(At11, i, j, k, nx, ny, inv_dx)
    dAt11_dz = deriv_z_4th(At11, i, j, k, nx, ny, inv_dx)
    At11_rhs[tid] = beta1_L * dAt11_dx + beta2_L * dAt11_dy + beta3_L * dAt11_dz
    
    dAt12_dx = deriv_x_4th(At12, i, j, k, nx, ny, inv_dx)
    dAt12_dy = deriv_y_4th(At12, i, j, k, nx, ny, inv_dx)
    dAt12_dz = deriv_z_4th(At12, i, j, k, nx, ny, inv_dx)
    At12_rhs[tid] = beta1_L * dAt12_dx + beta2_L * dAt12_dy + beta3_L * dAt12_dz
    
    dAt13_dx = deriv_x_4th(At13, i, j, k, nx, ny, inv_dx)
    dAt13_dy = deriv_y_4th(At13, i, j, k, nx, ny, inv_dx)
    dAt13_dz = deriv_z_4th(At13, i, j, k, nx, ny, inv_dx)
    At13_rhs[tid] = beta1_L * dAt13_dx + beta2_L * dAt13_dy + beta3_L * dAt13_dz
    
    dAt22_dx = deriv_x_4th(At22, i, j, k, nx, ny, inv_dx)
    dAt22_dy = deriv_y_4th(At22, i, j, k, nx, ny, inv_dx)
    dAt22_dz = deriv_z_4th(At22, i, j, k, nx, ny, inv_dx)
    At22_rhs[tid] = beta1_L * dAt22_dx + beta2_L * dAt22_dy + beta3_L * dAt22_dz
    
    dAt23_dx = deriv_x_4th(At23, i, j, k, nx, ny, inv_dx)
    dAt23_dy = deriv_y_4th(At23, i, j, k, nx, ny, inv_dx)
    dAt23_dz = deriv_z_4th(At23, i, j, k, nx, ny, inv_dx)
    At23_rhs[tid] = beta1_L * dAt23_dx + beta2_L * dAt23_dy + beta3_L * dAt23_dz
    
    dAt33_dx = deriv_x_4th(At33, i, j, k, nx, ny, inv_dx)
    dAt33_dy = deriv_y_4th(At33, i, j, k, nx, ny, inv_dx)
    dAt33_dz = deriv_z_4th(At33, i, j, k, nx, ny, inv_dx)
    At33_rhs[tid] = beta1_L * dAt33_dx + beta2_L * dAt33_dy + beta3_L * dAt33_dz
    
    # ===== Xt RHS (simplified - just advection for flat spacetime) =====
    dXt1_dx = deriv_x_4th(Xt1, i, j, k, nx, ny, inv_dx)
    dXt1_dy = deriv_y_4th(Xt1, i, j, k, nx, ny, inv_dx)
    dXt1_dz = deriv_z_4th(Xt1, i, j, k, nx, ny, inv_dx)
    Xt1_rhs[tid] = beta1_L * dXt1_dx + beta2_L * dXt1_dy + beta3_L * dXt1_dz
    
    dXt2_dx = deriv_x_4th(Xt2, i, j, k, nx, ny, inv_dx)
    dXt2_dy = deriv_y_4th(Xt2, i, j, k, nx, ny, inv_dx)
    dXt2_dz = deriv_z_4th(Xt2, i, j, k, nx, ny, inv_dx)
    Xt2_rhs[tid] = beta1_L * dXt2_dx + beta2_L * dXt2_dy + beta3_L * dXt2_dz
    
    dXt3_dx = deriv_x_4th(Xt3, i, j, k, nx, ny, inv_dx)
    dXt3_dy = deriv_y_4th(Xt3, i, j, k, nx, ny, inv_dx)
    dXt3_dz = deriv_z_4th(Xt3, i, j, k, nx, ny, inv_dx)
    Xt3_rhs[tid] = beta1_L * dXt3_dx + beta2_L * dXt3_dy + beta3_L * dXt3_dz
    
    # ===== alpha RHS (1+log slicing) =====
    # ∂ₜα = -2αK + βⁱ∂ᵢα
    dalpha_dx = deriv_x_4th(alpha, i, j, k, nx, ny, inv_dx)
    dalpha_dy = deriv_y_4th(alpha, i, j, k, nx, ny, inv_dx)
    dalpha_dz = deriv_z_4th(alpha, i, j, k, nx, ny, inv_dx)
    advect_alpha = beta1_L * dalpha_dx + beta2_L * dalpha_dy + beta3_L * dalpha_dz
    
    diss_alpha = ko_diss_3d(alpha, i, j, k, nx, ny, eps_diss)
    alpha_rhs[tid] = -HARMONIC_F * alpha_L * trK_L + advect_alpha + diss_alpha
    
    # ===== beta RHS (Gamma-driver) =====
    # ∂ₜβⁱ = (3/4)Γ̃ⁱ - ηβⁱ + βʲ∂ⱼβⁱ
    dbeta1_advect = beta1_L * dbeta1_dx + beta2_L * dbeta1_dy + beta3_L * dbeta1_dz
    dbeta2_advect = beta1_L * dbeta2_dx + beta2_L * dbeta2_dy + beta3_L * dbeta2_dz
    dbeta3_advect = beta1_L * dbeta3_dx + beta2_L * dbeta3_dy + beta3_L * dbeta3_dz
    
    beta1_rhs[tid] = 0.75 * Xt1_L - ETA_SHIFT * beta1_L + dbeta1_advect
    beta2_rhs[tid] = 0.75 * Xt2_L - ETA_SHIFT * beta2_L + dbeta2_advect
    beta3_rhs[tid] = 0.75 * Xt3_L - ETA_SHIFT * beta3_L + dbeta3_advect


def test_bssn_rhs_flat():
    """Test BSSN RHS computation on flat spacetime (all RHS should be ~zero)."""
    import sys
    sys.path.insert(0, '/workspace/NR/src')
    from bssn_vars import BSSNGrid
    
    wp.init()
    print("=== BSSN RHS Test (Flat Spacetime) ===\n")
    
    nx, ny, nz = 20, 20, 20
    dx = 0.1
    
    grid = BSSNGrid(nx, ny, nz, dx)
    grid.set_flat_spacetime()
    
    inv_dx = 1.0 / dx
    eps_diss = 0.2 * dx  # Typical dissipation coefficient (eps * dx)
    
    # Compute RHS
    wp.launch(
        compute_bssn_rhs_kernel,
        dim=grid.n_points,
        inputs=[
            grid.phi, grid.gt11, grid.gt12, grid.gt13, grid.gt22, grid.gt23, grid.gt33,
            grid.trK, grid.At11, grid.At12, grid.At13, grid.At22, grid.At23, grid.At33,
            grid.Xt1, grid.Xt2, grid.Xt3,
            grid.alpha, grid.beta1, grid.beta2, grid.beta3,
            grid.phi_rhs, grid.gt11_rhs, grid.gt12_rhs, grid.gt13_rhs, 
            grid.gt22_rhs, grid.gt23_rhs, grid.gt33_rhs,
            grid.trK_rhs, grid.At11_rhs, grid.At12_rhs, grid.At13_rhs,
            grid.At22_rhs, grid.At23_rhs, grid.At33_rhs,
            grid.Xt1_rhs, grid.Xt2_rhs, grid.Xt3_rhs,
            grid.alpha_rhs, grid.beta1_rhs, grid.beta2_rhs, grid.beta3_rhs,
            nx, ny, nz, inv_dx, eps_diss
        ]
    )
    
    # Check RHS values (should all be near zero for flat spacetime)
    import numpy as np
    
    print("RHS max absolute values (should be ~0 for flat spacetime):")
    print(f"  phi_rhs:   {np.abs(grid.phi_rhs.numpy()).max():.6e}")
    print(f"  gt11_rhs:  {np.abs(grid.gt11_rhs.numpy()).max():.6e}")
    print(f"  gt22_rhs:  {np.abs(grid.gt22_rhs.numpy()).max():.6e}")
    print(f"  gt33_rhs:  {np.abs(grid.gt33_rhs.numpy()).max():.6e}")
    print(f"  gt12_rhs:  {np.abs(grid.gt12_rhs.numpy()).max():.6e}")
    print(f"  trK_rhs:   {np.abs(grid.trK_rhs.numpy()).max():.6e}")
    print(f"  At11_rhs:  {np.abs(grid.At11_rhs.numpy()).max():.6e}")
    print(f"  Xt1_rhs:   {np.abs(grid.Xt1_rhs.numpy()).max():.6e}")
    print(f"  alpha_rhs: {np.abs(grid.alpha_rhs.numpy()).max():.6e}")
    print(f"  beta1_rhs: {np.abs(grid.beta1_rhs.numpy()).max():.6e}")
    
    # Verify all RHS are near zero
    max_rhs = max(
        np.abs(grid.phi_rhs.numpy()).max(),
        np.abs(grid.gt11_rhs.numpy()).max(),
        np.abs(grid.gt22_rhs.numpy()).max(),
        np.abs(grid.trK_rhs.numpy()).max(),
        np.abs(grid.alpha_rhs.numpy()).max()
    )
    
    if max_rhs < 1e-6:
        print(f"\n✓ All RHS values are near zero (max: {max_rhs:.2e})")
    else:
        print(f"\n⚠ Some RHS values are non-zero (max: {max_rhs:.2e})")
    
    print("\n✓ BSSN RHS computation test completed.")


if __name__ == "__main__":
    test_bssn_rhs_flat()
