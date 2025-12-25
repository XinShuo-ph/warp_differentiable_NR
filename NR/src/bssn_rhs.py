"""
BSSN RHS Implementation

Implements the right-hand side of BSSN evolution equations.
Following the equations from refs/bssn_equations.md
"""

import warp as wp
from finite_diff import (
    deriv_4th_x, deriv_4th_y, deriv_4th_z,
    deriv2_4th_x, deriv2_4th_y, deriv2_4th_z,
    deriv2_4th_xy, deriv2_4th_xz, deriv2_4th_yz,
    dissipation_4th
)

# Import variable indices
from bssn_warp import (
    PHI, GT11, GT12, GT13, GT22, GT23, GT33,
    AT11, AT12, AT13, AT22, AT23, AT33,
    TRK, XT1, XT2, XT3, ALPHA, BETA1, BETA2, BETA3,
    NUM_VARS
)


@wp.kernel
def compute_bssn_rhs_kernel(
    vars: wp.array4d(dtype=wp.float32),
    rhs: wp.array4d(dtype=wp.float32),
    dx: float,
    dy: float,
    dz: float,
    epsDiss: float,
):
    """
    Compute RHS of BSSN evolution equations
    
    Implements simplified BSSN (no shift, simple lapse)
    """
    i, j, k = wp.tid()
    
    nx = vars.shape[0]
    ny = vars.shape[1]
    nz = vars.shape[2]
    
    # Skip boundary points (need at least 2 ghost zones for 4th order)
    if i < 2 or i >= nx - 2:
        return
    if j < 2 or j >= ny - 2:
        return
    if k < 2 or k >= nz - 2:
        return
    
    # Extract current values at this point
    phi = vars[i, j, k, PHI]
    
    gt11 = vars[i, j, k, GT11]
    gt12 = vars[i, j, k, GT12]
    gt13 = vars[i, j, k, GT13]
    gt22 = vars[i, j, k, GT22]
    gt23 = vars[i, j, k, GT23]
    gt33 = vars[i, j, k, GT33]
    
    At11 = vars[i, j, k, AT11]
    At12 = vars[i, j, k, AT12]
    At13 = vars[i, j, k, AT13]
    At22 = vars[i, j, k, AT22]
    At23 = vars[i, j, k, AT23]
    At33 = vars[i, j, k, AT33]
    
    trK = vars[i, j, k, TRK]
    
    Xt1 = vars[i, j, k, XT1]
    Xt2 = vars[i, j, k, XT2]
    Xt3 = vars[i, j, k, XT3]
    
    alpha = vars[i, j, k, ALPHA]
    
    beta1 = vars[i, j, k, BETA1]
    beta2 = vars[i, j, k, BETA2]
    beta3 = vars[i, j, k, BETA3]
    
    # Compute inverse metric g^ij
    detgt = (gt11 * gt22 * gt33 + 2.0 * gt12 * gt13 * gt23 - 
             gt11 * gt23 * gt23 - gt22 * gt13 * gt13 - gt33 * gt12 * gt12)
    
    invdetgt = 1.0 / detgt
    
    gtu11 = (gt22 * gt33 - gt23 * gt23) * invdetgt
    gtu12 = (gt13 * gt23 - gt12 * gt33) * invdetgt
    gtu13 = (gt12 * gt23 - gt13 * gt22) * invdetgt
    gtu22 = (gt11 * gt33 - gt13 * gt13) * invdetgt
    gtu23 = (gt12 * gt13 - gt11 * gt23) * invdetgt
    gtu33 = (gt11 * gt22 - gt12 * gt12) * invdetgt
    
    # Compute e^{4phi}
    e4phi = wp.exp(4.0 * phi)
    
    # Compute derivatives of lapse
    dalpha_dx = deriv_4th_x(vars, i, j, k, ALPHA, dx)
    dalpha_dy = deriv_4th_y(vars, i, j, k, ALPHA, dy)
    dalpha_dz = deriv_4th_z(vars, i, j, k, ALPHA, dz)
    
    # Compute derivatives of trK
    dtrK_dx = deriv_4th_x(vars, i, j, k, TRK, dx)
    dtrK_dy = deriv_4th_y(vars, i, j, k, TRK, dy)
    dtrK_dz = deriv_4th_z(vars, i, j, k, TRK, dz)
    
    # Compute derivatives of phi
    dphi_dx = deriv_4th_x(vars, i, j, k, PHI, dx)
    dphi_dy = deriv_4th_y(vars, i, j, k, PHI, dy)
    dphi_dz = deriv_4th_z(vars, i, j, k, PHI, dz)
    
    # Compute derivatives of shift (for simplicity, assume zero shift initially)
    dbeta1_dx = 0.0  # deriv_4th_x(vars, i, j, k, BETA1, dx)
    dbeta2_dy = 0.0  # deriv_4th_y(vars, i, j, k, BETA2, dy)
    dbeta3_dz = 0.0  # deriv_4th_z(vars, i, j, k, BETA3, dz)
    div_beta = dbeta1_dx + dbeta2_dy + dbeta3_dz
    
    # ========================================================================
    # Evolution equation for phi
    # dot[phi] = 1/3 phi * (alpha * trK - div_beta)
    # ========================================================================
    rhs[i, j, k, PHI] = (1.0/3.0) * phi * (alpha * trK - div_beta) + dissipation_4th(vars, i, j, k, PHI, dx, dy, dz, epsDiss)
    
    # ========================================================================
    # Evolution equations for conformal metric gt_ij
    # dot[gt_ij] = -2 alpha At_ij + (Lie derivative terms)
    # ========================================================================
    rhs[i, j, k, GT11] = -2.0 * alpha * At11 + dissipation_4th(vars, i, j, k, GT11, dx, dy, dz, epsDiss)
    rhs[i, j, k, GT12] = -2.0 * alpha * At12 + dissipation_4th(vars, i, j, k, GT12, dx, dy, dz, epsDiss)
    rhs[i, j, k, GT13] = -2.0 * alpha * At13 + dissipation_4th(vars, i, j, k, GT13, dx, dy, dz, epsDiss)
    rhs[i, j, k, GT22] = -2.0 * alpha * At22 + dissipation_4th(vars, i, j, k, GT22, dx, dy, dz, epsDiss)
    rhs[i, j, k, GT23] = -2.0 * alpha * At23 + dissipation_4th(vars, i, j, k, GT23, dx, dy, dz, epsDiss)
    rhs[i, j, k, GT33] = -2.0 * alpha * At33 + dissipation_4th(vars, i, j, k, GT33, dx, dy, dz, epsDiss)
    
    # ========================================================================
    # Evolution equation for trK
    # dot[trK] = -e^{-4phi} g^ij D_ij alpha + alpha (At^ij At_ij + 1/3 trK^2)
    # ========================================================================
    
    # Compute Laplacian of alpha
    d2alpha_dx2 = deriv2_4th_x(vars, i, j, k, ALPHA, dx)
    d2alpha_dy2 = deriv2_4th_y(vars, i, j, k, ALPHA, dy)
    d2alpha_dz2 = deriv2_4th_z(vars, i, j, k, ALPHA, dz)
    
    laplacian_alpha = gtu11 * d2alpha_dx2 + gtu22 * d2alpha_dy2 + gtu33 * d2alpha_dz2
    
    # Term: 2 dphi^i dalpha_i
    dphi_dot_dalpha = dphi_dx * dalpha_dx + dphi_dy * dalpha_dy + dphi_dz * dalpha_dz
    
    # Compute At^ij At_ij (raising indices)
    At_sq = (At11 * At11 * gtu11 * gtu11 + 
             At22 * At22 * gtu22 * gtu22 +
             At33 * At33 * gtu33 * gtu33 +
             2.0 * At12 * At12 * gtu11 * gtu22 +
             2.0 * At13 * At13 * gtu11 * gtu33 +
             2.0 * At23 * At23 * gtu22 * gtu33)
    
    em4phi = wp.exp(-4.0 * phi)
    
    rhs[i, j, k, TRK] = (-em4phi * (laplacian_alpha + 2.0 * dphi_dot_dalpha) + 
                         alpha * (At_sq + (1.0/3.0) * trK * trK) +
                         dissipation_4th(vars, i, j, k, TRK, dx, dy, dz, epsDiss))
    
    # ========================================================================
    # Evolution equations for At_ij (simplified - no Ricci tensor yet)
    # For flat spacetime test, keep At_ij = 0
    # ========================================================================
    rhs[i, j, k, AT11] = (alpha * trK * At11 + dissipation_4th(vars, i, j, k, AT11, dx, dy, dz, epsDiss))
    rhs[i, j, k, AT12] = (alpha * trK * At12 + dissipation_4th(vars, i, j, k, AT12, dx, dy, dz, epsDiss))
    rhs[i, j, k, AT13] = (alpha * trK * At13 + dissipation_4th(vars, i, j, k, AT13, dx, dy, dz, epsDiss))
    rhs[i, j, k, AT22] = (alpha * trK * At22 + dissipation_4th(vars, i, j, k, AT22, dx, dy, dz, epsDiss))
    rhs[i, j, k, AT23] = (alpha * trK * At23 + dissipation_4th(vars, i, j, k, AT23, dx, dy, dz, epsDiss))
    rhs[i, j, k, AT33] = (alpha * trK * At33 + dissipation_4th(vars, i, j, k, AT33, dx, dy, dz, epsDiss))
    
    # ========================================================================
    # Evolution equations for Xt^i (simplified)
    # ========================================================================
    rhs[i, j, k, XT1] = dissipation_4th(vars, i, j, k, XT1, dx, dy, dz, epsDiss)
    rhs[i, j, k, XT2] = dissipation_4th(vars, i, j, k, XT2, dx, dy, dz, epsDiss)
    rhs[i, j, k, XT3] = dissipation_4th(vars, i, j, k, XT3, dx, dy, dz, epsDiss)
    
    # ========================================================================
    # Evolution equation for lapse (1+log slicing)
    # dot[alpha] = -2 alpha K
    # ========================================================================
    rhs[i, j, k, ALPHA] = -2.0 * alpha * trK + dissipation_4th(vars, i, j, k, ALPHA, dx, dy, dz, epsDiss)
    
    # ========================================================================
    # Shift remains zero (frozen)
    # ========================================================================
    rhs[i, j, k, BETA1] = 0.0
    rhs[i, j, k, BETA2] = 0.0
    rhs[i, j, k, BETA3] = 0.0
