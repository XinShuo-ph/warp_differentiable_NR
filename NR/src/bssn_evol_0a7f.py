"""
Complete BSSN Evolution with RK4 Integration

Implements proper RK4 time integration and more complete BSSN RHS.
"""

import numpy as np
import warp as wp


# ============================================================================
# Helper Functions
# ============================================================================

@wp.func
def clamp_idx(i: int, n: int) -> int:
    """Clamp index to valid range [0, n-1]."""
    return wp.clamp(i, 0, n - 1)


@wp.func
def get_val(f: wp.array3d(dtype=float), i: int, j: int, k: int, 
            nx: int, ny: int, nz: int) -> float:
    """Get value with clamped boundary handling."""
    ii = clamp_idx(i, nx)
    jj = clamp_idx(j, ny)
    kk = clamp_idx(k, nz)
    return f[ii, jj, kk]


@wp.func
def dx_4th(f: wp.array3d(dtype=float), i: int, j: int, k: int,
           h: float, nx: int, ny: int, nz: int) -> float:
    """4th order derivative in x direction."""
    fm2 = get_val(f, i - 2, j, k, nx, ny, nz)
    fm1 = get_val(f, i - 1, j, k, nx, ny, nz)
    fp1 = get_val(f, i + 1, j, k, nx, ny, nz)
    fp2 = get_val(f, i + 2, j, k, nx, ny, nz)
    return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)


@wp.func
def dy_4th(f: wp.array3d(dtype=float), i: int, j: int, k: int,
           h: float, nx: int, ny: int, nz: int) -> float:
    """4th order derivative in y direction."""
    fm2 = get_val(f, i, j - 2, k, nx, ny, nz)
    fm1 = get_val(f, i, j - 1, k, nx, ny, nz)
    fp1 = get_val(f, i, j + 1, k, nx, ny, nz)
    fp2 = get_val(f, i, j + 2, k, nx, ny, nz)
    return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)


@wp.func
def dz_4th(f: wp.array3d(dtype=float), i: int, j: int, k: int,
           h: float, nx: int, ny: int, nz: int) -> float:
    """4th order derivative in z direction."""
    fm2 = get_val(f, i, j, k - 2, nx, ny, nz)
    fm1 = get_val(f, i, j, k - 1, nx, ny, nz)
    fp1 = get_val(f, i, j, k + 1, nx, ny, nz)
    fp2 = get_val(f, i, j, k + 2, nx, ny, nz)
    return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)


@wp.func
def dxx_4th(f: wp.array3d(dtype=float), i: int, j: int, k: int,
            h: float, nx: int, ny: int, nz: int) -> float:
    """4th order second derivative in x."""
    fm2 = get_val(f, i - 2, j, k, nx, ny, nz)
    fm1 = get_val(f, i - 1, j, k, nx, ny, nz)
    f0 = f[i, j, k]
    fp1 = get_val(f, i + 1, j, k, nx, ny, nz)
    fp2 = get_val(f, i + 2, j, k, nx, ny, nz)
    return (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h * h)


@wp.func
def dyy_4th(f: wp.array3d(dtype=float), i: int, j: int, k: int,
            h: float, nx: int, ny: int, nz: int) -> float:
    """4th order second derivative in y."""
    fm2 = get_val(f, i, j - 2, k, nx, ny, nz)
    fm1 = get_val(f, i, j - 1, k, nx, ny, nz)
    f0 = f[i, j, k]
    fp1 = get_val(f, i, j + 1, k, nx, ny, nz)
    fp2 = get_val(f, i, j + 2, k, nx, ny, nz)
    return (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h * h)


@wp.func
def dzz_4th(f: wp.array3d(dtype=float), i: int, j: int, k: int,
            h: float, nx: int, ny: int, nz: int) -> float:
    """4th order second derivative in z."""
    fm2 = get_val(f, i, j, k - 2, nx, ny, nz)
    fm1 = get_val(f, i, j, k - 1, nx, ny, nz)
    f0 = f[i, j, k]
    fp1 = get_val(f, i, j, k + 1, nx, ny, nz)
    fp2 = get_val(f, i, j, k + 2, nx, ny, nz)
    return (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h * h)


@wp.func 
def laplacian_4th(f: wp.array3d(dtype=float), i: int, j: int, k: int,
                  h: float, nx: int, ny: int, nz: int) -> float:
    """4th order Laplacian."""
    return dxx_4th(f, i, j, k, h, nx, ny, nz) + \
           dyy_4th(f, i, j, k, h, nx, ny, nz) + \
           dzz_4th(f, i, j, k, h, nx, ny, nz)


# ============================================================================
# BSSN RHS Kernels - Full equations
# ============================================================================

@wp.kernel
def rhs_phi_full(
    # Input fields
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    betax: wp.array3d(dtype=float),
    betay: wp.array3d(dtype=float),
    betaz: wp.array3d(dtype=float),
    # Output
    rhs: wp.array3d(dtype=float),
    # Grid params
    h: float, nx: int, ny: int, nz: int
):
    """
    ∂ₜφ = -αK/6 + βⁱ∂ᵢφ + ∂ᵢβⁱ/6
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    Kval = K[i, j, k]
    
    # Advection: βⁱ∂ᵢφ
    bx = betax[i, j, k]
    by = betay[i, j, k]
    bz = betaz[i, j, k]
    
    dphi_x = dx_4th(phi, i, j, k, h, nx, ny, nz)
    dphi_y = dy_4th(phi, i, j, k, h, nx, ny, nz)
    dphi_z = dz_4th(phi, i, j, k, h, nx, ny, nz)
    
    advection = bx * dphi_x + by * dphi_y + bz * dphi_z
    
    # Divergence: ∂ᵢβⁱ
    div_beta = dx_4th(betax, i, j, k, h, nx, ny, nz) + \
               dy_4th(betay, i, j, k, h, nx, ny, nz) + \
               dz_4th(betaz, i, j, k, h, nx, ny, nz)
    
    rhs[i, j, k] = -a * Kval / 6.0 + advection + div_beta / 6.0


@wp.kernel
def rhs_gtij_full(
    # Metric components
    gtxx: wp.array3d(dtype=float),
    gtxy: wp.array3d(dtype=float),
    gtxz: wp.array3d(dtype=float),
    gtyy: wp.array3d(dtype=float),
    gtyz: wp.array3d(dtype=float),
    gtzz: wp.array3d(dtype=float),
    # Traceless extrinsic curvature
    Atxx: wp.array3d(dtype=float),
    Atxy: wp.array3d(dtype=float),
    Atxz: wp.array3d(dtype=float),
    Atyy: wp.array3d(dtype=float),
    Atyz: wp.array3d(dtype=float),
    Atzz: wp.array3d(dtype=float),
    # Gauge
    alpha: wp.array3d(dtype=float),
    betax: wp.array3d(dtype=float),
    betay: wp.array3d(dtype=float),
    betaz: wp.array3d(dtype=float),
    # Output RHS (6 components)
    rhs_xx: wp.array3d(dtype=float),
    rhs_xy: wp.array3d(dtype=float),
    rhs_xz: wp.array3d(dtype=float),
    rhs_yy: wp.array3d(dtype=float),
    rhs_yz: wp.array3d(dtype=float),
    rhs_zz: wp.array3d(dtype=float),
    # Grid params
    h: float, nx: int, ny: int, nz: int
):
    """
    ∂ₜγ̃ᵢⱼ = -2αÃᵢⱼ + βᵏ∂ₖγ̃ᵢⱼ + γ̃ᵢₖ∂ⱼβᵏ + γ̃ⱼₖ∂ᵢβᵏ - (2/3)γ̃ᵢⱼ∂ₖβᵏ
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    
    # Load metric and At values
    gxx = gtxx[i, j, k]
    gxy = gtxy[i, j, k]
    gxz = gtxz[i, j, k]
    gyy = gtyy[i, j, k]
    gyz = gtyz[i, j, k]
    gzz = gtzz[i, j, k]
    
    Axx = Atxx[i, j, k]
    Axy = Atxy[i, j, k]
    Axz = Atxz[i, j, k]
    Ayy = Atyy[i, j, k]
    Ayz = Atyz[i, j, k]
    Azz = Atzz[i, j, k]
    
    # Shift derivatives
    bx = betax[i, j, k]
    by = betay[i, j, k]
    bz = betaz[i, j, k]
    
    dbx_x = dx_4th(betax, i, j, k, h, nx, ny, nz)
    dbx_y = dy_4th(betax, i, j, k, h, nx, ny, nz)
    dbx_z = dz_4th(betax, i, j, k, h, nx, ny, nz)
    dby_x = dx_4th(betay, i, j, k, h, nx, ny, nz)
    dby_y = dy_4th(betay, i, j, k, h, nx, ny, nz)
    dby_z = dz_4th(betay, i, j, k, h, nx, ny, nz)
    dbz_x = dx_4th(betaz, i, j, k, h, nx, ny, nz)
    dbz_y = dy_4th(betaz, i, j, k, h, nx, ny, nz)
    dbz_z = dz_4th(betaz, i, j, k, h, nx, ny, nz)
    
    div_beta = dbx_x + dby_y + dbz_z
    
    # Advection terms: βᵏ∂ₖγ̃ᵢⱼ
    dgxx_adv = bx * dx_4th(gtxx, i, j, k, h, nx, ny, nz) + \
               by * dy_4th(gtxx, i, j, k, h, nx, ny, nz) + \
               bz * dz_4th(gtxx, i, j, k, h, nx, ny, nz)
    dgxy_adv = bx * dx_4th(gtxy, i, j, k, h, nx, ny, nz) + \
               by * dy_4th(gtxy, i, j, k, h, nx, ny, nz) + \
               bz * dz_4th(gtxy, i, j, k, h, nx, ny, nz)
    dgxz_adv = bx * dx_4th(gtxz, i, j, k, h, nx, ny, nz) + \
               by * dy_4th(gtxz, i, j, k, h, nx, ny, nz) + \
               bz * dz_4th(gtxz, i, j, k, h, nx, ny, nz)
    dgyy_adv = bx * dx_4th(gtyy, i, j, k, h, nx, ny, nz) + \
               by * dy_4th(gtyy, i, j, k, h, nx, ny, nz) + \
               bz * dz_4th(gtyy, i, j, k, h, nx, ny, nz)
    dgyz_adv = bx * dx_4th(gtyz, i, j, k, h, nx, ny, nz) + \
               by * dy_4th(gtyz, i, j, k, h, nx, ny, nz) + \
               bz * dz_4th(gtyz, i, j, k, h, nx, ny, nz)
    dgzz_adv = bx * dx_4th(gtzz, i, j, k, h, nx, ny, nz) + \
               by * dy_4th(gtzz, i, j, k, h, nx, ny, nz) + \
               bz * dz_4th(gtzz, i, j, k, h, nx, ny, nz)
    
    # Compute RHS for each component
    # γ̃ᵢₖ∂ⱼβᵏ + γ̃ⱼₖ∂ᵢβᵏ terms
    
    # xx: γ̃ₓₖ∂ₓβᵏ + γ̃ₓₖ∂ₓβᵏ = 2(gxx*dbx_x + gxy*dby_x + gxz*dbz_x)
    rhs_xx[i, j, k] = -2.0 * a * Axx + dgxx_adv + \
                       2.0 * (gxx * dbx_x + gxy * dby_x + gxz * dbz_x) - \
                       (2.0/3.0) * gxx * div_beta
    
    # xy: γ̃ₓₖ∂ᵧβᵏ + γ̃ᵧₖ∂ₓβᵏ
    rhs_xy[i, j, k] = -2.0 * a * Axy + dgxy_adv + \
                       (gxx * dbx_y + gxy * dby_y + gxz * dbz_y) + \
                       (gxy * dbx_x + gyy * dby_x + gyz * dbz_x) - \
                       (2.0/3.0) * gxy * div_beta
    
    # xz: γ̃ₓₖ∂ᵤβᵏ + γ̃ᵤₖ∂ₓβᵏ
    rhs_xz[i, j, k] = -2.0 * a * Axz + dgxz_adv + \
                       (gxx * dbx_z + gxy * dby_z + gxz * dbz_z) + \
                       (gxz * dbx_x + gyz * dby_x + gzz * dbz_x) - \
                       (2.0/3.0) * gxz * div_beta
    
    # yy: 2(γ̃ᵧₖ∂ᵧβᵏ)
    rhs_yy[i, j, k] = -2.0 * a * Ayy + dgyy_adv + \
                       2.0 * (gxy * dbx_y + gyy * dby_y + gyz * dbz_y) - \
                       (2.0/3.0) * gyy * div_beta
    
    # yz: γ̃ᵧₖ∂ᵤβᵏ + γ̃ᵤₖ∂ᵧβᵏ
    rhs_yz[i, j, k] = -2.0 * a * Ayz + dgyz_adv + \
                       (gxy * dbx_z + gyy * dby_z + gyz * dbz_z) + \
                       (gxz * dbx_y + gyz * dby_y + gzz * dbz_y) - \
                       (2.0/3.0) * gyz * div_beta
    
    # zz: 2(γ̃ᵤₖ∂ᵤβᵏ)
    rhs_zz[i, j, k] = -2.0 * a * Azz + dgzz_adv + \
                       2.0 * (gxz * dbx_z + gyz * dby_z + gzz * dbz_z) - \
                       (2.0/3.0) * gzz * div_beta


@wp.kernel
def rhs_At_full(
    # Traceless extrinsic curvature
    Atxx: wp.array3d(dtype=float),
    Atxy: wp.array3d(dtype=float),
    Atxz: wp.array3d(dtype=float),
    Atyy: wp.array3d(dtype=float),
    Atyz: wp.array3d(dtype=float),
    Atzz: wp.array3d(dtype=float),
    # Other fields
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    betax: wp.array3d(dtype=float),
    betay: wp.array3d(dtype=float),
    betaz: wp.array3d(dtype=float),
    # Conformal metric (for raising indices)
    gtxx: wp.array3d(dtype=float),
    gtyy: wp.array3d(dtype=float),
    gtzz: wp.array3d(dtype=float),
    # Output RHS
    rhs_Atxx: wp.array3d(dtype=float),
    rhs_Atxy: wp.array3d(dtype=float),
    rhs_Atxz: wp.array3d(dtype=float),
    rhs_Atyy: wp.array3d(dtype=float),
    rhs_Atyz: wp.array3d(dtype=float),
    rhs_Atzz: wp.array3d(dtype=float),
    # Grid
    h: float, nx: int, ny: int, nz: int
):
    """
    ∂ₜÃᵢⱼ = e^{-4φ}[-DᵢDⱼα + αRᵢⱼ]^TF + α(KÃᵢⱼ - 2ÃᵢₖÃᵏⱼ) 
            + βᵏ∂ₖÃᵢⱼ + Ãᵢₖ∂ⱼβᵏ + Ãⱼₖ∂ᵢβᵏ - (2/3)Ãᵢⱼ∂ₖβᵏ
    
    Simplified for conformal flat metric.
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    Kval = K[i, j, k]
    phi_val = phi[i, j, k]
    e4phi = wp.exp(4.0 * phi_val)
    eminus4phi = 1.0 / e4phi
    
    # Load At values
    Axx = Atxx[i, j, k]
    Axy = Atxy[i, j, k]
    Axz = Atxz[i, j, k]
    Ayy = Atyy[i, j, k]
    Ayz = Atyz[i, j, k]
    Azz = Atzz[i, j, k]
    
    # Second derivatives of alpha (for -DᵢDⱼα term)
    d2a_xx = dxx_4th(alpha, i, j, k, h, nx, ny, nz)
    d2a_yy = dyy_4th(alpha, i, j, k, h, nx, ny, nz)
    d2a_zz = dzz_4th(alpha, i, j, k, h, nx, ny, nz)
    
    # Mixed derivatives (2nd order for simplicity)
    da_x = dx_4th(alpha, i, j, k, h, nx, ny, nz)
    da_y = dy_4th(alpha, i, j, k, h, nx, ny, nz)
    da_z = dz_4th(alpha, i, j, k, h, nx, ny, nz)
    
    # Trace of D_i D_j alpha (for flat metric)
    tr_DDa = d2a_xx + d2a_yy + d2a_zz
    
    # Trace-free part: [D_i D_j α]^TF = D_i D_j α - (1/3) δ_ij tr(D D α)
    DDa_xx_TF = d2a_xx - tr_DDa / 3.0
    DDa_yy_TF = d2a_yy - tr_DDa / 3.0
    DDa_zz_TF = d2a_zz - tr_DDa / 3.0
    # Off-diagonal terms are already trace-free
    
    # Ãᵢₖ Ãᵏⱼ term (for flat conformal metric)
    # Ãⁱʲ = e^{-4φ} Ãᵢⱼ
    AA_xx = eminus4phi * (Axx*Axx + Axy*Axy + Axz*Axz)
    AA_xy = eminus4phi * (Axx*Axy + Axy*Ayy + Axz*Ayz)
    AA_xz = eminus4phi * (Axx*Axz + Axy*Ayz + Axz*Azz)
    AA_yy = eminus4phi * (Axy*Axy + Ayy*Ayy + Ayz*Ayz)
    AA_yz = eminus4phi * (Axy*Axz + Ayy*Ayz + Ayz*Azz)
    AA_zz = eminus4phi * (Axz*Axz + Ayz*Ayz + Azz*Azz)
    
    # Shift derivatives for advection and Lie derivative terms
    bx = betax[i, j, k]
    by = betay[i, j, k]
    bz = betaz[i, j, k]
    
    dbx_x = dx_4th(betax, i, j, k, h, nx, ny, nz)
    dbx_y = dy_4th(betax, i, j, k, h, nx, ny, nz)
    dbx_z = dz_4th(betax, i, j, k, h, nx, ny, nz)
    dby_x = dx_4th(betay, i, j, k, h, nx, ny, nz)
    dby_y = dy_4th(betay, i, j, k, h, nx, ny, nz)
    dby_z = dz_4th(betay, i, j, k, h, nx, ny, nz)
    dbz_x = dx_4th(betaz, i, j, k, h, nx, ny, nz)
    dbz_y = dy_4th(betaz, i, j, k, h, nx, ny, nz)
    dbz_z = dz_4th(betaz, i, j, k, h, nx, ny, nz)
    
    div_beta = dbx_x + dby_y + dbz_z
    
    # Advection: βᵏ∂ₖÃᵢⱼ
    adv_xx = bx * dx_4th(Atxx, i, j, k, h, nx, ny, nz) + \
             by * dy_4th(Atxx, i, j, k, h, nx, ny, nz) + \
             bz * dz_4th(Atxx, i, j, k, h, nx, ny, nz)
    adv_xy = bx * dx_4th(Atxy, i, j, k, h, nx, ny, nz) + \
             by * dy_4th(Atxy, i, j, k, h, nx, ny, nz) + \
             bz * dz_4th(Atxy, i, j, k, h, nx, ny, nz)
    adv_xz = bx * dx_4th(Atxz, i, j, k, h, nx, ny, nz) + \
             by * dy_4th(Atxz, i, j, k, h, nx, ny, nz) + \
             bz * dz_4th(Atxz, i, j, k, h, nx, ny, nz)
    adv_yy = bx * dx_4th(Atyy, i, j, k, h, nx, ny, nz) + \
             by * dy_4th(Atyy, i, j, k, h, nx, ny, nz) + \
             bz * dz_4th(Atyy, i, j, k, h, nx, ny, nz)
    adv_yz = bx * dx_4th(Atyz, i, j, k, h, nx, ny, nz) + \
             by * dy_4th(Atyz, i, j, k, h, nx, ny, nz) + \
             bz * dz_4th(Atyz, i, j, k, h, nx, ny, nz)
    adv_zz = bx * dx_4th(Atzz, i, j, k, h, nx, ny, nz) + \
             by * dy_4th(Atzz, i, j, k, h, nx, ny, nz) + \
             bz * dz_4th(Atzz, i, j, k, h, nx, ny, nz)
    
    # Compute RHS: -e^{-4φ}[DDα]^TF + α(K*At - 2*At*At) + advection + Lie terms
    rhs_Atxx[i, j, k] = -eminus4phi * DDa_xx_TF + a * (Kval * Axx - 2.0 * AA_xx) + \
                         adv_xx + 2.0 * (Axx * dbx_x + Axy * dby_x + Axz * dbz_x) - \
                         (2.0/3.0) * Axx * div_beta
    
    rhs_Atxy[i, j, k] = a * (Kval * Axy - 2.0 * AA_xy) + \
                         adv_xy + (Axx * dbx_y + Axy * dby_y + Axz * dbz_y) + \
                         (Axy * dbx_x + Ayy * dby_x + Ayz * dbz_x) - \
                         (2.0/3.0) * Axy * div_beta
    
    rhs_Atxz[i, j, k] = a * (Kval * Axz - 2.0 * AA_xz) + \
                         adv_xz + (Axx * dbx_z + Axy * dby_z + Axz * dbz_z) + \
                         (Axz * dbx_x + Ayz * dby_x + Azz * dbz_x) - \
                         (2.0/3.0) * Axz * div_beta
    
    rhs_Atyy[i, j, k] = -eminus4phi * DDa_yy_TF + a * (Kval * Ayy - 2.0 * AA_yy) + \
                         adv_yy + 2.0 * (Axy * dbx_y + Ayy * dby_y + Ayz * dbz_y) - \
                         (2.0/3.0) * Ayy * div_beta
    
    rhs_Atyz[i, j, k] = a * (Kval * Ayz - 2.0 * AA_yz) + \
                         adv_yz + (Axy * dbx_z + Ayy * dby_z + Ayz * dbz_z) + \
                         (Axz * dbx_y + Ayz * dby_y + Azz * dbz_y) - \
                         (2.0/3.0) * Ayz * div_beta
    
    rhs_Atzz[i, j, k] = -eminus4phi * DDa_zz_TF + a * (Kval * Azz - 2.0 * AA_zz) + \
                         adv_zz + 2.0 * (Axz * dbx_z + Ayz * dby_z + Azz * dbz_z) - \
                         (2.0/3.0) * Azz * div_beta


@wp.kernel
def rhs_K_full(
    # Fields
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    # Traceless At (for Ãᵢⱼ Ãⁱʲ term)
    Atxx: wp.array3d(dtype=float),
    Atxy: wp.array3d(dtype=float),
    Atxz: wp.array3d(dtype=float),
    Atyy: wp.array3d(dtype=float),
    Atyz: wp.array3d(dtype=float),
    Atzz: wp.array3d(dtype=float),
    # Shift
    betax: wp.array3d(dtype=float),
    betay: wp.array3d(dtype=float),
    betaz: wp.array3d(dtype=float),
    # Output
    rhs: wp.array3d(dtype=float),
    # Grid
    h: float, nx: int, ny: int, nz: int
):
    """
    ∂ₜK = -γⁱʲDᵢDⱼα + α(ÃᵢⱼÃⁱʲ + K²/3) + βⁱ∂ᵢK
    
    For conformal flat metric: γⁱʲDᵢDⱼα ≈ e^{-4φ}∇²α
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    Kval = K[i, j, k]
    phi_val = phi[i, j, k]
    
    # Laplacian of alpha
    lap_alpha = laplacian_4th(alpha, i, j, k, h, nx, ny, nz)
    
    # Conformal factor
    e4phi = wp.exp(4.0 * phi_val)
    
    # For flat conformal metric: γⁱʲ = e^{-4φ}δⁱʲ
    # So γⁱʲDᵢDⱼα ≈ e^{-4φ} ∇²α (ignoring connection terms for now)
    
    # Ãᵢⱼ Ãⁱʲ (for flat conformal metric: Ãⁱʲ = e^{-4φ}Ãᵢⱼ)
    Axx = Atxx[i, j, k]
    Axy = Atxy[i, j, k]
    Axz = Atxz[i, j, k]
    Ayy = Atyy[i, j, k]
    Ayz = Atyz[i, j, k]
    Azz = Atzz[i, j, k]
    
    # For flat conformal: AijAij = e^{-8φ}(Axx² + Ayy² + Azz² + 2Axy² + 2Axz² + 2Ayz²)
    At_sq = Axx*Axx + Ayy*Ayy + Azz*Azz + 2.0*(Axy*Axy + Axz*Axz + Ayz*Ayz)
    
    # Advection
    bx = betax[i, j, k]
    by = betay[i, j, k]
    bz = betaz[i, j, k]
    dK_x = dx_4th(K, i, j, k, h, nx, ny, nz)
    dK_y = dy_4th(K, i, j, k, h, nx, ny, nz)
    dK_z = dz_4th(K, i, j, k, h, nx, ny, nz)
    advection = bx * dK_x + by * dK_y + bz * dK_z
    
    rhs[i, j, k] = -lap_alpha / e4phi + a * (At_sq / e4phi + Kval * Kval / 3.0) + advection


@wp.kernel
def rhs_alpha_1log(
    alpha: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    betax: wp.array3d(dtype=float),
    betay: wp.array3d(dtype=float),
    betaz: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    h: float, nx: int, ny: int, nz: int
):
    """
    1+log slicing: ∂ₜα = -2αK + βⁱ∂ᵢα
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    Kval = K[i, j, k]
    
    # Advection
    bx = betax[i, j, k]
    by = betay[i, j, k]
    bz = betaz[i, j, k]
    da_x = dx_4th(alpha, i, j, k, h, nx, ny, nz)
    da_y = dy_4th(alpha, i, j, k, h, nx, ny, nz)
    da_z = dz_4th(alpha, i, j, k, h, nx, ny, nz)
    advection = bx * da_x + by * da_y + bz * da_z
    
    rhs[i, j, k] = -2.0 * a * Kval + advection


@wp.kernel
def rhs_shift_gamma_driver(
    betax: wp.array3d(dtype=float),
    betay: wp.array3d(dtype=float),
    betaz: wp.array3d(dtype=float),
    Bx: wp.array3d(dtype=float),
    By: wp.array3d(dtype=float),
    Bz: wp.array3d(dtype=float),
    Gtx: wp.array3d(dtype=float),
    Gty: wp.array3d(dtype=float),
    Gtz: wp.array3d(dtype=float),
    rhs_betax: wp.array3d(dtype=float),
    rhs_betay: wp.array3d(dtype=float),
    rhs_betaz: wp.array3d(dtype=float),
    rhs_Bx: wp.array3d(dtype=float),
    rhs_By: wp.array3d(dtype=float),
    rhs_Bz: wp.array3d(dtype=float),
    eta: float,  # Damping parameter
    h: float, nx: int, ny: int, nz: int
):
    """
    Gamma-driver shift evolution:
    ∂ₜβⁱ = (3/4)Bⁱ + βʲ∂ⱼβⁱ
    ∂ₜBⁱ = ∂ₜΓ̃ⁱ - ηBⁱ + βʲ∂ⱼBⁱ
    
    Simplified version (without computing full ∂ₜΓ̃):
    ∂ₜBⁱ = (3/4)Γ̃ⁱ - ηBⁱ + βʲ∂ⱼBⁱ
    """
    i, j, k = wp.tid()
    
    bx = betax[i, j, k]
    by = betay[i, j, k]
    bz = betaz[i, j, k]
    
    Bx_val = Bx[i, j, k]
    By_val = By[i, j, k]
    Bz_val = Bz[i, j, k]
    
    Gtx_val = Gtx[i, j, k]
    Gty_val = Gty[i, j, k]
    Gtz_val = Gtz[i, j, k]
    
    # Advection of β
    dbx_x = dx_4th(betax, i, j, k, h, nx, ny, nz)
    dbx_y = dy_4th(betax, i, j, k, h, nx, ny, nz)
    dbx_z = dz_4th(betax, i, j, k, h, nx, ny, nz)
    dby_x = dx_4th(betay, i, j, k, h, nx, ny, nz)
    dby_y = dy_4th(betay, i, j, k, h, nx, ny, nz)
    dby_z = dz_4th(betay, i, j, k, h, nx, ny, nz)
    dbz_x = dx_4th(betaz, i, j, k, h, nx, ny, nz)
    dbz_y = dy_4th(betaz, i, j, k, h, nx, ny, nz)
    dbz_z = dz_4th(betaz, i, j, k, h, nx, ny, nz)
    
    adv_bx = bx * dbx_x + by * dbx_y + bz * dbx_z
    adv_by = bx * dby_x + by * dby_y + bz * dby_z
    adv_bz = bx * dbz_x + by * dbz_y + bz * dbz_z
    
    # Advection of B
    dBx_x = dx_4th(Bx, i, j, k, h, nx, ny, nz)
    dBx_y = dy_4th(Bx, i, j, k, h, nx, ny, nz)
    dBx_z = dz_4th(Bx, i, j, k, h, nx, ny, nz)
    dBy_x = dx_4th(By, i, j, k, h, nx, ny, nz)
    dBy_y = dy_4th(By, i, j, k, h, nx, ny, nz)
    dBy_z = dz_4th(By, i, j, k, h, nx, ny, nz)
    dBz_x = dx_4th(Bz, i, j, k, h, nx, ny, nz)
    dBz_y = dy_4th(Bz, i, j, k, h, nx, ny, nz)
    dBz_z = dz_4th(Bz, i, j, k, h, nx, ny, nz)
    
    adv_Bx = bx * dBx_x + by * dBx_y + bz * dBx_z
    adv_By = bx * dBy_x + by * dBy_y + bz * dBy_z
    adv_Bz = bx * dBz_x + by * dBz_y + bz * dBz_z
    
    # ∂ₜβⁱ = (3/4)Bⁱ + advection
    rhs_betax[i, j, k] = 0.75 * Bx_val + adv_bx
    rhs_betay[i, j, k] = 0.75 * By_val + adv_by
    rhs_betaz[i, j, k] = 0.75 * Bz_val + adv_bz
    
    # ∂ₜBⁱ = (3/4)Γ̃ⁱ - ηBⁱ + advection (simplified)
    rhs_Bx[i, j, k] = 0.75 * Gtx_val - eta * Bx_val + adv_Bx
    rhs_By[i, j, k] = 0.75 * Gty_val - eta * By_val + adv_By
    rhs_Bz[i, j, k] = 0.75 * Gtz_val - eta * Bz_val + adv_Bz


# ============================================================================
# RK4 Helpers
# ============================================================================

@wp.kernel
def axpy(y: wp.array3d(dtype=float), 
         x: wp.array3d(dtype=float),
         a: float):
    """y = y + a*x"""
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + a * x[i, j, k]


@wp.kernel
def copy_array(dst: wp.array3d(dtype=float),
               src: wp.array3d(dtype=float)):
    """dst = src"""
    i, j, k = wp.tid()
    dst[i, j, k] = src[i, j, k]


@wp.kernel
def rk4_combine(y: wp.array3d(dtype=float),
                k1: wp.array3d(dtype=float),
                k2: wp.array3d(dtype=float),
                k3: wp.array3d(dtype=float),
                k4: wp.array3d(dtype=float),
                dt: float):
    """y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)"""
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + dt / 6.0 * (
        k1[i, j, k] + 2.0 * k2[i, j, k] + 2.0 * k3[i, j, k] + k4[i, j, k]
    )


# ============================================================================
# Constraint Monitoring
# ============================================================================

# ============================================================================
# Momentum Constraints
# ============================================================================

@wp.kernel
def compute_momentum_constraint(
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    Atxx: wp.array3d(dtype=float),
    Atxy: wp.array3d(dtype=float),
    Atxz: wp.array3d(dtype=float),
    Atyy: wp.array3d(dtype=float),
    Atyz: wp.array3d(dtype=float),
    Atzz: wp.array3d(dtype=float),
    Mx: wp.array3d(dtype=float),
    My: wp.array3d(dtype=float),
    Mz: wp.array3d(dtype=float),
    h: float, nx: int, ny: int, nz: int
):
    """
    Momentum constraints: Mⁱ = ∂ⱼÃⁱʲ - (2/3)γ̃ⁱʲ∂ⱼK + 6Ãⁱʲ∂ⱼφ = 0
    
    For conformal flat metric: Ãⁱʲ = e^{-4φ}Ãᵢⱼ
    """
    i, j, k = wp.tid()
    
    phi_val = phi[i, j, k]
    e4phi = wp.exp(4.0 * phi_val)
    eminus4phi = 1.0 / e4phi
    
    # Derivatives of K
    dK_x = dx_4th(K, i, j, k, h, nx, ny, nz)
    dK_y = dy_4th(K, i, j, k, h, nx, ny, nz)
    dK_z = dz_4th(K, i, j, k, h, nx, ny, nz)
    
    # Derivatives of phi
    dphi_x = dx_4th(phi, i, j, k, h, nx, ny, nz)
    dphi_y = dy_4th(phi, i, j, k, h, nx, ny, nz)
    dphi_z = dz_4th(phi, i, j, k, h, nx, ny, nz)
    
    # Derivatives of At components
    dAtxx_x = dx_4th(Atxx, i, j, k, h, nx, ny, nz)
    dAtxy_x = dx_4th(Atxy, i, j, k, h, nx, ny, nz)
    dAtxz_x = dx_4th(Atxz, i, j, k, h, nx, ny, nz)
    dAtxy_y = dy_4th(Atxy, i, j, k, h, nx, ny, nz)
    dAtyy_y = dy_4th(Atyy, i, j, k, h, nx, ny, nz)
    dAtyz_y = dy_4th(Atyz, i, j, k, h, nx, ny, nz)
    dAtxz_z = dz_4th(Atxz, i, j, k, h, nx, ny, nz)
    dAtyz_z = dz_4th(Atyz, i, j, k, h, nx, ny, nz)
    dAtzz_z = dz_4th(Atzz, i, j, k, h, nx, ny, nz)
    
    # At values for the 6φ term
    Axx = Atxx[i, j, k]
    Axy = Atxy[i, j, k]
    Axz = Atxz[i, j, k]
    Ayy = Atyy[i, j, k]
    Ayz = Atyz[i, j, k]
    Azz = Atzz[i, j, k]
    
    # M^x = ∂_j Ã^{xj} - (2/3)∂^x K + 6 Ã^{xj} ∂_j φ
    # For flat conformal: Ã^{xj} = e^{-4φ} Ã_{xj}
    Mx[i, j, k] = eminus4phi * (dAtxx_x + dAtxy_y + dAtxz_z) - \
                  (2.0/3.0) * eminus4phi * dK_x + \
                  6.0 * eminus4phi * (Axx * dphi_x + Axy * dphi_y + Axz * dphi_z)
    
    My[i, j, k] = eminus4phi * (dAtxy_x + dAtyy_y + dAtyz_z) - \
                  (2.0/3.0) * eminus4phi * dK_y + \
                  6.0 * eminus4phi * (Axy * dphi_x + Ayy * dphi_y + Ayz * dphi_z)
    
    Mz[i, j, k] = eminus4phi * (dAtxz_x + dAtyz_y + dAtzz_z) - \
                  (2.0/3.0) * eminus4phi * dK_z + \
                  6.0 * eminus4phi * (Axz * dphi_x + Ayz * dphi_y + Azz * dphi_z)


@wp.kernel
def compute_hamiltonian_constraint(
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    Atxx: wp.array3d(dtype=float),
    Atxy: wp.array3d(dtype=float),
    Atxz: wp.array3d(dtype=float),
    Atyy: wp.array3d(dtype=float),
    Atyz: wp.array3d(dtype=float),
    Atzz: wp.array3d(dtype=float),
    H: wp.array3d(dtype=float),
    h: float, nx: int, ny: int, nz: int
):
    """
    Hamiltonian constraint: H = R + K² - AᵢⱼAⁱʲ = 0
    
    For conformal flat space, R ≈ -8e^{-4φ}∇²φ - 8e^{-4φ}(∂φ)²
    """
    i, j, k = wp.tid()
    
    Kval = K[i, j, k]
    phi_val = phi[i, j, k]
    e4phi = wp.exp(4.0 * phi_val)
    
    # Laplacian of phi
    lap_phi = laplacian_4th(phi, i, j, k, h, nx, ny, nz)
    
    # Gradient squared (approximation)
    dphi_x = dx_4th(phi, i, j, k, h, nx, ny, nz)
    dphi_y = dy_4th(phi, i, j, k, h, nx, ny, nz)
    dphi_z = dz_4th(phi, i, j, k, h, nx, ny, nz)
    grad_phi_sq = dphi_x*dphi_x + dphi_y*dphi_y + dphi_z*dphi_z
    
    # Ricci scalar for conformal flat metric (approximate)
    R = -8.0 * lap_phi / e4phi - 8.0 * grad_phi_sq / e4phi
    
    # AijAij
    Axx = Atxx[i, j, k]
    Axy = Atxy[i, j, k]
    Axz = Atxz[i, j, k]
    Ayy = Atyy[i, j, k]
    Ayz = Atyz[i, j, k]
    Azz = Atzz[i, j, k]
    At_sq = (Axx*Axx + Ayy*Ayy + Azz*Azz + 2.0*(Axy*Axy + Axz*Axz + Ayz*Ayz)) / (e4phi * e4phi)
    
    H[i, j, k] = R + Kval * Kval - At_sq


# ============================================================================
# Kreiss-Oliger Dissipation
# ============================================================================

@wp.func
def ko_diss_x(f: wp.array3d(dtype=float), i: int, j: int, k: int,
              sigma: float, nx: int, ny: int, nz: int) -> float:
    """KO dissipation in x direction."""
    fm3 = get_val(f, i - 3, j, k, nx, ny, nz)
    fm2 = get_val(f, i - 2, j, k, nx, ny, nz)
    fm1 = get_val(f, i - 1, j, k, nx, ny, nz)
    f0 = f[i, j, k]
    fp1 = get_val(f, i + 1, j, k, nx, ny, nz)
    fp2 = get_val(f, i + 2, j, k, nx, ny, nz)
    fp3 = get_val(f, i + 3, j, k, nx, ny, nz)
    return -sigma / 64.0 * (fm3 - 6.0*fm2 + 15.0*fm1 - 20.0*f0 + 15.0*fp1 - 6.0*fp2 + fp3)


@wp.kernel
def add_ko_dissipation(f: wp.array3d(dtype=float),
                       rhs: wp.array3d(dtype=float),
                       sigma: float,
                       nx: int, ny: int, nz: int):
    """Add KO dissipation to RHS."""
    i, j, k = wp.tid()
    
    diss_x = ko_diss_x(f, i, j, k, sigma, nx, ny, nz)
    
    # Y direction
    fm3 = get_val(f, i, j - 3, k, nx, ny, nz)
    fm2 = get_val(f, i, j - 2, k, nx, ny, nz)
    fm1 = get_val(f, i, j - 1, k, nx, ny, nz)
    f0 = f[i, j, k]
    fp1 = get_val(f, i, j + 1, k, nx, ny, nz)
    fp2 = get_val(f, i, j + 2, k, nx, ny, nz)
    fp3 = get_val(f, i, j + 3, k, nx, ny, nz)
    diss_y = -sigma / 64.0 * (fm3 - 6.0*fm2 + 15.0*fm1 - 20.0*f0 + 15.0*fp1 - 6.0*fp2 + fp3)
    
    # Z direction
    fm3 = get_val(f, i, j, k - 3, nx, ny, nz)
    fm2 = get_val(f, i, j, k - 2, nx, ny, nz)
    fm1 = get_val(f, i, j, k - 1, nx, ny, nz)
    fp1 = get_val(f, i, j, k + 1, nx, ny, nz)
    fp2 = get_val(f, i, j, k + 2, nx, ny, nz)
    fp3 = get_val(f, i, j, k + 3, nx, ny, nz)
    diss_z = -sigma / 64.0 * (fm3 - 6.0*fm2 + 15.0*fm1 - 20.0*f0 + 15.0*fp1 - 6.0*fp2 + fp3)
    
    rhs[i, j, k] = rhs[i, j, k] + diss_x + diss_y + diss_z


# ============================================================================
# Gauge Wave Initial Data
# ============================================================================

# ============================================================================
# Brill-Lindquist Initial Data (Single Puncture Black Hole)
# ============================================================================

@wp.kernel
def init_brill_lindquist(
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    gtxx: wp.array3d(dtype=float),
    gtyy: wp.array3d(dtype=float),
    gtzz: wp.array3d(dtype=float),
    mass: float,
    h: float,
    nx: int, ny: int, nz: int
):
    """
    Initialize Brill-Lindquist (puncture) data for a single black hole.
    
    The conformal factor is: ψ = 1 + M/(2r)
    where M is the ADM mass and r is the coordinate distance from the puncture.
    
    In BSSN: φ = ln(ψ)/4, γ̃ᵢⱼ = δᵢⱼ (conformally flat)
    K = 0, Ãᵢⱼ = 0 (time symmetric)
    
    Pre-collapsed lapse: α = 1/ψ² = 1/(1 + M/(2r))²
    """
    i, j, k = wp.tid()
    
    # Coordinates (centered grid)
    x = (float(i) - float(nx)/2.0 + 0.5) * h
    y = (float(j) - float(ny)/2.0 + 0.5) * h
    z = (float(k) - float(nz)/2.0 + 0.5) * h
    
    # Distance from puncture (avoid r=0)
    r = wp.sqrt(x*x + y*y + z*z)
    r = wp.max(r, 0.1 * h)  # Regularize at puncture
    
    # Conformal factor ψ = 1 + M/(2r)
    psi = 1.0 + mass / (2.0 * r)
    
    # BSSN conformal factor: φ = ln(ψ)
    # Note: Some formulations use φ = ln(ψ)/4, here we use e^{4φ} = ψ⁴
    # So φ = ln(ψ) gives e^{4φ} = ψ⁴ ✓
    phi[i, j, k] = wp.log(psi)
    
    # Time-symmetric: K = 0
    K[i, j, k] = 0.0
    
    # Pre-collapsed lapse: α = 1/ψ² (avoids slice stretching)
    alpha[i, j, k] = 1.0 / (psi * psi)
    
    # Conformal metric: flat (already initialized to identity)
    gtxx[i, j, k] = 1.0
    gtyy[i, j, k] = 1.0
    gtzz[i, j, k] = 1.0


@wp.kernel
def init_binary_brill_lindquist(
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    gtxx: wp.array3d(dtype=float),
    gtyy: wp.array3d(dtype=float),
    gtzz: wp.array3d(dtype=float),
    mass1: float, mass2: float,
    x1: float, x2: float,  # x-positions of punctures
    h: float,
    nx: int, ny: int, nz: int
):
    """
    Initialize Brill-Lindquist data for binary black holes (two punctures).
    
    The conformal factor is: ψ = 1 + M₁/(2r₁) + M₂/(2r₂)
    where r₁, r₂ are distances from each puncture.
    """
    i, j, k = wp.tid()
    
    # Coordinates (centered grid)
    x = (float(i) - float(nx)/2.0 + 0.5) * h
    y = (float(j) - float(ny)/2.0 + 0.5) * h
    z = (float(k) - float(nz)/2.0 + 0.5) * h
    
    # Distance from first puncture at (x1, 0, 0)
    dx1 = x - x1
    r1 = wp.sqrt(dx1*dx1 + y*y + z*z)
    r1 = wp.max(r1, 0.1 * h)
    
    # Distance from second puncture at (x2, 0, 0)
    dx2 = x - x2
    r2 = wp.sqrt(dx2*dx2 + y*y + z*z)
    r2 = wp.max(r2, 0.1 * h)
    
    # Conformal factor
    psi = 1.0 + mass1 / (2.0 * r1) + mass2 / (2.0 * r2)
    
    # BSSN conformal factor
    phi[i, j, k] = wp.log(psi)
    
    # Time-symmetric: K = 0
    K[i, j, k] = 0.0
    
    # Pre-collapsed lapse
    alpha[i, j, k] = 1.0 / (psi * psi)
    
    # Conformal metric: flat
    gtxx[i, j, k] = 1.0
    gtyy[i, j, k] = 1.0
    gtzz[i, j, k] = 1.0


@wp.kernel
def init_gauge_wave(
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    amplitude: float,
    wavelength: float,
    h: float,
    nx: int, ny: int, nz: int
):
    """
    Initialize 1D gauge wave propagating in x direction.
    
    α = 1 - A*sin(2πx/L)
    K = A*(2π/L)*cos(2πx/L) / (2*α)
    
    This is a known exact solution that should propagate unchanged.
    """
    i, j, k = wp.tid()
    
    x = float(i) * h
    pi = 3.141592653589793
    kwave = 2.0 * pi / wavelength
    
    # Lapse perturbation
    a = 1.0 - amplitude * wp.sin(kwave * x)
    alpha[i, j, k] = a
    
    # Extrinsic curvature from gauge wave
    K[i, j, k] = amplitude * kwave * wp.cos(kwave * x) / (2.0 * a)
    
    # φ = 0 for gauge wave
    phi[i, j, k] = 0.0


# ============================================================================
# Sommerfeld (Radiative) Boundary Conditions
# ============================================================================

@wp.kernel
def apply_sommerfeld_bc(
    f: wp.array3d(dtype=float),
    f0: float,  # asymptotic value
    h: float,
    nx: int, ny: int, nz: int
):
    """
    Apply Sommerfeld outgoing wave boundary condition.
    
    At boundaries: ∂f/∂t + ∂f/∂r = 0 where r is radial direction
    Discretized: f_new = f_old - (dt/dr)*(f - f_neighbor) → f0 at infinity
    
    Simplified: just extrapolate using interior values and blend toward f0.
    """
    i, j, k = wp.tid()
    
    # Only apply at boundary cells (3 layers for 4th order stencil)
    is_boundary = (i < 3 or i >= nx - 3 or
                   j < 3 or j >= ny - 3 or
                   k < 3 or k >= nz - 3)
    
    if is_boundary:
        # Distance from center
        cx = float(nx) / 2.0
        cy = float(ny) / 2.0
        cz = float(nz) / 2.0
        
        dx = float(i) - cx
        dy = float(j) - cy
        dz = float(k) - cz
        
        r = wp.sqrt(dx*dx + dy*dy + dz*dz)
        r_max = wp.sqrt(cx*cx + cy*cy + cz*cz)
        
        # Blend factor: 0 at interior, 1 at boundary
        blend = wp.clamp((r - (r_max - 3.0)) / 3.0, 0.0, 1.0)
        
        # Blend current value toward asymptotic value
        f[i, j, k] = (1.0 - blend) * f[i, j, k] + blend * f0


@wp.kernel
def apply_sommerfeld_rhs(
    f: wp.array3d(dtype=float),
    f0: float,  # asymptotic value
    rhs: wp.array3d(dtype=float),
    h: float,
    nx: int, ny: int, nz: int
):
    """
    Apply Sommerfeld condition to RHS at boundaries.
    
    ∂f/∂t = -(1/r)(f - f0) - ∂f/∂r
    
    This drives f toward f0 at boundaries.
    """
    i, j, k = wp.tid()
    
    # Distance from center
    cx = float(nx) / 2.0
    cy = float(ny) / 2.0
    cz = float(nz) / 2.0
    
    dx_c = float(i) - cx
    dy_c = float(j) - cy  
    dz_c = float(k) - cz
    
    r = wp.sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c)
    r = wp.max(r, 1.0)  # Avoid division by zero
    
    r_max = wp.sqrt(cx*cx + cy*cy + cz*cz)
    
    # Only apply near boundaries
    if r > r_max - 5.0:
        # Radial derivative (approximate using centered difference)
        # ∂f/∂r ≈ (x/r)∂f/∂x + (y/r)∂f/∂y + (z/r)∂f/∂z
        
        nx_hat = dx_c / r
        ny_hat = dy_c / r
        nz_hat = dz_c / r
        
        df_dx = dx_4th(f, i, j, k, h, nx, ny, nz)
        df_dy = dy_4th(f, i, j, k, h, nx, ny, nz)
        df_dz = dz_4th(f, i, j, k, h, nx, ny, nz)
        
        df_dr = nx_hat * df_dx + ny_hat * df_dy + nz_hat * df_dz
        
        # Sommerfeld: ∂f/∂t = -(f - f0)/r - ∂f/∂r (outgoing wave)
        damping = (f[i, j, k] - f0) / r
        
        # Blend factor
        blend = wp.clamp((r - (r_max - 5.0)) / 5.0, 0.0, 1.0)
        
        # Modify RHS
        rhs[i, j, k] = (1.0 - blend) * rhs[i, j, k] + blend * (-damping - df_dr)


# ============================================================================
# Main Evolution Class
# ============================================================================

class BSSNEvolver:
    """BSSN evolution with RK4 time integration."""
    
    def __init__(self, nx, ny, nz, dx, sigma=0.1, use_sommerfeld=False):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.sigma = sigma  # KO dissipation strength
        self.use_sommerfeld = use_sommerfeld  # Use Sommerfeld BCs
        
        shape = (nx, ny, nz)
        
        # State variables
        self.phi = wp.zeros(shape, dtype=float)
        self.K = wp.zeros(shape, dtype=float)
        self.alpha = wp.ones(shape, dtype=float)
        
        # Conformal metric (diagonal = 1 for flat)
        self.gtxx = wp.ones(shape, dtype=float)
        self.gtxy = wp.zeros(shape, dtype=float)
        self.gtxz = wp.zeros(shape, dtype=float)
        self.gtyy = wp.ones(shape, dtype=float)
        self.gtyz = wp.zeros(shape, dtype=float)
        self.gtzz = wp.ones(shape, dtype=float)
        
        # Traceless extrinsic curvature
        self.Atxx = wp.zeros(shape, dtype=float)
        self.Atxy = wp.zeros(shape, dtype=float)
        self.Atxz = wp.zeros(shape, dtype=float)
        self.Atyy = wp.zeros(shape, dtype=float)
        self.Atyz = wp.zeros(shape, dtype=float)
        self.Atzz = wp.zeros(shape, dtype=float)
        
        # Shift (zero for now)
        self.betax = wp.zeros(shape, dtype=float)
        self.betay = wp.zeros(shape, dtype=float)
        self.betaz = wp.zeros(shape, dtype=float)
        
        # RHS arrays
        self.rhs_phi = wp.zeros(shape, dtype=float)
        self.rhs_K = wp.zeros(shape, dtype=float)
        self.rhs_alpha = wp.zeros(shape, dtype=float)
        
        # RK4 intermediate storage
        self.k1_phi = wp.zeros(shape, dtype=float)
        self.k2_phi = wp.zeros(shape, dtype=float)
        self.k3_phi = wp.zeros(shape, dtype=float)
        self.k4_phi = wp.zeros(shape, dtype=float)
        
        self.k1_K = wp.zeros(shape, dtype=float)
        self.k2_K = wp.zeros(shape, dtype=float)
        self.k3_K = wp.zeros(shape, dtype=float)
        self.k4_K = wp.zeros(shape, dtype=float)
        
        self.k1_alpha = wp.zeros(shape, dtype=float)
        self.k2_alpha = wp.zeros(shape, dtype=float)
        self.k3_alpha = wp.zeros(shape, dtype=float)
        self.k4_alpha = wp.zeros(shape, dtype=float)
        
        # Temporary storage for RK4 stages
        self.tmp_phi = wp.zeros(shape, dtype=float)
        self.tmp_K = wp.zeros(shape, dtype=float)
        self.tmp_alpha = wp.zeros(shape, dtype=float)
        
        # Constraint monitoring
        self.H = wp.zeros(shape, dtype=float)
        self.Mx = wp.zeros(shape, dtype=float)
        self.My = wp.zeros(shape, dtype=float)
        self.Mz = wp.zeros(shape, dtype=float)
    
    def compute_rhs(self, phi, K, alpha, rhs_phi, rhs_K, rhs_alpha):
        """Compute RHS for all evolved variables."""
        dim = (self.nx, self.ny, self.nz)
        h = self.dx
        
        # φ RHS
        wp.launch(rhs_phi_full, dim=dim, inputs=[
            phi, K, alpha, self.betax, self.betay, self.betaz,
            rhs_phi, h, self.nx, self.ny, self.nz
        ])
        
        # K RHS  
        wp.launch(rhs_K_full, dim=dim, inputs=[
            K, alpha, phi,
            self.Atxx, self.Atxy, self.Atxz,
            self.Atyy, self.Atyz, self.Atzz,
            self.betax, self.betay, self.betaz,
            rhs_K, h, self.nx, self.ny, self.nz
        ])
        
        # α RHS (1+log slicing)
        wp.launch(rhs_alpha_1log, dim=dim, inputs=[
            alpha, K, self.betax, self.betay, self.betaz,
            rhs_alpha, h, self.nx, self.ny, self.nz
        ])
        
        # Add KO dissipation
        if self.sigma > 0:
            wp.launch(add_ko_dissipation, dim=dim, inputs=[
                phi, rhs_phi, self.sigma, self.nx, self.ny, self.nz
            ])
            wp.launch(add_ko_dissipation, dim=dim, inputs=[
                K, rhs_K, self.sigma, self.nx, self.ny, self.nz
            ])
            wp.launch(add_ko_dissipation, dim=dim, inputs=[
                alpha, rhs_alpha, self.sigma, self.nx, self.ny, self.nz
            ])
        
        # Apply Sommerfeld boundary conditions
        if self.use_sommerfeld:
            wp.launch(apply_sommerfeld_rhs, dim=dim, inputs=[
                phi, 0.0, rhs_phi, h, self.nx, self.ny, self.nz
            ])
            wp.launch(apply_sommerfeld_rhs, dim=dim, inputs=[
                K, 0.0, rhs_K, h, self.nx, self.ny, self.nz
            ])
            wp.launch(apply_sommerfeld_rhs, dim=dim, inputs=[
                alpha, 1.0, rhs_alpha, h, self.nx, self.ny, self.nz
            ])
    
    def step_rk4(self, dt):
        """Perform one RK4 timestep."""
        dim = (self.nx, self.ny, self.nz)
        
        # k1 = f(y_n)
        self.compute_rhs(self.phi, self.K, self.alpha,
                         self.k1_phi, self.k1_K, self.k1_alpha)
        
        # y_tmp = y_n + dt/2 * k1
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_phi, self.phi])
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_K, self.K])
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_alpha, self.alpha])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_phi, self.k1_phi, 0.5 * dt])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_K, self.k1_K, 0.5 * dt])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_alpha, self.k1_alpha, 0.5 * dt])
        
        # k2 = f(y_n + dt/2 * k1)
        self.compute_rhs(self.tmp_phi, self.tmp_K, self.tmp_alpha,
                         self.k2_phi, self.k2_K, self.k2_alpha)
        
        # y_tmp = y_n + dt/2 * k2
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_phi, self.phi])
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_K, self.K])
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_alpha, self.alpha])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_phi, self.k2_phi, 0.5 * dt])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_K, self.k2_K, 0.5 * dt])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_alpha, self.k2_alpha, 0.5 * dt])
        
        # k3 = f(y_n + dt/2 * k2)
        self.compute_rhs(self.tmp_phi, self.tmp_K, self.tmp_alpha,
                         self.k3_phi, self.k3_K, self.k3_alpha)
        
        # y_tmp = y_n + dt * k3
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_phi, self.phi])
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_K, self.K])
        wp.launch(copy_array, dim=dim, inputs=[self.tmp_alpha, self.alpha])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_phi, self.k3_phi, dt])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_K, self.k3_K, dt])
        wp.launch(axpy, dim=dim, inputs=[self.tmp_alpha, self.k3_alpha, dt])
        
        # k4 = f(y_n + dt * k3)
        self.compute_rhs(self.tmp_phi, self.tmp_K, self.tmp_alpha,
                         self.k4_phi, self.k4_K, self.k4_alpha)
        
        # y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        wp.launch(rk4_combine, dim=dim, inputs=[
            self.phi, self.k1_phi, self.k2_phi, self.k3_phi, self.k4_phi, dt
        ])
        wp.launch(rk4_combine, dim=dim, inputs=[
            self.K, self.k1_K, self.k2_K, self.k3_K, self.k4_K, dt
        ])
        wp.launch(rk4_combine, dim=dim, inputs=[
            self.alpha, self.k1_alpha, self.k2_alpha, self.k3_alpha, self.k4_alpha, dt
        ])
    
    def compute_constraints(self):
        """Compute Hamiltonian and momentum constraints."""
        dim = (self.nx, self.ny, self.nz)
        
        # Hamiltonian constraint
        wp.launch(compute_hamiltonian_constraint, dim=dim,
                  inputs=[
                      self.phi, self.K,
                      self.Atxx, self.Atxy, self.Atxz,
                      self.Atyy, self.Atyz, self.Atzz,
                      self.H, self.dx, self.nx, self.ny, self.nz
                  ])
        
        # Momentum constraints
        wp.launch(compute_momentum_constraint, dim=dim,
                  inputs=[
                      self.phi, self.K,
                      self.Atxx, self.Atxy, self.Atxz,
                      self.Atyy, self.Atyz, self.Atzz,
                      self.Mx, self.My, self.Mz,
                      self.dx, self.nx, self.ny, self.nz
                  ])
        
        H_max = np.max(np.abs(self.H.numpy()))
        M_max = max(np.max(np.abs(self.Mx.numpy())),
                    np.max(np.abs(self.My.numpy())),
                    np.max(np.abs(self.Mz.numpy())))
        
        return H_max, M_max
    
    def init_gauge_wave(self, amplitude=0.01, wavelength=1.0):
        """Initialize gauge wave test."""
        wp.launch(init_gauge_wave, dim=(self.nx, self.ny, self.nz),
                  inputs=[self.phi, self.K, self.alpha, 
                          amplitude, wavelength, self.dx,
                          self.nx, self.ny, self.nz])
    
    def init_brill_lindquist(self, mass=1.0):
        """Initialize Brill-Lindquist puncture data for single black hole."""
        wp.launch(init_brill_lindquist, dim=(self.nx, self.ny, self.nz),
                  inputs=[self.phi, self.K, self.alpha,
                          self.gtxx, self.gtyy, self.gtzz,
                          mass, self.dx,
                          self.nx, self.ny, self.nz])
    
    def init_binary_bh(self, mass1=0.5, mass2=0.5, separation=2.0):
        """Initialize binary black hole (two punctures) data."""
        x1 = -separation / 2.0
        x2 = separation / 2.0
        wp.launch(init_binary_brill_lindquist, dim=(self.nx, self.ny, self.nz),
                  inputs=[self.phi, self.K, self.alpha,
                          self.gtxx, self.gtyy, self.gtzz,
                          mass1, mass2, x1, x2, self.dx,
                          self.nx, self.ny, self.nz])


def test_gauge_wave():
    """Test gauge wave evolution."""
    wp.init()
    
    nx = 32
    dx = 1.0 / nx
    dt = 0.25 * dx  # CFL condition
    n_steps = 100
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.1)
    evolver.init_gauge_wave(amplitude=0.01, wavelength=1.0)
    
    print(f"Gauge wave test: {nx}³ grid, dt={dt:.4f}, {n_steps} steps")
    print(f"Initial: α_max={np.max(evolver.alpha.numpy()):.6f}, α_min={np.min(evolver.alpha.numpy()):.6f}")
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
        
        if step % 20 == 0:
            alpha_np = evolver.alpha.numpy()
            K_np = evolver.K.numpy()
            H_max = evolver.compute_constraints()
            print(f"  Step {step}: α∈[{alpha_np.min():.4f},{alpha_np.max():.4f}], "
                  f"|K|_max={np.max(np.abs(K_np)):.4e}, H_max={H_max:.4e}")
    
    # Check stability
    alpha_np = evolver.alpha.numpy()
    stable = np.all(np.isfinite(alpha_np)) and alpha_np.min() > 0.5
    print(f"\nFinal: α∈[{alpha_np.min():.4f},{alpha_np.max():.4f}]")
    print(f"Test {'PASSED' if stable else 'FAILED'}")
    
    return stable


if __name__ == "__main__":
    test_gauge_wave()
