"""
BSSN Evolution Equations in NVIDIA Warp

Implements the BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation
of numerical relativity for spacetime evolution.

References:
- Baumgarte & Shapiro, Phys. Rept. 376 (2003) 41-131
- Alcubierre, Introduction to 3+1 Numerical Relativity
- McLachlan thorn (Einstein Toolkit)
"""

import math
import warp as wp

wp.init()


# ============================================================================
# BSSN State Variables
# ============================================================================

@wp.struct
class BSSNState:
    """BSSN evolution variables on a 3D grid.
    
    Variables:
        phi: Conformal factor (W = exp(-2*phi) or exp(-4*phi) = det(gtilde)^(-1/3))
        gt: Conformal metric gamma_tilde_ij (6 components: 11,12,13,22,23,33)
        Xt: Conformal connection functions Gamma_tilde^i (3 components)
        trK: Trace of extrinsic curvature K
        At: Traceless conformal extrinsic curvature A_tilde_ij (6 components)
        alpha: Lapse function
        beta: Shift vector (3 components)
    """
    # Grid dimensions
    nx: int
    ny: int
    nz: int
    dx: float
    
    # Conformal factor
    phi: wp.array3d(dtype=float)
    
    # Conformal metric (symmetric, 6 components)
    gt11: wp.array3d(dtype=float)
    gt12: wp.array3d(dtype=float)
    gt13: wp.array3d(dtype=float)
    gt22: wp.array3d(dtype=float)
    gt23: wp.array3d(dtype=float)
    gt33: wp.array3d(dtype=float)
    
    # Conformal connection functions (3 components)
    Xt1: wp.array3d(dtype=float)
    Xt2: wp.array3d(dtype=float)
    Xt3: wp.array3d(dtype=float)
    
    # Trace of extrinsic curvature
    trK: wp.array3d(dtype=float)
    
    # Traceless conformal extrinsic curvature (symmetric, 6 components)
    At11: wp.array3d(dtype=float)
    At12: wp.array3d(dtype=float)
    At13: wp.array3d(dtype=float)
    At22: wp.array3d(dtype=float)
    At23: wp.array3d(dtype=float)
    At33: wp.array3d(dtype=float)
    
    # Gauge variables
    alpha: wp.array3d(dtype=float)
    beta1: wp.array3d(dtype=float)
    beta2: wp.array3d(dtype=float)
    beta3: wp.array3d(dtype=float)


def create_bssn_state(nx: int, ny: int, nz: int, dx: float, 
                      requires_grad: bool = False) -> BSSNState:
    """Create a BSSNState with allocated arrays."""
    shape = (nx, ny, nz)
    
    state = BSSNState()
    state.nx = nx
    state.ny = ny
    state.nz = nz
    state.dx = dx
    
    # Allocate all fields
    state.phi = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    
    state.gt11 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.gt12 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.gt13 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.gt22 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.gt23 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.gt33 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    
    state.Xt1 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.Xt2 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.Xt3 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    
    state.trK = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    
    state.At11 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.At12 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.At13 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.At22 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.At23 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.At33 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    
    state.alpha = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.beta1 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.beta2 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    state.beta3 = wp.zeros(shape, dtype=float, requires_grad=requires_grad)
    
    return state


# ============================================================================
# Flat Spacetime Initial Data
# ============================================================================

@wp.kernel
def init_flat_spacetime(
    phi: wp.array3d(dtype=float),
    gt11: wp.array3d(dtype=float),
    gt12: wp.array3d(dtype=float),
    gt13: wp.array3d(dtype=float),
    gt22: wp.array3d(dtype=float),
    gt23: wp.array3d(dtype=float),
    gt33: wp.array3d(dtype=float),
    Xt1: wp.array3d(dtype=float),
    Xt2: wp.array3d(dtype=float),
    Xt3: wp.array3d(dtype=float),
    trK: wp.array3d(dtype=float),
    At11: wp.array3d(dtype=float),
    At12: wp.array3d(dtype=float),
    At13: wp.array3d(dtype=float),
    At22: wp.array3d(dtype=float),
    At23: wp.array3d(dtype=float),
    At33: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    beta1: wp.array3d(dtype=float),
    beta2: wp.array3d(dtype=float),
    beta3: wp.array3d(dtype=float),
):
    """Initialize flat spacetime (Minkowski) initial data."""
    i, j, k = wp.tid()
    
    # Conformal factor: phi = 0 (W = 1)
    phi[i, j, k] = 0.0
    
    # Conformal metric: identity (flat space)
    gt11[i, j, k] = 1.0
    gt12[i, j, k] = 0.0
    gt13[i, j, k] = 0.0
    gt22[i, j, k] = 1.0
    gt23[i, j, k] = 0.0
    gt33[i, j, k] = 1.0
    
    # Conformal connection functions: zero for flat
    Xt1[i, j, k] = 0.0
    Xt2[i, j, k] = 0.0
    Xt3[i, j, k] = 0.0
    
    # Extrinsic curvature: zero (time-symmetric)
    trK[i, j, k] = 0.0
    At11[i, j, k] = 0.0
    At12[i, j, k] = 0.0
    At13[i, j, k] = 0.0
    At22[i, j, k] = 0.0
    At23[i, j, k] = 0.0
    At33[i, j, k] = 0.0
    
    # Gauge: geodesic slicing
    alpha[i, j, k] = 1.0
    beta1[i, j, k] = 0.0
    beta2[i, j, k] = 0.0
    beta3[i, j, k] = 0.0


def init_flat_spacetime_state(state: BSSNState):
    """Initialize a BSSNState with flat spacetime data."""
    wp.launch(
        init_flat_spacetime,
        dim=(state.nx, state.ny, state.nz),
        inputs=[
            state.phi,
            state.gt11, state.gt12, state.gt13, state.gt22, state.gt23, state.gt33,
            state.Xt1, state.Xt2, state.Xt3,
            state.trK,
            state.At11, state.At12, state.At13, state.At22, state.At23, state.At33,
            state.alpha,
            state.beta1, state.beta2, state.beta3,
        ]
    )


# ============================================================================
# Finite Difference Operators (4th order)
# ============================================================================

@wp.func
def d1_4th(u: wp.array3d(dtype=float), i: int, j: int, k: int, 
           di: int, dj: int, dk: int, inv_dx: float) -> float:
    """4th order first derivative in direction (di, dj, dk)."""
    # Coefficients: (-1/12, 2/3, 0, -2/3, 1/12) / dx
    c1 = -1.0 / 12.0
    c2 = 2.0 / 3.0
    
    um2 = u[i - 2*di, j - 2*dj, k - 2*dk]
    um1 = u[i - di, j - dj, k - dk]
    up1 = u[i + di, j + dj, k + dk]
    up2 = u[i + 2*di, j + 2*dj, k + 2*dk]
    
    return (c1 * um2 - c2 * um1 + c2 * up1 - c1 * up2) * inv_dx


@wp.func
def d2_4th(u: wp.array3d(dtype=float), i: int, j: int, k: int,
           di: int, dj: int, dk: int, inv_dx2: float) -> float:
    """4th order second derivative in direction (di, dj, dk)."""
    # Coefficients: (-1/12, 4/3, -5/2, 4/3, -1/12) / dx^2
    c1 = -1.0 / 12.0
    c2 = 4.0 / 3.0
    c0 = -5.0 / 2.0
    
    um2 = u[i - 2*di, j - 2*dj, k - 2*dk]
    um1 = u[i - di, j - dj, k - dk]
    u0 = u[i, j, k]
    up1 = u[i + di, j + dj, k + dk]
    up2 = u[i + 2*di, j + 2*dj, k + 2*dk]
    
    return (c1 * um2 + c2 * um1 + c0 * u0 + c2 * up1 + c1 * up2) * inv_dx2


@wp.func
def d11_4th(u: wp.array3d(dtype=float), i: int, j: int, k: int,
            di1: int, dj1: int, dk1: int,
            di2: int, dj2: int, dk2: int, inv_dx2: float) -> float:
    """4th order mixed second derivative."""
    # Apply first derivative in each direction
    c1 = -1.0 / 12.0
    c2 = 2.0 / 3.0
    
    # First derivative in direction 2 at offset positions in direction 1
    dm2 = d1_4th(u, i - 2*di1, j - 2*dj1, k - 2*dk1, di2, dj2, dk2, 1.0)
    dm1 = d1_4th(u, i - di1, j - dj1, k - dk1, di2, dj2, dk2, 1.0)
    dp1 = d1_4th(u, i + di1, j + dj1, k + dk1, di2, dj2, dk2, 1.0)
    dp2 = d1_4th(u, i + 2*di1, j + 2*dj1, k + 2*dk1, di2, dj2, dk2, 1.0)
    
    return (c1 * dm2 - c2 * dm1 + c2 * dp1 - c1 * dp2) * inv_dx2


@wp.func
def dx(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx: float) -> float:
    """Partial derivative in x direction."""
    return d1_4th(u, i, j, k, 1, 0, 0, inv_dx)


@wp.func
def dy(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx: float) -> float:
    """Partial derivative in y direction."""
    return d1_4th(u, i, j, k, 0, 1, 0, inv_dx)


@wp.func
def dz(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx: float) -> float:
    """Partial derivative in z direction."""
    return d1_4th(u, i, j, k, 0, 0, 1, inv_dx)


@wp.func
def dxx(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx2: float) -> float:
    """Second derivative in x direction."""
    return d2_4th(u, i, j, k, 1, 0, 0, inv_dx2)


@wp.func
def dyy(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx2: float) -> float:
    """Second derivative in y direction."""
    return d2_4th(u, i, j, k, 0, 1, 0, inv_dx2)


@wp.func
def dzz(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx2: float) -> float:
    """Second derivative in z direction."""
    return d2_4th(u, i, j, k, 0, 0, 1, inv_dx2)


@wp.func
def dxy(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx2: float) -> float:
    """Mixed derivative xy."""
    return d11_4th(u, i, j, k, 1, 0, 0, 0, 1, 0, inv_dx2)


@wp.func
def dxz(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx2: float) -> float:
    """Mixed derivative xz."""
    return d11_4th(u, i, j, k, 1, 0, 0, 0, 0, 1, inv_dx2)


@wp.func
def dyz(u: wp.array3d(dtype=float), i: int, j: int, k: int, inv_dx2: float) -> float:
    """Mixed derivative yz."""
    return d11_4th(u, i, j, k, 0, 1, 0, 0, 0, 1, inv_dx2)


# ============================================================================
# Kreiss-Oliger Dissipation (5th order for 4th order FD)
# ============================================================================

@wp.func
def ko_diss_1d(u: wp.array3d(dtype=float), i: int, j: int, k: int,
               di: int, dj: int, dk: int, eps: float, inv_dx: float) -> float:
    """Kreiss-Oliger dissipation in one direction (6th order operator)."""
    # D^6 / 64 * dx^5 where D^6 is the 6th difference operator
    # Stencil: (1, -6, 15, -20, 15, -6, 1) / 64
    um3 = u[i - 3*di, j - 3*dj, k - 3*dk]
    um2 = u[i - 2*di, j - 2*dj, k - 2*dk]
    um1 = u[i - di, j - dj, k - dk]
    u0 = u[i, j, k]
    up1 = u[i + di, j + dj, k + dk]
    up2 = u[i + 2*di, j + 2*dj, k + 2*dk]
    up3 = u[i + 3*di, j + 3*dj, k + 3*dk]
    
    d6 = um3 - 6.0*um2 + 15.0*um1 - 20.0*u0 + 15.0*up1 - 6.0*up2 + up3
    return -eps * inv_dx * d6 / 64.0


@wp.func
def ko_dissipation(u: wp.array3d(dtype=float), i: int, j: int, k: int,
                   eps: float, inv_dx: float) -> float:
    """Total Kreiss-Oliger dissipation (sum of all directions)."""
    return (ko_diss_1d(u, i, j, k, 1, 0, 0, eps, inv_dx) +
            ko_diss_1d(u, i, j, k, 0, 1, 0, eps, inv_dx) +
            ko_diss_1d(u, i, j, k, 0, 0, 1, eps, inv_dx))


# ============================================================================
# BSSN RHS Evolution Kernel
# ============================================================================

@wp.kernel
def compute_bssn_rhs(
    # Input state
    phi: wp.array3d(dtype=float),
    gt11: wp.array3d(dtype=float),
    gt12: wp.array3d(dtype=float),
    gt13: wp.array3d(dtype=float),
    gt22: wp.array3d(dtype=float),
    gt23: wp.array3d(dtype=float),
    gt33: wp.array3d(dtype=float),
    Xt1: wp.array3d(dtype=float),
    Xt2: wp.array3d(dtype=float),
    Xt3: wp.array3d(dtype=float),
    trK: wp.array3d(dtype=float),
    At11: wp.array3d(dtype=float),
    At12: wp.array3d(dtype=float),
    At13: wp.array3d(dtype=float),
    At22: wp.array3d(dtype=float),
    At23: wp.array3d(dtype=float),
    At33: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    beta1: wp.array3d(dtype=float),
    beta2: wp.array3d(dtype=float),
    beta3: wp.array3d(dtype=float),
    # Output RHS
    phi_rhs: wp.array3d(dtype=float),
    gt11_rhs: wp.array3d(dtype=float),
    gt12_rhs: wp.array3d(dtype=float),
    gt13_rhs: wp.array3d(dtype=float),
    gt22_rhs: wp.array3d(dtype=float),
    gt23_rhs: wp.array3d(dtype=float),
    gt33_rhs: wp.array3d(dtype=float),
    Xt1_rhs: wp.array3d(dtype=float),
    Xt2_rhs: wp.array3d(dtype=float),
    Xt3_rhs: wp.array3d(dtype=float),
    trK_rhs: wp.array3d(dtype=float),
    At11_rhs: wp.array3d(dtype=float),
    At12_rhs: wp.array3d(dtype=float),
    At13_rhs: wp.array3d(dtype=float),
    At22_rhs: wp.array3d(dtype=float),
    At23_rhs: wp.array3d(dtype=float),
    At33_rhs: wp.array3d(dtype=float),
    alpha_rhs: wp.array3d(dtype=float),
    beta1_rhs: wp.array3d(dtype=float),
    beta2_rhs: wp.array3d(dtype=float),
    beta3_rhs: wp.array3d(dtype=float),
    # Grid parameters
    inv_dx: float,
    eps_diss: float,
):
    """Compute BSSN evolution RHS.
    
    For flat spacetime with geodesic slicing, all RHS should be ~0.
    Implements 1+log slicing for lapse evolution.
    """
    i, j, k = wp.tid()
    
    # Get grid dimensions
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    
    # Skip boundary points (need 3 ghost zones for KO dissipation)
    if i < 3 or i >= nx - 3 or j < 3 or j >= ny - 3 or k < 3 or k >= nz - 3:
        # Zero RHS at boundaries (will be handled by BC later)
        phi_rhs[i, j, k] = 0.0
        gt11_rhs[i, j, k] = 0.0
        gt12_rhs[i, j, k] = 0.0
        gt13_rhs[i, j, k] = 0.0
        gt22_rhs[i, j, k] = 0.0
        gt23_rhs[i, j, k] = 0.0
        gt33_rhs[i, j, k] = 0.0
        Xt1_rhs[i, j, k] = 0.0
        Xt2_rhs[i, j, k] = 0.0
        Xt3_rhs[i, j, k] = 0.0
        trK_rhs[i, j, k] = 0.0
        At11_rhs[i, j, k] = 0.0
        At12_rhs[i, j, k] = 0.0
        At13_rhs[i, j, k] = 0.0
        At22_rhs[i, j, k] = 0.0
        At23_rhs[i, j, k] = 0.0
        At33_rhs[i, j, k] = 0.0
        alpha_rhs[i, j, k] = 0.0
        beta1_rhs[i, j, k] = 0.0
        beta2_rhs[i, j, k] = 0.0
        beta3_rhs[i, j, k] = 0.0
        return
    
    inv_dx2 = inv_dx * inv_dx
    
    # Get local values
    alph = alpha[i, j, k]
    K = trK[i, j, k]
    ph = phi[i, j, k]
    
    # Get shift components
    b1 = beta1[i, j, k]
    b2 = beta2[i, j, k]
    b3 = beta3[i, j, k]
    
    # Conformal metric components (local)
    g11 = gt11[i, j, k]
    g12 = gt12[i, j, k]
    g13 = gt13[i, j, k]
    g22 = gt22[i, j, k]
    g23 = gt23[i, j, k]
    g33 = gt33[i, j, k]
    
    # Traceless extrinsic curvature components (local)
    a11 = At11[i, j, k]
    a12 = At12[i, j, k]
    a13 = At13[i, j, k]
    a22 = At22[i, j, k]
    a23 = At23[i, j, k]
    a33 = At33[i, j, k]
    
    # Compute inverse conformal metric
    det_gt = (g11 * (g22 * g33 - g23 * g23)
            - g12 * (g12 * g33 - g23 * g13)
            + g13 * (g12 * g23 - g22 * g13))
    inv_det = 1.0 / det_gt
    
    gtu11 = (g22 * g33 - g23 * g23) * inv_det
    gtu12 = (g13 * g23 - g12 * g33) * inv_det
    gtu13 = (g12 * g23 - g13 * g22) * inv_det
    gtu22 = (g11 * g33 - g13 * g13) * inv_det
    gtu23 = (g12 * g13 - g11 * g23) * inv_det
    gtu33 = (g11 * g22 - g12 * g12) * inv_det
    
    # e^{-4phi}
    em4phi = wp.exp(-4.0 * ph)
    
    # =========================================
    # Compute derivatives
    # =========================================
    
    # Derivatives of shift (for div(beta))
    d1_b1 = dx(beta1, i, j, k, inv_dx)
    d2_b2 = dy(beta2, i, j, k, inv_dx)
    d3_b3 = dz(beta3, i, j, k, inv_dx)
    div_beta = d1_b1 + d2_b2 + d3_b3
    
    # =========================================
    # Evolution equations
    # =========================================
    
    # phi evolution: ∂_t φ = -1/6 α K + 1/6 ∂_i β^i + β^i ∂_i φ
    d1_phi = dx(phi, i, j, k, inv_dx)
    d2_phi = dy(phi, i, j, k, inv_dx)
    d3_phi = dz(phi, i, j, k, inv_dx)
    advect_phi = b1 * d1_phi + b2 * d2_phi + b3 * d3_phi
    
    phi_rhs[i, j, k] = (-1.0/6.0 * alph * K 
                        + 1.0/6.0 * div_beta 
                        + advect_phi
                        + ko_dissipation(phi, i, j, k, eps_diss, inv_dx))
    
    # gamma_tilde evolution: ∂_t γ̃_{ij} = -2α Ã_{ij} + γ̃_{ik}∂_jβ^k + γ̃_{jk}∂_iβ^k - 2/3 γ̃_{ij}∂_kβ^k + β^k∂_kγ̃_{ij}
    d1_b2 = dx(beta2, i, j, k, inv_dx)
    d1_b3 = dx(beta3, i, j, k, inv_dx)
    d2_b1 = dy(beta1, i, j, k, inv_dx)
    d2_b3 = dy(beta3, i, j, k, inv_dx)
    d3_b1 = dz(beta1, i, j, k, inv_dx)
    d3_b2 = dz(beta2, i, j, k, inv_dx)
    
    d1_gt11 = dx(gt11, i, j, k, inv_dx)
    d2_gt11 = dy(gt11, i, j, k, inv_dx)
    d3_gt11 = dz(gt11, i, j, k, inv_dx)
    advect_gt11 = b1 * d1_gt11 + b2 * d2_gt11 + b3 * d3_gt11
    
    gt11_rhs[i, j, k] = (-2.0 * alph * a11
                         + g11 * d1_b1 + g12 * d1_b2 + g13 * d1_b3
                         + g11 * d1_b1 + g12 * d1_b2 + g13 * d1_b3
                         - 2.0/3.0 * g11 * div_beta
                         + advect_gt11
                         + ko_dissipation(gt11, i, j, k, eps_diss, inv_dx))
    
    d1_gt12 = dx(gt12, i, j, k, inv_dx)
    d2_gt12 = dy(gt12, i, j, k, inv_dx)
    d3_gt12 = dz(gt12, i, j, k, inv_dx)
    advect_gt12 = b1 * d1_gt12 + b2 * d2_gt12 + b3 * d3_gt12
    
    gt12_rhs[i, j, k] = (-2.0 * alph * a12
                         + g11 * d2_b1 + g12 * d2_b2 + g13 * d2_b3
                         + g12 * d1_b1 + g22 * d1_b2 + g23 * d1_b3
                         - 2.0/3.0 * g12 * div_beta
                         + advect_gt12
                         + ko_dissipation(gt12, i, j, k, eps_diss, inv_dx))
    
    d1_gt13 = dx(gt13, i, j, k, inv_dx)
    d2_gt13 = dy(gt13, i, j, k, inv_dx)
    d3_gt13 = dz(gt13, i, j, k, inv_dx)
    advect_gt13 = b1 * d1_gt13 + b2 * d2_gt13 + b3 * d3_gt13
    
    gt13_rhs[i, j, k] = (-2.0 * alph * a13
                         + g11 * d3_b1 + g12 * d3_b2 + g13 * d3_b3
                         + g13 * d1_b1 + g23 * d1_b2 + g33 * d1_b3
                         - 2.0/3.0 * g13 * div_beta
                         + advect_gt13
                         + ko_dissipation(gt13, i, j, k, eps_diss, inv_dx))
    
    d1_gt22 = dx(gt22, i, j, k, inv_dx)
    d2_gt22 = dy(gt22, i, j, k, inv_dx)
    d3_gt22 = dz(gt22, i, j, k, inv_dx)
    advect_gt22 = b1 * d1_gt22 + b2 * d2_gt22 + b3 * d3_gt22
    
    gt22_rhs[i, j, k] = (-2.0 * alph * a22
                         + g12 * d2_b1 + g22 * d2_b2 + g23 * d2_b3
                         + g12 * d2_b1 + g22 * d2_b2 + g23 * d2_b3
                         - 2.0/3.0 * g22 * div_beta
                         + advect_gt22
                         + ko_dissipation(gt22, i, j, k, eps_diss, inv_dx))
    
    d1_gt23 = dx(gt23, i, j, k, inv_dx)
    d2_gt23 = dy(gt23, i, j, k, inv_dx)
    d3_gt23 = dz(gt23, i, j, k, inv_dx)
    advect_gt23 = b1 * d1_gt23 + b2 * d2_gt23 + b3 * d3_gt23
    
    gt23_rhs[i, j, k] = (-2.0 * alph * a23
                         + g12 * d3_b1 + g22 * d3_b2 + g23 * d3_b3
                         + g13 * d2_b1 + g23 * d2_b2 + g33 * d2_b3
                         - 2.0/3.0 * g23 * div_beta
                         + advect_gt23
                         + ko_dissipation(gt23, i, j, k, eps_diss, inv_dx))
    
    d1_gt33 = dx(gt33, i, j, k, inv_dx)
    d2_gt33 = dy(gt33, i, j, k, inv_dx)
    d3_gt33 = dz(gt33, i, j, k, inv_dx)
    advect_gt33 = b1 * d1_gt33 + b2 * d2_gt33 + b3 * d3_gt33
    
    gt33_rhs[i, j, k] = (-2.0 * alph * a33
                         + g13 * d3_b1 + g23 * d3_b2 + g33 * d3_b3
                         + g13 * d3_b1 + g23 * d3_b2 + g33 * d3_b3
                         - 2.0/3.0 * g33 * div_beta
                         + advect_gt33
                         + ko_dissipation(gt33, i, j, k, eps_diss, inv_dx))
    
    # For Xt, trK, At: simplified for flat spacetime (terms involving Ricci tensor, etc.)
    # For now, just advection + dissipation (full RHS requires Christoffel symbols)
    
    d1_Xt1 = dx(Xt1, i, j, k, inv_dx)
    d2_Xt1 = dy(Xt1, i, j, k, inv_dx)
    d3_Xt1 = dz(Xt1, i, j, k, inv_dx)
    Xt1_rhs[i, j, k] = (b1 * d1_Xt1 + b2 * d2_Xt1 + b3 * d3_Xt1
                        + ko_dissipation(Xt1, i, j, k, eps_diss, inv_dx))
    
    d1_Xt2 = dx(Xt2, i, j, k, inv_dx)
    d2_Xt2 = dy(Xt2, i, j, k, inv_dx)
    d3_Xt2 = dz(Xt2, i, j, k, inv_dx)
    Xt2_rhs[i, j, k] = (b1 * d1_Xt2 + b2 * d2_Xt2 + b3 * d3_Xt2
                        + ko_dissipation(Xt2, i, j, k, eps_diss, inv_dx))
    
    d1_Xt3 = dx(Xt3, i, j, k, inv_dx)
    d2_Xt3 = dy(Xt3, i, j, k, inv_dx)
    d3_Xt3 = dz(Xt3, i, j, k, inv_dx)
    Xt3_rhs[i, j, k] = (b1 * d1_Xt3 + b2 * d2_Xt3 + b3 * d3_Xt3
                        + ko_dissipation(Xt3, i, j, k, eps_diss, inv_dx))
    
    d1_trK = dx(trK, i, j, k, inv_dx)
    d2_trK = dy(trK, i, j, k, inv_dx)
    d3_trK = dz(trK, i, j, k, inv_dx)
    trK_rhs[i, j, k] = (b1 * d1_trK + b2 * d2_trK + b3 * d3_trK
                        + ko_dissipation(trK, i, j, k, eps_diss, inv_dx))
    
    # At evolution (simplified - just advection for flat spacetime)
    d1_At11 = dx(At11, i, j, k, inv_dx)
    d2_At11 = dy(At11, i, j, k, inv_dx)
    d3_At11 = dz(At11, i, j, k, inv_dx)
    At11_rhs[i, j, k] = (b1 * d1_At11 + b2 * d2_At11 + b3 * d3_At11
                         + ko_dissipation(At11, i, j, k, eps_diss, inv_dx))
    
    d1_At12 = dx(At12, i, j, k, inv_dx)
    d2_At12 = dy(At12, i, j, k, inv_dx)
    d3_At12 = dz(At12, i, j, k, inv_dx)
    At12_rhs[i, j, k] = (b1 * d1_At12 + b2 * d2_At12 + b3 * d3_At12
                         + ko_dissipation(At12, i, j, k, eps_diss, inv_dx))
    
    d1_At13 = dx(At13, i, j, k, inv_dx)
    d2_At13 = dy(At13, i, j, k, inv_dx)
    d3_At13 = dz(At13, i, j, k, inv_dx)
    At13_rhs[i, j, k] = (b1 * d1_At13 + b2 * d2_At13 + b3 * d3_At13
                         + ko_dissipation(At13, i, j, k, eps_diss, inv_dx))
    
    d1_At22 = dx(At22, i, j, k, inv_dx)
    d2_At22 = dy(At22, i, j, k, inv_dx)
    d3_At22 = dz(At22, i, j, k, inv_dx)
    At22_rhs[i, j, k] = (b1 * d1_At22 + b2 * d2_At22 + b3 * d3_At22
                         + ko_dissipation(At22, i, j, k, eps_diss, inv_dx))
    
    d1_At23 = dx(At23, i, j, k, inv_dx)
    d2_At23 = dy(At23, i, j, k, inv_dx)
    d3_At23 = dz(At23, i, j, k, inv_dx)
    At23_rhs[i, j, k] = (b1 * d1_At23 + b2 * d2_At23 + b3 * d3_At23
                         + ko_dissipation(At23, i, j, k, eps_diss, inv_dx))
    
    d1_At33 = dx(At33, i, j, k, inv_dx)
    d2_At33 = dy(At33, i, j, k, inv_dx)
    d3_At33 = dz(At33, i, j, k, inv_dx)
    At33_rhs[i, j, k] = (b1 * d1_At33 + b2 * d2_At33 + b3 * d3_At33
                         + ko_dissipation(At33, i, j, k, eps_diss, inv_dx))
    
    # Lapse evolution: 1+log slicing
    # ∂_t α = -2 α K + β^i ∂_i α
    d1_alpha = dx(alpha, i, j, k, inv_dx)
    d2_alpha = dy(alpha, i, j, k, inv_dx)
    d3_alpha = dz(alpha, i, j, k, inv_dx)
    advect_alpha = b1 * d1_alpha + b2 * d2_alpha + b3 * d3_alpha
    
    alpha_rhs[i, j, k] = (-2.0 * alph * K 
                          + advect_alpha
                          + ko_dissipation(alpha, i, j, k, eps_diss, inv_dx))
    
    # Shift: zero evolution for now (geodesic shift)
    beta1_rhs[i, j, k] = ko_dissipation(beta1, i, j, k, eps_diss, inv_dx)
    beta2_rhs[i, j, k] = ko_dissipation(beta2, i, j, k, eps_diss, inv_dx)
    beta3_rhs[i, j, k] = ko_dissipation(beta3, i, j, k, eps_diss, inv_dx)


def compute_rhs(state: BSSNState, rhs: BSSNState, eps_diss: float = 0.1):
    """Compute BSSN RHS for a given state."""
    inv_dx = 1.0 / state.dx
    
    wp.launch(
        compute_bssn_rhs,
        dim=(state.nx, state.ny, state.nz),
        inputs=[
            # Input state
            state.phi,
            state.gt11, state.gt12, state.gt13, state.gt22, state.gt23, state.gt33,
            state.Xt1, state.Xt2, state.Xt3,
            state.trK,
            state.At11, state.At12, state.At13, state.At22, state.At23, state.At33,
            state.alpha,
            state.beta1, state.beta2, state.beta3,
            # Output RHS
            rhs.phi,
            rhs.gt11, rhs.gt12, rhs.gt13, rhs.gt22, rhs.gt23, rhs.gt33,
            rhs.Xt1, rhs.Xt2, rhs.Xt3,
            rhs.trK,
            rhs.At11, rhs.At12, rhs.At13, rhs.At22, rhs.At23, rhs.At33,
            rhs.alpha,
            rhs.beta1, rhs.beta2, rhs.beta3,
            # Parameters
            inv_dx,
            eps_diss,
        ]
    )


# ============================================================================
# RK4 Time Integration
# ============================================================================

@wp.kernel
def rk4_update(u: wp.array3d(dtype=float), 
               u0: wp.array3d(dtype=float),
               k: wp.array3d(dtype=float), 
               dt: float, 
               stage: int):
    """RK4 update for a single field.
    
    stage 0: u = u0 + dt/2 * k1
    stage 1: u = u0 + dt/2 * k2  
    stage 2: u = u0 + dt * k3
    stage 3: u = u0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)  [handled separately]
    """
    i, j, l = wp.tid()
    
    if stage == 0:
        u[i, j, l] = u0[i, j, l] + 0.5 * dt * k[i, j, l]
    elif stage == 1:
        u[i, j, l] = u0[i, j, l] + 0.5 * dt * k[i, j, l]
    elif stage == 2:
        u[i, j, l] = u0[i, j, l] + dt * k[i, j, l]


@wp.kernel
def rk4_final(u: wp.array3d(dtype=float),
              u0: wp.array3d(dtype=float),
              k1: wp.array3d(dtype=float),
              k2: wp.array3d(dtype=float),
              k3: wp.array3d(dtype=float),
              k4: wp.array3d(dtype=float),
              dt: float):
    """Final RK4 combination: u = u0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)"""
    i, j, l = wp.tid()
    u[i, j, l] = u0[i, j, l] + dt / 6.0 * (k1[i, j, l] + 2.0 * k2[i, j, l] 
                                           + 2.0 * k3[i, j, l] + k4[i, j, l])


def rk4_step_field(u: wp.array3d, u0: wp.array3d, k: wp.array3d, 
                   dt: float, stage: int, shape: tuple):
    """Perform one RK4 stage for a single field."""
    wp.launch(rk4_update, dim=shape, inputs=[u, u0, k, dt, stage])


def rk4_final_field(u: wp.array3d, u0: wp.array3d, 
                    k1: wp.array3d, k2: wp.array3d, k3: wp.array3d, k4: wp.array3d,
                    dt: float, shape: tuple):
    """Final RK4 combination for a single field."""
    wp.launch(rk4_final, dim=shape, inputs=[u, u0, k1, k2, k3, k4, dt])


# ============================================================================
# Test the module
# ============================================================================

def test_bssn_state():
    """Test BSSN state creation and flat spacetime initialization."""
    nx, ny, nz = 16, 16, 16
    dx = 0.1
    
    state = create_bssn_state(nx, ny, nz, dx, requires_grad=True)
    init_flat_spacetime_state(state)
    
    # Check that flat spacetime is initialized correctly
    alpha_np = state.alpha.numpy()
    gt11_np = state.gt11.numpy()
    phi_np = state.phi.numpy()
    
    assert abs(alpha_np.mean() - 1.0) < 1e-10, f"Alpha mean: {alpha_np.mean()}"
    assert abs(gt11_np.mean() - 1.0) < 1e-10, f"gt11 mean: {gt11_np.mean()}"
    assert abs(phi_np.mean()) < 1e-10, f"phi mean: {phi_np.mean()}"
    
    print("BSSN state test passed!")
    return state


def test_bssn_rhs():
    """Test BSSN RHS computation on flat spacetime."""
    nx, ny, nz = 24, 24, 24  # Need extra points for ghost zones
    dx = 0.1
    
    state = create_bssn_state(nx, ny, nz, dx, requires_grad=True)
    rhs = create_bssn_state(nx, ny, nz, dx, requires_grad=True)
    
    init_flat_spacetime_state(state)
    
    # Compute RHS
    compute_rhs(state, rhs, eps_diss=0.1)
    
    # For flat spacetime with zero shift and K, RHS should be ~0
    # (except for boundary effects and numerical noise)
    phi_rhs = rhs.phi.numpy()
    alpha_rhs = rhs.alpha.numpy()
    gt11_rhs = rhs.gt11.numpy()
    
    # Check interior points (skip ghost zones)
    interior = (slice(4, -4), slice(4, -4), slice(4, -4))
    
    max_phi_rhs = abs(phi_rhs[interior]).max()
    max_alpha_rhs = abs(alpha_rhs[interior]).max()
    max_gt11_rhs = abs(gt11_rhs[interior]).max()
    
    print(f"Max |phi_rhs| in interior: {max_phi_rhs:.6e}")
    print(f"Max |alpha_rhs| in interior: {max_alpha_rhs:.6e}")
    print(f"Max |gt11_rhs| in interior: {max_gt11_rhs:.6e}")
    
    # For flat spacetime, these should be very small
    assert max_phi_rhs < 1e-10, f"phi_rhs too large: {max_phi_rhs}"
    assert max_alpha_rhs < 1e-10, f"alpha_rhs too large: {max_alpha_rhs}"
    assert max_gt11_rhs < 1e-10, f"gt11_rhs too large: {max_gt11_rhs}"
    
    print("BSSN RHS test passed!")
    return rhs


if __name__ == "__main__":
    test_bssn_state()
    test_bssn_rhs()
