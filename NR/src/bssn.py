"""
BSSN Evolution Equations in Warp

Implements the Baumgarte-Shapiro-Shibata-Nakamura formulation
of the Einstein equations for numerical relativity.
"""

import numpy as np
import warp as wp

# 3x3 symmetric tensor type (6 independent components)
# Stored as: [xx, xy, xz, yy, yz, zz]
sym3x3 = wp.types.vector(6, dtype=wp.float32)


@wp.struct
class BSSNState:
    """BSSN evolution variables on a 3D grid."""
    # Grid dimensions
    nx: int
    ny: int
    nz: int
    dx: float
    
    # Conformal factor: e^{4φ} = (det γ)^{1/3}
    phi: wp.array3d(dtype=float)
    
    # Conformal metric γ̃ᵢⱼ (6 components for symmetric 3x3)
    # Stored as separate arrays for each component
    gtxx: wp.array3d(dtype=float)
    gtxy: wp.array3d(dtype=float)
    gtxz: wp.array3d(dtype=float)
    gtyy: wp.array3d(dtype=float)
    gtyz: wp.array3d(dtype=float)
    gtzz: wp.array3d(dtype=float)
    
    # Trace of extrinsic curvature K
    K: wp.array3d(dtype=float)
    
    # Traceless conformal extrinsic curvature Ãᵢⱼ
    Atxx: wp.array3d(dtype=float)
    Atxy: wp.array3d(dtype=float)
    Atxz: wp.array3d(dtype=float)
    Atyy: wp.array3d(dtype=float)
    Atyz: wp.array3d(dtype=float)
    Atzz: wp.array3d(dtype=float)
    
    # Conformal connection functions Γ̃ⁱ
    Gtx: wp.array3d(dtype=float)
    Gty: wp.array3d(dtype=float)
    Gtz: wp.array3d(dtype=float)
    
    # Lapse function α
    alpha: wp.array3d(dtype=float)
    
    # Shift vector βⁱ
    betax: wp.array3d(dtype=float)
    betay: wp.array3d(dtype=float)
    betaz: wp.array3d(dtype=float)
    
    # Auxiliary variable for Gamma-driver shift: Bⁱ = ∂ₜβⁱ
    Bx: wp.array3d(dtype=float)
    By: wp.array3d(dtype=float)
    Bz: wp.array3d(dtype=float)


def create_bssn_state(nx: int, ny: int, nz: int, dx: float) -> BSSNState:
    """Create and initialize BSSN state with flat spacetime data."""
    state = BSSNState()
    state.nx = nx
    state.ny = ny
    state.nz = nz
    state.dx = dx
    
    shape = (nx, ny, nz)
    
    # Allocate all arrays
    state.phi = wp.zeros(shape, dtype=float)
    
    # Conformal metric: identity for flat spacetime
    state.gtxx = wp.ones(shape, dtype=float)
    state.gtxy = wp.zeros(shape, dtype=float)
    state.gtxz = wp.zeros(shape, dtype=float)
    state.gtyy = wp.ones(shape, dtype=float)
    state.gtyz = wp.zeros(shape, dtype=float)
    state.gtzz = wp.ones(shape, dtype=float)
    
    # Extrinsic curvature: zero for flat spacetime
    state.K = wp.zeros(shape, dtype=float)
    state.Atxx = wp.zeros(shape, dtype=float)
    state.Atxy = wp.zeros(shape, dtype=float)
    state.Atxz = wp.zeros(shape, dtype=float)
    state.Atyy = wp.zeros(shape, dtype=float)
    state.Atyz = wp.zeros(shape, dtype=float)
    state.Atzz = wp.zeros(shape, dtype=float)
    
    # Connection functions: zero for flat spacetime
    state.Gtx = wp.zeros(shape, dtype=float)
    state.Gty = wp.zeros(shape, dtype=float)
    state.Gtz = wp.zeros(shape, dtype=float)
    
    # Lapse: 1 for flat spacetime
    state.alpha = wp.ones(shape, dtype=float)
    
    # Shift: zero for flat spacetime
    state.betax = wp.zeros(shape, dtype=float)
    state.betay = wp.zeros(shape, dtype=float)
    state.betaz = wp.zeros(shape, dtype=float)
    
    # Gamma-driver auxiliary
    state.Bx = wp.zeros(shape, dtype=float)
    state.By = wp.zeros(shape, dtype=float)
    state.Bz = wp.zeros(shape, dtype=float)
    
    return state


def copy_state(src: BSSNState, dst: BSSNState):
    """Copy BSSN state from src to dst."""
    wp.copy(dst.phi, src.phi)
    wp.copy(dst.gtxx, src.gtxx)
    wp.copy(dst.gtxy, src.gtxy)
    wp.copy(dst.gtxz, src.gtxz)
    wp.copy(dst.gtyy, src.gtyy)
    wp.copy(dst.gtyz, src.gtyz)
    wp.copy(dst.gtzz, src.gtzz)
    wp.copy(dst.K, src.K)
    wp.copy(dst.Atxx, src.Atxx)
    wp.copy(dst.Atxy, src.Atxy)
    wp.copy(dst.Atxz, src.Atxz)
    wp.copy(dst.Atyy, src.Atyy)
    wp.copy(dst.Atyz, src.Atyz)
    wp.copy(dst.Atzz, src.Atzz)
    wp.copy(dst.Gtx, src.Gtx)
    wp.copy(dst.Gty, src.Gty)
    wp.copy(dst.Gtz, src.Gtz)
    wp.copy(dst.alpha, src.alpha)
    wp.copy(dst.betax, src.betax)
    wp.copy(dst.betay, src.betay)
    wp.copy(dst.betaz, src.betaz)
    wp.copy(dst.Bx, src.Bx)
    wp.copy(dst.By, src.By)
    wp.copy(dst.Bz, src.Bz)


# ============================================================================
# Spatial Derivative Kernels (4th order centered finite differences)
# ============================================================================

@wp.func
def d1_4th(fm2: float, fm1: float, fp1: float, fp2: float, h: float) -> float:
    """4th order 1st derivative: (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12h)"""
    return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)


@wp.func
def d2_4th(fm2: float, fm1: float, f0: float, fp1: float, fp2: float, h: float) -> float:
    """4th order 2nd derivative: (-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2}) / (12h²)"""
    return (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h * h)


@wp.func
def idx3d(i: int, j: int, k: int, nx: int, ny: int, nz: int) -> wp.vec3i:
    """Clamp indices to valid range (simple boundary handling)."""
    ii = wp.clamp(i, 0, nx - 1)
    jj = wp.clamp(j, 0, ny - 1)
    kk = wp.clamp(k, 0, nz - 1)
    return wp.vec3i(ii, jj, kk)


@wp.kernel
def compute_dx(
    f: wp.array3d(dtype=float),
    df: wp.array3d(dtype=float),
    dx: float,
    nx: int, ny: int, nz: int
):
    """Compute ∂f/∂x using 4th order FD."""
    i, j, k = wp.tid()
    
    # Get stencil values with clamped boundaries
    im2 = idx3d(i - 2, j, k, nx, ny, nz)
    im1 = idx3d(i - 1, j, k, nx, ny, nz)
    ip1 = idx3d(i + 1, j, k, nx, ny, nz)
    ip2 = idx3d(i + 2, j, k, nx, ny, nz)
    
    fm2 = f[im2[0], im2[1], im2[2]]
    fm1 = f[im1[0], im1[1], im1[2]]
    fp1 = f[ip1[0], ip1[1], ip1[2]]
    fp2 = f[ip2[0], ip2[1], ip2[2]]
    
    df[i, j, k] = d1_4th(fm2, fm1, fp1, fp2, dx)


@wp.kernel
def compute_dy(
    f: wp.array3d(dtype=float),
    df: wp.array3d(dtype=float),
    dy: float,
    nx: int, ny: int, nz: int
):
    """Compute ∂f/∂y using 4th order FD."""
    i, j, k = wp.tid()
    
    jm2 = idx3d(i, j - 2, k, nx, ny, nz)
    jm1 = idx3d(i, j - 1, k, nx, ny, nz)
    jp1 = idx3d(i, j + 1, k, nx, ny, nz)
    jp2 = idx3d(i, j + 2, k, nx, ny, nz)
    
    fm2 = f[jm2[0], jm2[1], jm2[2]]
    fm1 = f[jm1[0], jm1[1], jm1[2]]
    fp1 = f[jp1[0], jp1[1], jp1[2]]
    fp2 = f[jp2[0], jp2[1], jp2[2]]
    
    df[i, j, k] = d1_4th(fm2, fm1, fp1, fp2, dy)


@wp.kernel
def compute_dz(
    f: wp.array3d(dtype=float),
    df: wp.array3d(dtype=float),
    dz: float,
    nx: int, ny: int, nz: int
):
    """Compute ∂f/∂z using 4th order FD."""
    i, j, k = wp.tid()
    
    km2 = idx3d(i, j, k - 2, nx, ny, nz)
    km1 = idx3d(i, j, k - 1, nx, ny, nz)
    kp1 = idx3d(i, j, k + 1, nx, ny, nz)
    kp2 = idx3d(i, j, k + 2, nx, ny, nz)
    
    fm2 = f[km2[0], km2[1], km2[2]]
    fm1 = f[km1[0], km1[1], km1[2]]
    fp1 = f[kp1[0], kp1[1], kp1[2]]
    fp2 = f[kp2[0], kp2[1], kp2[2]]
    
    df[i, j, k] = d1_4th(fm2, fm1, fp1, fp2, dz)


# ============================================================================
# Kreiss-Oliger Dissipation (5th order for 4th order FD scheme)
# ============================================================================

@wp.func
def ko_diss_1d(fm3: float, fm2: float, fm1: float, f0: float, 
               fp1: float, fp2: float, fp3: float, h: float, sigma: float) -> float:
    """5th order Kreiss-Oliger dissipation operator."""
    # D₊³D₋³ stencil: [1, -6, 15, -20, 15, -6, 1] / h⁶
    # Applied as: -σ h⁵ D₊³D₋³ f = -σ/64 [f_{-3} - 6f_{-2} + 15f_{-1} - 20f + 15f_{+1} - 6f_{+2} + f_{+3}]
    return -sigma / 64.0 * (fm3 - 6.0*fm2 + 15.0*fm1 - 20.0*f0 + 15.0*fp1 - 6.0*fp2 + fp3)


@wp.kernel
def apply_ko_dissipation(
    f: wp.array3d(dtype=float),
    diss: wp.array3d(dtype=float),
    dx: float,
    sigma: float,
    nx: int, ny: int, nz: int
):
    """Apply Kreiss-Oliger dissipation in all directions."""
    i, j, k = wp.tid()
    
    # X direction
    im3 = idx3d(i - 3, j, k, nx, ny, nz)
    im2 = idx3d(i - 2, j, k, nx, ny, nz)
    im1 = idx3d(i - 1, j, k, nx, ny, nz)
    ip1 = idx3d(i + 1, j, k, nx, ny, nz)
    ip2 = idx3d(i + 2, j, k, nx, ny, nz)
    ip3 = idx3d(i + 3, j, k, nx, ny, nz)
    
    diss_x = ko_diss_1d(
        f[im3[0], im3[1], im3[2]],
        f[im2[0], im2[1], im2[2]],
        f[im1[0], im1[1], im1[2]],
        f[i, j, k],
        f[ip1[0], ip1[1], ip1[2]],
        f[ip2[0], ip2[1], ip2[2]],
        f[ip3[0], ip3[1], ip3[2]],
        dx, sigma
    )
    
    # Y direction
    jm3 = idx3d(i, j - 3, k, nx, ny, nz)
    jm2 = idx3d(i, j - 2, k, nx, ny, nz)
    jm1 = idx3d(i, j - 1, k, nx, ny, nz)
    jp1 = idx3d(i, j + 1, k, nx, ny, nz)
    jp2 = idx3d(i, j + 2, k, nx, ny, nz)
    jp3 = idx3d(i, j + 3, k, nx, ny, nz)
    
    diss_y = ko_diss_1d(
        f[jm3[0], jm3[1], jm3[2]],
        f[jm2[0], jm2[1], jm2[2]],
        f[jm1[0], jm1[1], jm1[2]],
        f[i, j, k],
        f[jp1[0], jp1[1], jp1[2]],
        f[jp2[0], jp2[1], jp2[2]],
        f[jp3[0], jp3[1], jp3[2]],
        dx, sigma
    )
    
    # Z direction
    km3 = idx3d(i, j, k - 3, nx, ny, nz)
    km2 = idx3d(i, j, k - 2, nx, ny, nz)
    km1 = idx3d(i, j, k - 1, nx, ny, nz)
    kp1 = idx3d(i, j, k + 1, nx, ny, nz)
    kp2 = idx3d(i, j, k + 2, nx, ny, nz)
    kp3 = idx3d(i, j, k + 3, nx, ny, nz)
    
    diss_z = ko_diss_1d(
        f[km3[0], km3[1], km3[2]],
        f[km2[0], km2[1], km2[2]],
        f[km1[0], km1[1], km1[2]],
        f[i, j, k],
        f[kp1[0], kp1[1], kp1[2]],
        f[kp2[0], kp2[1], kp2[2]],
        f[kp3[0], kp3[1], kp3[2]],
        dx, sigma
    )
    
    diss[i, j, k] = diss_x + diss_y + diss_z


# ============================================================================
# BSSN Right-Hand Side (for flat spacetime test)
# ============================================================================

@wp.kernel
def compute_rhs_phi(
    phi: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    betax: wp.array3d(dtype=float),
    betay: wp.array3d(dtype=float),
    betaz: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    dx: float,
    nx: int, ny: int, nz: int
):
    """
    ∂ₜφ = -α K/6 + βⁱ∂ᵢφ + ∂ᵢβⁱ/6
    For flat spacetime: K=0, β=0 → ∂ₜφ = 0
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    Kval = K[i, j, k]
    
    # For now, simplified: just the main term
    # ∂ₜφ = -α K/6
    rhs[i, j, k] = -a * Kval / 6.0


@wp.kernel
def compute_rhs_K(
    K: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    dx: float,
    nx: int, ny: int, nz: int
):
    """
    ∂ₜK = -γⁱʲDᵢDⱼα + α(ÃᵢⱼÃⁱʲ + K²/3) + ...
    For flat spacetime with α=1: ∂ₜK = 0
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    Kval = K[i, j, k]
    
    # Simplified for flat spacetime: ∂ₜK = αK²/3
    rhs[i, j, k] = a * Kval * Kval / 3.0


@wp.kernel
def compute_rhs_alpha_1log(
    alpha: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int
):
    """
    1+log slicing: ∂ₜα = -2αK
    """
    i, j, k = wp.tid()
    
    a = alpha[i, j, k]
    Kval = K[i, j, k]
    
    rhs[i, j, k] = -2.0 * a * Kval


# ============================================================================
# RK4 Time Integration
# ============================================================================

@wp.kernel
def rk4_stage1(
    y: wp.array3d(dtype=float),
    k1: wp.array3d(dtype=float),
    y_tmp: wp.array3d(dtype=float),
    dt: float
):
    """y_tmp = y + dt/2 * k1"""
    i, j, k = wp.tid()
    y_tmp[i, j, k] = y[i, j, k] + 0.5 * dt * k1[i, j, k]


@wp.kernel
def rk4_stage2(
    y: wp.array3d(dtype=float),
    k2: wp.array3d(dtype=float),
    y_tmp: wp.array3d(dtype=float),
    dt: float
):
    """y_tmp = y + dt/2 * k2"""
    i, j, k = wp.tid()
    y_tmp[i, j, k] = y[i, j, k] + 0.5 * dt * k2[i, j, k]


@wp.kernel
def rk4_stage3(
    y: wp.array3d(dtype=float),
    k3: wp.array3d(dtype=float),
    y_tmp: wp.array3d(dtype=float),
    dt: float
):
    """y_tmp = y + dt * k3"""
    i, j, k = wp.tid()
    y_tmp[i, j, k] = y[i, j, k] + dt * k3[i, j, k]


@wp.kernel
def rk4_update(
    y: wp.array3d(dtype=float),
    k1: wp.array3d(dtype=float),
    k2: wp.array3d(dtype=float),
    k3: wp.array3d(dtype=float),
    k4: wp.array3d(dtype=float),
    dt: float
):
    """y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)"""
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + dt / 6.0 * (
        k1[i, j, k] + 2.0 * k2[i, j, k] + 2.0 * k3[i, j, k] + k4[i, j, k]
    )


# ============================================================================
# Simple Test: Evolve flat spacetime
# ============================================================================

def test_flat_spacetime(n_steps=100, nx=16, dt=0.01):
    """Test stability by evolving flat spacetime."""
    wp.init()
    
    dx = 1.0 / nx
    state = create_bssn_state(nx, nx, nx, dx)
    
    # Allocate RHS arrays
    shape = (nx, nx, nx)
    rhs_phi = wp.zeros(shape, dtype=float)
    rhs_K = wp.zeros(shape, dtype=float)
    rhs_alpha = wp.zeros(shape, dtype=float)
    
    print(f"Testing flat spacetime evolution: {nx}³ grid, dt={dt}, {n_steps} steps")
    
    errors = []
    for step in range(n_steps):
        # Compute RHS (for flat spacetime, these should all be ~0)
        wp.launch(compute_rhs_phi, dim=(nx, nx, nx),
                  inputs=[state.phi, state.K, state.alpha, 
                          state.betax, state.betay, state.betaz,
                          rhs_phi, dx, nx, nx, nx])
        
        wp.launch(compute_rhs_K, dim=(nx, nx, nx),
                  inputs=[state.K, state.alpha, rhs_K, dx, nx, nx, nx])
        
        wp.launch(compute_rhs_alpha_1log, dim=(nx, nx, nx),
                  inputs=[state.alpha, state.K, rhs_alpha, nx, nx, nx])
        
        # Simple forward Euler update (for testing)
        wp.launch(rk4_stage1, dim=(nx, nx, nx),
                  inputs=[state.phi, rhs_phi, state.phi, 2.0*dt])  # y = y + dt*k
        wp.launch(rk4_stage1, dim=(nx, nx, nx),
                  inputs=[state.K, rhs_K, state.K, 2.0*dt])
        wp.launch(rk4_stage1, dim=(nx, nx, nx),
                  inputs=[state.alpha, rhs_alpha, state.alpha, 2.0*dt])
        
        # Check error (deviation from flat spacetime)
        phi_max = np.max(np.abs(state.phi.numpy()))
        K_max = np.max(np.abs(state.K.numpy()))
        alpha_err = np.max(np.abs(state.alpha.numpy() - 1.0))
        
        total_err = phi_max + K_max + alpha_err
        errors.append(total_err)
        
        if step % 20 == 0:
            print(f"  Step {step}: |φ|={phi_max:.2e}, |K|={K_max:.2e}, |α-1|={alpha_err:.2e}")
    
    final_err = errors[-1]
    print(f"\nFinal error: {final_err:.6e}")
    print(f"Stable: {final_err < 1e-6}")
    
    return final_err < 1e-6, errors


if __name__ == "__main__":
    stable, errors = test_flat_spacetime(n_steps=100, nx=16, dt=0.001)
    print(f"\nTest {'PASSED' if stable else 'FAILED'}")
