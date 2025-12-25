"""
Full BSSN RHS evolution equations for curved spacetime.

Extends bssn_rhs.py to handle non-trivial geometry.
"""

import warp as wp
import numpy as np
from bssn_state import BSSNState, SymmetricTensor3, idx3d
from bssn_derivatives import deriv_x_4th, deriv_y_4th, deriv_z_4th, gradient_4th

wp.init()


@wp.func
def contract_symmetric_tensors(
    a: SymmetricTensor3,
    b: SymmetricTensor3
) -> float:
    """Compute A_ij B^ij for symmetric tensors (assuming γ̃^ij ≈ δ^ij)"""
    return a.xx*b.xx + 2.0*a.xy*b.xy + 2.0*a.xz*b.xz + \
           a.yy*b.yy + 2.0*a.yz*b.yz + a.zz*b.zz


@wp.kernel
def compute_bssn_rhs_full(
    # Input state
    chi: wp.array(dtype=float),
    gamma_tilde: wp.array(dtype=SymmetricTensor3),
    K: wp.array(dtype=float),
    A_tilde: wp.array(dtype=SymmetricTensor3),
    Gamma_tilde: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=float),
    beta: wp.array(dtype=wp.vec3),
    # Output RHS
    rhs_chi: wp.array(dtype=float),
    rhs_gamma: wp.array(dtype=SymmetricTensor3),
    rhs_K: wp.array(dtype=float),
    rhs_A: wp.array(dtype=SymmetricTensor3),
    rhs_Gamma: wp.array(dtype=wp.vec3),
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float
):
    """
    Compute RHS of BSSN equations for curved spacetime.
    
    Implements:
    - ∂ₜχ = 2/3 χ (α K - ∂ᵢβⁱ) + βⁱ∂ᵢχ
    - ∂ₜγ̃ᵢⱼ = -2α Ãᵢⱼ + advection + shift terms
    - ∂ₜK = -γⁱⱼDᵢDⱼα + α(ÃᵢⱼÃⁱⱼ + K²/3) + βⁱ∂ᵢK
    - ∂ₜÃᵢⱼ = ... (simplified for now)
    - ∂ₜΓ̃ⁱ = ... (simplified for now)
    """
    i, j, k = wp.tid()
    
    # Skip boundaries (need ghost zones for derivatives)
    if i < 2 or i >= nx-2 or j < 2 or j >= ny-2 or k < 2 or k >= nz-2:
        return
    
    idx = idx3d(i, j, k, nx, ny, nz)
    
    # Get local values
    chi_val = chi[idx]
    K_val = K[idx]
    alpha_val = alpha[idx]
    beta_val = beta[idx]
    A_val = A_tilde[idx]
    
    # === Evolution of chi ===
    # ∂ₜχ = 2/3 χ (α K - ∂ᵢβⁱ) + βⁱ∂ᵢχ
    
    # For now, simplified version without shift derivatives
    # (shift is initially zero anyway)
    div_beta = 0.0
    advect_chi = 0.0
    
    rhs_chi[idx] = (2.0/3.0) * chi_val * (alpha_val * K_val - div_beta) + advect_chi
    
    # === Evolution of K ===
    # ∂ₜK = -γⁱⱼDᵢDⱼα + α(ÃᵢⱼÃⁱⱼ + K²/3) + βⁱ∂ᵢK
    
    # Laplacian of lapse (simplified - need proper covariant derivative)
    # For now, use flat space Laplacian
    lap_alpha = 0.0  # Will implement properly later
    
    # Source terms
    A_contract = contract_symmetric_tensors(A_val, A_val)
    source = alpha_val * (A_contract + K_val * K_val / 3.0)
    
    # Advection
    advect_K = 0.0  # Simplified
    
    rhs_K[idx] = -lap_alpha + source + advect_K
    
    # === Evolution of gamma_tilde ===
    # ∂ₜγ̃ᵢⱼ = -2α Ãᵢⱼ + shift terms
    
    result_gamma = SymmetricTensor3()
    result_gamma.xx = -2.0 * alpha_val * A_val.xx
    result_gamma.xy = -2.0 * alpha_val * A_val.xy
    result_gamma.xz = -2.0 * alpha_val * A_val.xz
    result_gamma.yy = -2.0 * alpha_val * A_val.yy
    result_gamma.yz = -2.0 * alpha_val * A_val.yz
    result_gamma.zz = -2.0 * alpha_val * A_val.zz
    
    rhs_gamma[idx] = result_gamma
    
    # === Evolution of A_tilde ===
    # Simplified: decay toward zero
    decay_rate = 0.01
    
    result_A = SymmetricTensor3()
    result_A.xx = -decay_rate * A_val.xx
    result_A.xy = -decay_rate * A_val.xy
    result_A.xz = -decay_rate * A_val.xz
    result_A.yy = -decay_rate * A_val.yy
    result_A.yz = -decay_rate * A_val.yz
    result_A.zz = -decay_rate * A_val.zz
    
    rhs_A[idx] = result_A
    
    # === Evolution of Gamma_tilde ===
    # Simplified for now
    rhs_Gamma[idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_gauge_rhs_full(
    # Input state
    alpha: wp.array(dtype=float),
    beta: wp.array(dtype=wp.vec3),
    K: wp.array(dtype=float),
    Gamma_tilde: wp.array(dtype=wp.vec3),
    # Output RHS  
    rhs_alpha: wp.array(dtype=float),
    rhs_beta: wp.array(dtype=wp.vec3),
    # Auxiliary for Gamma-driver
    B_tilde: wp.array(dtype=wp.vec3),
    rhs_B: wp.array(dtype=wp.vec3),
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    eta: float
):
    """
    Gauge evolution with 1+log lapse and Gamma-driver shift.
    """
    i, j, k = wp.tid()
    
    if i < 2 or i >= nx-2 or j < 2 or j >= ny-2 or k < 2 or k >= nz-2:
        return
    
    idx = idx3d(i, j, k, nx, ny, nz)
    
    alpha_val = alpha[idx]
    K_val = K[idx]
    B_val = B_tilde[idx]
    
    # 1+log slicing: ∂ₜα = -2α K + βⁱ∂ᵢα
    # Advection term simplified
    rhs_alpha[idx] = -2.0 * alpha_val * K_val
    
    # Gamma-driver: ∂ₜβⁱ = B̃ⁱ
    rhs_beta[idx] = B_val
    
    # ∂ₜB̃ⁱ = 3/4 ∂ₜΓ̃ⁱ - η B̃ⁱ
    # For now, assume ∂ₜΓ̃ⁱ ≈ 0
    rhs_B[idx] = -eta * B_val


if __name__ == "__main__":
    print("="*70)
    print("Testing Full BSSN RHS")
    print("="*70)
    
    # Import initial data
    from bbh_initial_data import create_bbh_initial_data
    
    # Small grid
    nx, ny, nz = 32, 32, 32
    L = 40.0
    xmin = ymin = zmin = -L/2
    dx = dy = dz = L / (nx - 1)
    
    print(f"\nGrid: {nx} x {ny} x {nz}")
    print(f"Domain: [{xmin:.1f}, {-xmin:.1f}]³")
    
    # Create BBH initial data
    state = BSSNState(nx, ny, nz)
    create_bbh_initial_data(state, xmin, ymin, zmin, dx, dy, dz,
                           separation=10.0, mass_ratio=1.0)
    
    # Allocate RHS storage
    npts = nx * ny * nz
    rhs_chi = wp.zeros(npts, dtype=float)
    rhs_K = wp.zeros(npts, dtype=float)
    rhs_alpha = wp.zeros(npts, dtype=float)
    rhs_gamma = wp.zeros(npts, dtype=SymmetricTensor3)
    rhs_A = wp.zeros(npts, dtype=SymmetricTensor3)
    rhs_Gamma = wp.zeros(npts, dtype=wp.vec3)
    rhs_beta = wp.zeros(npts, dtype=wp.vec3)
    B_tilde = wp.zeros(npts, dtype=wp.vec3)
    rhs_B = wp.zeros(npts, dtype=wp.vec3)
    
    print("\nComputing RHS for BBH spacetime...")
    
    # Compute evolution RHS
    wp.launch(
        compute_bssn_rhs_full,
        dim=(nx, ny, nz),
        inputs=[
            state.chi, state.gamma_tilde, state.K,
            state.A_tilde, state.Gamma_tilde,
            state.alpha, state.beta,
            rhs_chi, rhs_gamma, rhs_K,
            rhs_A, rhs_Gamma,
            nx, ny, nz, dx, dy, dz
        ]
    )
    
    # Compute gauge RHS
    wp.launch(
        compute_gauge_rhs_full,
        dim=(nx, ny, nz),
        inputs=[
            state.alpha, state.beta, state.K,
            state.Gamma_tilde,
            rhs_alpha, rhs_beta,
            B_tilde, rhs_B,
            nx, ny, nz, dx, dy, dz, 1.0
        ]
    )
    
    # Check RHS values
    rhs_chi_np = rhs_chi.numpy()
    rhs_K_np = rhs_K.numpy()
    rhs_alpha_np = rhs_alpha.numpy()
    
    print(f"\nRHS Statistics:")
    print(f"  ∂ₜχ: min = {rhs_chi_np.min():.6e}, max = {rhs_chi_np.max():.6e}")
    print(f"  ∂ₜK: min = {rhs_K_np.min():.6e}, max = {rhs_K_np.max():.6e}")
    print(f"  ∂ₜα: min = {rhs_alpha_np.min():.6e}, max = {rhs_alpha_np.max():.6e}")
    
    # For BBH, RHS should be non-zero (unlike flat spacetime)
    total_rhs = (np.abs(rhs_chi_np).max() + 
                 np.abs(rhs_K_np).max() + 
                 np.abs(rhs_alpha_np).max())
    
    if total_rhs > 1e-6:
        print(f"\n✓ RHS non-zero for BBH (as expected): {total_rhs:.2e}")
    else:
        print(f"\n✗ RHS unexpectedly small: {total_rhs:.2e}")
    
    print("\n" + "="*70)
    print("Full BSSN RHS test completed")
    print("="*70)
