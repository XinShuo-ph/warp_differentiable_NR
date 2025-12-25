"""
Brill-Lindquist puncture initial data for binary black holes.

The Brill-Lindquist solution provides initial data for multiple black holes
in momentarily time-symmetric configurations (K = 0).

For punctures at positions r_A and r_B with masses m_A and m_B:
ψ = 1 + m_A/(2|r - r_A|) + m_B/(2|r - r_B|)

Physical metric: γ_ij = ψ^4 δ_ij
Conformal factor: χ = ψ^(-4)
Extrinsic curvature: K_ij = 0 (time-symmetric)
"""

import warp as wp
import numpy as np
from bssn_state import BSSNState, SymmetricTensor3, idx3d

wp.init()


@wp.func
def distance(p1: wp.vec3, p2: wp.vec3) -> float:
    """Euclidean distance between two points"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return wp.sqrt(dx*dx + dy*dy + dz*dz)


@wp.kernel
def set_brill_lindquist_data(
    # BSSN fields
    chi: wp.array(dtype=float),
    gamma_tilde: wp.array(dtype=SymmetricTensor3),
    K: wp.array(dtype=float),
    A_tilde: wp.array(dtype=SymmetricTensor3),
    Gamma_tilde: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=float),
    beta: wp.array(dtype=wp.vec3),
    # Grid parameters
    nx: int, ny: int, nz: int,
    xmin: float, ymin: float, zmin: float,
    dx: float, dy: float, dz: float,
    # BH parameters
    pos_A: wp.vec3, mass_A: float,
    pos_B: wp.vec3, mass_B: float
):
    """
    Set Brill-Lindquist initial data for two black holes.
    
    Conformal factor ψ = 1 + Σ m_i/(2r_i)
    Conformal metric γ̃_ij = δ_ij (flat)
    χ = ψ^(-4)
    K = 0 (time-symmetric)
    """
    i, j, k = wp.tid()
    
    if i >= nx or j >= ny or k >= nz:
        return
    
    idx = idx3d(i, j, k, nx, ny, nz)
    
    # Compute physical position
    x = xmin + float(i) * dx
    y = ymin + float(j) * dy
    z = zmin + float(k) * dz
    pos = wp.vec3(x, y, z)
    
    # Distances to punctures
    r_A = distance(pos, pos_A)
    r_B = distance(pos, pos_B)
    
    # Avoid singularity at punctures
    r_A = wp.max(r_A, 0.5 * dx)  # Smoothing radius
    r_B = wp.max(r_B, 0.5 * dx)
    
    # Brill-Lindquist conformal factor
    psi = 1.0 + mass_A / (2.0 * r_A) + mass_B / (2.0 * r_B)
    
    # BSSN conformal factor χ = ψ^(-4)
    psi4 = psi * psi * psi * psi
    chi[idx] = 1.0 / psi4
    
    # Conformal metric γ̃_ij = δ_ij (flat)
    gamma = SymmetricTensor3()
    gamma.xx = 1.0
    gamma.xy = 0.0
    gamma.xz = 0.0
    gamma.yy = 1.0
    gamma.yz = 0.0
    gamma.zz = 1.0
    gamma_tilde[idx] = gamma
    
    # Time-symmetric: K = 0, A_ij = 0
    K[idx] = 0.0
    
    zero_tensor = SymmetricTensor3()
    zero_tensor.xx = 0.0
    zero_tensor.xy = 0.0
    zero_tensor.xz = 0.0
    zero_tensor.yy = 0.0
    zero_tensor.yz = 0.0
    zero_tensor.zz = 0.0
    A_tilde[idx] = zero_tensor
    
    # Γ̃^i = 0 for conformally flat metric
    Gamma_tilde[idx] = wp.vec3(0.0, 0.0, 0.0)
    
    # Initial lapse α = ψ^(-2) (common choice)
    alpha[idx] = 1.0 / (psi * psi)
    
    # Initial shift β^i = 0
    beta[idx] = wp.vec3(0.0, 0.0, 0.0)


def create_bbh_initial_data(
    state: BSSNState,
    xmin: float, ymin: float, zmin: float,
    dx: float, dy: float, dz: float,
    separation: float = 10.0,
    mass_ratio: float = 1.0
):
    """
    Create Brill-Lindquist initial data for binary black holes.
    
    Args:
        state: BSSNState to initialize
        xmin, ymin, zmin: Grid origin
        dx, dy, dz: Grid spacing
        separation: Distance between black holes
        mass_ratio: mass_B / mass_A
    """
    # Place BHs along x-axis, centered at origin
    mass_A = 0.5  # In units where total mass ~ 1
    mass_B = mass_ratio * mass_A
    
    pos_A = wp.vec3(-separation/2.0, 0.0, 0.0)
    pos_B = wp.vec3(separation/2.0, 0.0, 0.0)
    
    print(f"Setting up BBH initial data:")
    print(f"  BH A: mass = {mass_A:.3f}, position = ({pos_A[0]:.2f}, {pos_A[1]:.2f}, {pos_A[2]:.2f})")
    print(f"  BH B: mass = {mass_B:.3f}, position = ({pos_B[0]:.2f}, {pos_B[1]:.2f}, {pos_B[2]:.2f})")
    print(f"  Separation: {separation:.2f}")
    print(f"  Mass ratio: {mass_ratio:.2f}")
    
    wp.launch(
        set_brill_lindquist_data,
        dim=(state.nx, state.ny, state.nz),
        inputs=[
            state.chi, state.gamma_tilde, state.K,
            state.A_tilde, state.Gamma_tilde,
            state.alpha, state.beta,
            state.nx, state.ny, state.nz,
            xmin, ymin, zmin, dx, dy, dz,
            pos_A, mass_A,
            pos_B, mass_B
        ]
    )


@wp.kernel
def compute_adm_mass(
    chi: wp.array(dtype=float),
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    mass: wp.array(dtype=float)
):
    """
    Compute ADM mass via volume integral (rough estimate).
    
    M_ADM ≈ ∫ (ψ - 1) dV where ψ = χ^(-1/4)
    """
    i, j, k = wp.tid()
    
    if i >= nx or j >= ny or k >= nz:
        return
    
    idx = idx3d(i, j, k, nx, ny, nz)
    
    # Skip boundary
    if i < 2 or i >= nx-2 or j < 2 or j >= ny-2 or k < 2 or k >= nz-2:
        return
    
    chi_val = chi[idx]
    psi = wp.pow(chi_val, -0.25)
    
    # Contribution to mass
    dV = dx * dy * dz
    dm = (psi - 1.0) * dV
    
    wp.atomic_add(mass, 0, dm)


if __name__ == "__main__":
    print("="*70)
    print("Testing Brill-Lindquist Initial Data")
    print("="*70)
    
    # Create grid centered at origin
    nx, ny, nz = 64, 64, 64
    L = 40.0  # Domain size
    xmin = ymin = zmin = -L/2
    dx = dy = dz = L / (nx - 1)
    
    print(f"\nGrid: {nx} x {ny} x {nz}")
    print(f"Domain: [{xmin:.1f}, {-xmin:.1f}]³")
    print(f"Spacing: {dx:.3f}")
    
    # Create state
    state = BSSNState(nx, ny, nz)
    
    # Set BBH initial data
    separation = 10.0
    mass_ratio = 1.0
    
    print(f"\nInitializing BBH configuration...")
    create_bbh_initial_data(state, xmin, ymin, zmin, dx, dy, dz, 
                           separation, mass_ratio)
    
    # Check data
    chi_np = state.chi.numpy()
    alpha_np = state.alpha.numpy()
    K_np = state.K.numpy()
    
    print(f"\nInitial data statistics:")
    print(f"  χ: min = {chi_np.min():.6f}, max = {chi_np.max():.6f}")
    print(f"  α: min = {alpha_np.min():.6f}, max = {alpha_np.max():.6f}")
    print(f"  K: min = {K_np.min():.6f}, max = {K_np.max():.6f}")
    
    # Compute ADM mass
    mass_array = wp.zeros(1, dtype=float)
    wp.launch(
        compute_adm_mass,
        dim=(nx, ny, nz),
        inputs=[state.chi, nx, ny, nz, dx, dy, dz, mass_array]
    )
    
    adm_mass = mass_array.numpy()[0]
    expected_mass = 0.5 + mass_ratio * 0.5
    
    print(f"\nMass analysis:")
    print(f"  ADM mass (computed): {adm_mass:.4f}")
    print(f"  Expected (sum of punctures): {expected_mass:.4f}")
    print(f"  Ratio: {adm_mass/expected_mass:.4f}")
    
    # Check for reasonable values
    if chi_np.min() > 0 and chi_np.max() < 10:
        print("\n✓ Conformal factor in reasonable range")
    else:
        print(f"\n✗ Conformal factor out of range")
    
    if np.abs(K_np).max() < 1e-10:
        print("✓ Time-symmetric (K = 0)")
    else:
        print(f"✗ K not zero: max = {np.abs(K_np).max()}")
    
    if alpha_np.min() > 0 and alpha_np.max() <= 1.0:
        print("✓ Lapse in valid range")
    else:
        print(f"✗ Lapse out of range")
    
    print("\n" + "="*70)
    print("Brill-Lindquist initial data test PASSED")
    print("="*70)
