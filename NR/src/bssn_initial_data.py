"""
BSSN Initial Data

Implements initial data for black hole spacetimes.
Starts with single Schwarzschild black hole in isotropic (puncture) coordinates.
"""

import warp as wp
import numpy as np


@wp.kernel
def set_schwarzschild_puncture_kernel(
    # Output arrays
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
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float,
    # BH parameters
    bh_mass: float,
    bh_x: float, bh_y: float, bh_z: float,
    # Options
    pre_collapse_lapse: int  # 1 = use pre-collapsed lapse, 0 = alpha=1
):
    """
    Set Schwarzschild puncture initial data.
    
    Puncture data for mass M at position (bh_x, bh_y, bh_z):
    - Conformal factor: ψ = 1 + M/(2r)
    - φ = ln(ψ) (log conformal factor)
    - Conformal metric: flat (γ̃ᵢⱼ = δᵢⱼ)
    - Extrinsic curvature: zero (time-symmetric)
    - Lapse: α = 1 or α = ψ^{-2} (pre-collapsed)
    - Shift: βⁱ = 0
    """
    tid = wp.tid()
    
    # Convert to 3D indices
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Grid coordinates (cell-centered)
    x = (float(i) + 0.5) * dx - float(nx) * dx / 2.0
    y = (float(j) + 0.5) * dx - float(ny) * dx / 2.0
    z = (float(k) + 0.5) * dx - float(nz) * dx / 2.0
    
    # Distance from BH
    rx = x - bh_x
    ry = y - bh_y
    rz = z - bh_z
    r = wp.sqrt(rx*rx + ry*ry + rz*rz)
    
    # Regularize at puncture (avoid division by zero)
    r_min = 0.1 * dx
    r = wp.max(r, r_min)
    
    # Conformal factor ψ = 1 + M/(2r)
    psi = 1.0 + bh_mass / (2.0 * r)
    
    # Log conformal factor φ = ln(ψ)
    # Note: In some formulations, W = ψ^{-2} = e^{-4φ}, so φ = ln(ψ)
    phi[tid] = wp.log(psi)
    
    # Conformal metric: flat
    gt11[tid] = 1.0
    gt12[tid] = 0.0
    gt13[tid] = 0.0
    gt22[tid] = 1.0
    gt23[tid] = 0.0
    gt33[tid] = 1.0
    
    # Extrinsic curvature: zero (time-symmetric data)
    trK[tid] = 0.0
    At11[tid] = 0.0
    At12[tid] = 0.0
    At13[tid] = 0.0
    At22[tid] = 0.0
    At23[tid] = 0.0
    At33[tid] = 0.0
    
    # Conformal connection: compute from flat metric with conformal factor
    # Γ̃ⁱ = γ̃ʲᵏΓ̃ⁱⱼₖ = 0 for flat conformal metric
    # But the physical Christoffel has contributions from ψ
    # For now, set to zero and let evolution correct it
    Xt1[tid] = 0.0
    Xt2[tid] = 0.0
    Xt3[tid] = 0.0
    
    # Lapse
    if pre_collapse_lapse == 1:
        # Pre-collapsed lapse: α = ψ^{-2}
        # This helps avoid initial gauge dynamics
        alpha[tid] = 1.0 / (psi * psi)
    else:
        alpha[tid] = 1.0
    
    # Shift: zero initially
    beta1[tid] = 0.0
    beta2[tid] = 0.0
    beta3[tid] = 0.0


def set_schwarzschild_puncture(grid, bh_mass=1.0, bh_pos=(0.0, 0.0, 0.0), 
                                pre_collapse_lapse=True):
    """
    Set Schwarzschild puncture initial data on a BSSNGrid.
    
    Args:
        grid: BSSNGrid object
        bh_mass: Black hole mass (in geometric units, G=c=1)
        bh_pos: (x, y, z) position of black hole
        pre_collapse_lapse: If True, use pre-collapsed lapse α = ψ^{-2}
    """
    wp.launch(
        set_schwarzschild_puncture_kernel,
        dim=grid.n_points,
        inputs=[
            grid.phi, grid.gt11, grid.gt12, grid.gt13, grid.gt22, grid.gt23, grid.gt33,
            grid.trK, grid.At11, grid.At12, grid.At13, grid.At22, grid.At23, grid.At33,
            grid.Xt1, grid.Xt2, grid.Xt3,
            grid.alpha, grid.beta1, grid.beta2, grid.beta3,
            grid.nx, grid.ny, grid.nz, grid.dx,
            bh_mass, bh_pos[0], bh_pos[1], bh_pos[2],
            1 if pre_collapse_lapse else 0
        ]
    )


@wp.kernel
def set_brill_lindquist_kernel(
    # Output arrays
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
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float,
    # BBH parameters
    m1: float, x1: float, y1: float, z1: float,
    m2: float, x2: float, y2: float, z2: float
):
    """
    Set Brill-Lindquist initial data for two black holes.
    
    Conformal factor: ψ = 1 + M₁/(2r₁) + M₂/(2r₂)
    """
    tid = wp.tid()
    
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    x = (float(i) + 0.5) * dx - float(nx) * dx / 2.0
    y = (float(j) + 0.5) * dx - float(ny) * dx / 2.0
    z = (float(k) + 0.5) * dx - float(nz) * dx / 2.0
    
    r_min = 0.1 * dx
    
    # Distance from BH 1
    r1 = wp.sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1))
    r1 = wp.max(r1, r_min)
    
    # Distance from BH 2
    r2 = wp.sqrt((x-x2)*(x-x2) + (y-y2)*(y-y2) + (z-z2)*(z-z2))
    r2 = wp.max(r2, r_min)
    
    # Brill-Lindquist conformal factor
    psi = 1.0 + m1/(2.0*r1) + m2/(2.0*r2)
    
    phi[tid] = wp.log(psi)
    
    gt11[tid] = 1.0
    gt12[tid] = 0.0
    gt13[tid] = 0.0
    gt22[tid] = 1.0
    gt23[tid] = 0.0
    gt33[tid] = 1.0
    
    trK[tid] = 0.0
    At11[tid] = 0.0
    At12[tid] = 0.0
    At13[tid] = 0.0
    At22[tid] = 0.0
    At23[tid] = 0.0
    At33[tid] = 0.0
    
    Xt1[tid] = 0.0
    Xt2[tid] = 0.0
    Xt3[tid] = 0.0
    
    alpha[tid] = 1.0 / (psi * psi)
    
    beta1[tid] = 0.0
    beta2[tid] = 0.0
    beta3[tid] = 0.0


def set_brill_lindquist(grid, m1=0.5, pos1=(-2.0, 0.0, 0.0),
                         m2=0.5, pos2=(2.0, 0.0, 0.0)):
    """
    Set Brill-Lindquist initial data for binary black holes.
    
    Args:
        grid: BSSNGrid object
        m1, m2: Black hole masses
        pos1, pos2: (x, y, z) positions of black holes
    """
    wp.launch(
        set_brill_lindquist_kernel,
        dim=grid.n_points,
        inputs=[
            grid.phi, grid.gt11, grid.gt12, grid.gt13, grid.gt22, grid.gt23, grid.gt33,
            grid.trK, grid.At11, grid.At12, grid.At13, grid.At22, grid.At23, grid.At33,
            grid.Xt1, grid.Xt2, grid.Xt3,
            grid.alpha, grid.beta1, grid.beta2, grid.beta3,
            grid.nx, grid.ny, grid.nz, grid.dx,
            m1, pos1[0], pos1[1], pos1[2],
            m2, pos2[0], pos2[1], pos2[2]
        ]
    )


def test_schwarzschild_initial_data():
    """Test Schwarzschild puncture initial data."""
    import sys
    sys.path.insert(0, '/workspace/NR/src')
    from bssn_vars import BSSNGrid
    
    wp.init()
    print("=== Schwarzschild Puncture Initial Data Test ===\n")
    
    # Create grid: 10M domain with M=1
    nx, ny, nz = 64, 64, 64
    domain_size = 20.0  # -10M to +10M
    dx = domain_size / nx
    
    print(f"Grid: {nx}x{ny}x{nz}, dx = {dx:.4f}M")
    print(f"Domain: [-{domain_size/2:.1f}M, +{domain_size/2:.1f}M]³")
    
    grid = BSSNGrid(nx, ny, nz, dx)
    
    # Set Schwarzschild data with M=1 at origin
    bh_mass = 1.0
    set_schwarzschild_puncture(grid, bh_mass=bh_mass, bh_pos=(0.0, 0.0, 0.0),
                                pre_collapse_lapse=True)
    
    # Check values at various radii
    phi_np = grid.phi.numpy().reshape(nx, ny, nz)
    alpha_np = grid.alpha.numpy().reshape(nx, ny, nz)
    gt11_np = grid.gt11.numpy().reshape(nx, ny, nz)
    
    print(f"\nBlack hole mass M = {bh_mass}")
    print("\nChecking values at different radii (along x-axis):")
    
    center = nx // 2
    for di in [0, 2, 4, 8, 16]:
        i = center + di
        if i < nx:
            x = (i + 0.5) * dx - nx * dx / 2.0
            r = abs(x)
            
            phi_val = phi_np[i, center, center]
            alpha_val = alpha_np[i, center, center]
            gt11_val = gt11_np[i, center, center]
            
            # Expected values
            psi_expected = 1.0 + bh_mass / (2.0 * max(r, 0.1*dx))
            phi_expected = np.log(psi_expected)
            alpha_expected = 1.0 / (psi_expected**2)
            
            print(f"  r = {r:6.3f}M: φ = {phi_val:8.4f} (exp: {phi_expected:8.4f}), "
                  f"α = {alpha_val:8.4f} (exp: {alpha_expected:8.4f}), γ̃₁₁ = {gt11_val:.1f}")
    
    # Check conformal metric is flat
    gt12_max = np.abs(grid.gt12.numpy()).max()
    gt_diag_mean = (grid.gt11.numpy().mean() + grid.gt22.numpy().mean() + grid.gt33.numpy().mean()) / 3
    print(f"\nConformal metric check:")
    print(f"  γ̃₁₂ max: {gt12_max:.6f} (should be 0)")
    print(f"  γ̃ᵢᵢ mean: {gt_diag_mean:.6f} (should be 1)")
    
    # Check extrinsic curvature is zero
    trK_max = np.abs(grid.trK.numpy()).max()
    At11_max = np.abs(grid.At11.numpy()).max()
    print(f"\nExtrinsic curvature check:")
    print(f"  K max:   {trK_max:.6f} (should be 0)")
    print(f"  Ã₁₁ max: {At11_max:.6f} (should be 0)")
    
    # Check lapse minimum (should be at center, near puncture)
    alpha_min = alpha_np.min()
    alpha_max = alpha_np.max()
    print(f"\nLapse (pre-collapsed):")
    print(f"  α min: {alpha_min:.6f} (near puncture)")
    print(f"  α max: {alpha_max:.6f} (far from puncture)")
    
    print("\n✓ Schwarzschild puncture initial data test completed.")


if __name__ == "__main__":
    test_schwarzschild_initial_data()
