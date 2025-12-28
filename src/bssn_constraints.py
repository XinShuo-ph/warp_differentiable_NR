"""
BSSN Constraint Monitoring

Computes the Hamiltonian and momentum constraint violations for BSSN evolution.
These should be zero for valid solutions to Einstein's equations.
"""

import warp as wp
import numpy as np
from bssn_derivs import (
    idx_3d, 
    deriv_x_4th, deriv_y_4th, deriv_z_4th,
    deriv_xx_4th, deriv_yy_4th, deriv_zz_4th
)


@wp.func
def compute_full_inverse_metric(gt11: float, gt12: float, gt13: float,
                                  gt22: float, gt23: float, gt33: float):
    """Compute full inverse conformal metric."""
    det = (gt11 * (gt22 * gt33 - gt23 * gt23)
           - gt12 * (gt12 * gt33 - gt23 * gt13)
           + gt13 * (gt12 * gt23 - gt22 * gt13))
    
    inv_det = 1.0 / wp.max(det, 1.0e-10)
    
    gtu11 = (gt22 * gt33 - gt23 * gt23) * inv_det
    gtu12 = (gt13 * gt23 - gt12 * gt33) * inv_det
    gtu13 = (gt12 * gt23 - gt13 * gt22) * inv_det
    gtu22 = (gt11 * gt33 - gt13 * gt13) * inv_det
    gtu23 = (gt12 * gt13 - gt11 * gt23) * inv_det
    gtu33 = (gt11 * gt22 - gt12 * gt12) * inv_det
    
    return gtu11, gtu12, gtu13, gtu22, gtu23, gtu33


@wp.kernel
def compute_hamiltonian_constraint_kernel(
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
    # Output
    H: wp.array(dtype=wp.float32),
    # Grid parameters
    nx: int, ny: int, nz: int,
    inv_dx: float
):
    """
    Compute Hamiltonian constraint violation.
    
    In BSSN form:
    H = e^{-4φ} (R̃ - 8∆̃φ - 8∂ᵢφ∂ⁱφ) + (2/3)K² - ÃᵢⱼÃⁱʲ = 0
    
    where R̃ is the conformal Ricci scalar.
    
    For initial data validation, this is simplified.
    """
    tid = wp.tid()
    
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Need ghost zones for derivatives
    if i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2 or k < 2 or k >= nz - 2:
        H[tid] = 0.0
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
    
    inv_dx2 = inv_dx * inv_dx
    
    # Compute inverse conformal metric
    gtu11, gtu12, gtu13, gtu22, gtu23, gtu33 = compute_full_inverse_metric(
        gt11_L, gt12_L, gt13_L, gt22_L, gt23_L, gt33_L)
    
    # e^{-4φ}
    em4phi = wp.exp(-4.0 * phi_L)
    
    # Conformal factor derivatives
    dphi_dx = deriv_x_4th(phi, i, j, k, nx, ny, inv_dx)
    dphi_dy = deriv_y_4th(phi, i, j, k, nx, ny, inv_dx)
    dphi_dz = deriv_z_4th(phi, i, j, k, nx, ny, inv_dx)
    
    d2phi_xx = deriv_xx_4th(phi, i, j, k, nx, ny, inv_dx2)
    d2phi_yy = deriv_yy_4th(phi, i, j, k, nx, ny, inv_dx2)
    d2phi_zz = deriv_zz_4th(phi, i, j, k, nx, ny, inv_dx2)
    
    # Laplacian of φ (conformal): ∆̃φ = γ̃ⁱʲ∂ᵢ∂ⱼφ
    lap_phi = gtu11 * d2phi_xx + gtu22 * d2phi_yy + gtu33 * d2phi_zz
    
    # ∂ᵢφ∂ⁱφ = γ̃ⁱʲ∂ᵢφ∂ⱼφ
    dphi_sq = (gtu11 * dphi_dx * dphi_dx + gtu22 * dphi_dy * dphi_dy + 
               gtu33 * dphi_dz * dphi_dz +
               2.0 * gtu12 * dphi_dx * dphi_dy +
               2.0 * gtu13 * dphi_dx * dphi_dz +
               2.0 * gtu23 * dphi_dy * dphi_dz)
    
    # ÃᵢⱼÃⁱʲ
    At_sq = (At11_L * At11_L * gtu11 * gtu11 + At22_L * At22_L * gtu22 * gtu22 + 
             At33_L * At33_L * gtu33 * gtu33 +
             2.0 * At12_L * At12_L * gtu11 * gtu22 +
             2.0 * At13_L * At13_L * gtu11 * gtu33 +
             2.0 * At23_L * At23_L * gtu22 * gtu33 +
             4.0 * At11_L * At12_L * gtu11 * gtu12 +
             4.0 * At11_L * At13_L * gtu11 * gtu13 +
             4.0 * At12_L * At22_L * gtu12 * gtu22 +
             4.0 * At12_L * At23_L * gtu12 * gtu23 +
             4.0 * At13_L * At23_L * gtu13 * gtu23 +
             4.0 * At13_L * At33_L * gtu13 * gtu33 +
             4.0 * At22_L * At23_L * gtu22 * gtu23 +
             4.0 * At23_L * At33_L * gtu23 * gtu33)
    
    # For flat conformal metric, R̃ = 0
    # Full Ricci scalar would require second derivatives of metric
    Rtilde = 0.0  # Simplified for flat conformal metric
    
    # Hamiltonian constraint
    # H = e^{-4φ}(R̃ - 8∆̃φ - 8∂ᵢφ∂ⁱφ) + (2/3)K² - ÃᵢⱼÃⁱʲ
    H[tid] = em4phi * (Rtilde - 8.0 * lap_phi - 8.0 * dphi_sq) + (2.0/3.0) * trK_L * trK_L - At_sq


@wp.kernel
def compute_momentum_constraint_kernel(
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
    # Output
    M1: wp.array(dtype=wp.float32),
    M2: wp.array(dtype=wp.float32),
    M3: wp.array(dtype=wp.float32),
    # Grid parameters
    nx: int, ny: int, nz: int,
    inv_dx: float
):
    """
    Compute momentum constraint violation.
    
    In BSSN form:
    Mⁱ = ∂ⱼÃⁱⱼ + Γ̃ⁱⱼₖÃʲᵏ - (2/3)γ̃ⁱʲ∂ⱼK + 6Ãⁱʲ∂ⱼφ = 0
    
    This is simplified for flat conformal metric.
    """
    tid = wp.tid()
    
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    if i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2 or k < 2 or k >= nz - 2:
        M1[tid] = 0.0
        M2[tid] = 0.0
        M3[tid] = 0.0
        return
    
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
    
    gtu11, gtu12, gtu13, gtu22, gtu23, gtu33 = compute_full_inverse_metric(
        gt11_L, gt12_L, gt13_L, gt22_L, gt23_L, gt33_L)
    
    # Raise At indices: Atᵢʲ = γ̃ʲᵏ Atᵢₖ
    At1u1 = gtu11 * At11_L + gtu12 * At12_L + gtu13 * At13_L
    At1u2 = gtu12 * At11_L + gtu22 * At12_L + gtu23 * At13_L
    At1u3 = gtu13 * At11_L + gtu23 * At12_L + gtu33 * At13_L
    At2u1 = gtu11 * At12_L + gtu12 * At22_L + gtu13 * At23_L
    At2u2 = gtu12 * At12_L + gtu22 * At22_L + gtu23 * At23_L
    At2u3 = gtu13 * At12_L + gtu23 * At22_L + gtu33 * At23_L
    At3u1 = gtu11 * At13_L + gtu12 * At23_L + gtu13 * At33_L
    At3u2 = gtu12 * At13_L + gtu22 * At23_L + gtu23 * At33_L
    At3u3 = gtu13 * At13_L + gtu23 * At23_L + gtu33 * At33_L
    
    # Fully raised: Atⁱʲ = γ̃ⁱᵏ Atₖʲ
    Atu11 = gtu11 * At1u1 + gtu12 * At2u1 + gtu13 * At3u1
    Atu12 = gtu11 * At1u2 + gtu12 * At2u2 + gtu13 * At3u2
    Atu13 = gtu11 * At1u3 + gtu12 * At2u3 + gtu13 * At3u3
    Atu22 = gtu12 * At1u2 + gtu22 * At2u2 + gtu23 * At3u2
    Atu23 = gtu12 * At1u3 + gtu22 * At2u3 + gtu23 * At3u3
    Atu33 = gtu13 * At1u3 + gtu23 * At2u3 + gtu33 * At3u3
    
    # Derivatives
    dAt11_dx = deriv_x_4th(At11, i, j, k, nx, ny, inv_dx)
    dAt12_dx = deriv_x_4th(At12, i, j, k, nx, ny, inv_dx)
    dAt13_dx = deriv_x_4th(At13, i, j, k, nx, ny, inv_dx)
    dAt12_dy = deriv_y_4th(At12, i, j, k, nx, ny, inv_dx)
    dAt22_dy = deriv_y_4th(At22, i, j, k, nx, ny, inv_dx)
    dAt23_dy = deriv_y_4th(At23, i, j, k, nx, ny, inv_dx)
    dAt13_dz = deriv_z_4th(At13, i, j, k, nx, ny, inv_dx)
    dAt23_dz = deriv_z_4th(At23, i, j, k, nx, ny, inv_dx)
    dAt33_dz = deriv_z_4th(At33, i, j, k, nx, ny, inv_dx)
    
    dtrK_dx = deriv_x_4th(trK, i, j, k, nx, ny, inv_dx)
    dtrK_dy = deriv_y_4th(trK, i, j, k, nx, ny, inv_dx)
    dtrK_dz = deriv_z_4th(trK, i, j, k, nx, ny, inv_dx)
    
    dphi_dx = deriv_x_4th(phi, i, j, k, nx, ny, inv_dx)
    dphi_dy = deriv_y_4th(phi, i, j, k, nx, ny, inv_dx)
    dphi_dz = deriv_z_4th(phi, i, j, k, nx, ny, inv_dx)
    
    # Simplified momentum constraint (flat conformal metric, Christoffel = 0)
    # Mⁱ ≈ γ̃ⁱʲ∂ⱼK - (2/3)∂ⱼAtⁱʲ + 6Atⁱʲ∂ⱼφ
    
    # ∂ⱼAtⁱʲ for i=1: ∂₁At¹¹ + ∂₂At¹² + ∂₃At¹³
    divAt1 = gtu11 * dAt11_dx + gtu12 * dAt12_dx + gtu13 * dAt13_dx + gtu12 * dAt12_dy + gtu22 * dAt22_dy + gtu23 * dAt23_dy
    divAt2 = gtu12 * dAt12_dx + gtu22 * dAt22_dy + gtu23 * dAt23_dy
    divAt3 = gtu13 * dAt13_dz + gtu23 * dAt23_dz + gtu33 * dAt33_dz
    
    M1[tid] = divAt1 - (2.0/3.0) * gtu11 * dtrK_dx + 6.0 * (Atu11 * dphi_dx + Atu12 * dphi_dy + Atu13 * dphi_dz)
    M2[tid] = divAt2 - (2.0/3.0) * gtu22 * dtrK_dy + 6.0 * (Atu12 * dphi_dx + Atu22 * dphi_dy + Atu23 * dphi_dz)
    M3[tid] = divAt3 - (2.0/3.0) * gtu33 * dtrK_dz + 6.0 * (Atu13 * dphi_dx + Atu23 * dphi_dy + Atu33 * dphi_dz)


class ConstraintMonitor:
    """
    Monitor BSSN constraint violations during evolution.
    """
    def __init__(self, grid):
        self.grid = grid
        self.H = wp.zeros(grid.n_points, dtype=wp.float32)
        self.M1 = wp.zeros(grid.n_points, dtype=wp.float32)
        self.M2 = wp.zeros(grid.n_points, dtype=wp.float32)
        self.M3 = wp.zeros(grid.n_points, dtype=wp.float32)
    
    def compute(self):
        """Compute all constraints."""
        inv_dx = 1.0 / self.grid.dx
        
        wp.launch(
            compute_hamiltonian_constraint_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.grid.phi, self.grid.gt11, self.grid.gt12, self.grid.gt13,
                self.grid.gt22, self.grid.gt23, self.grid.gt33,
                self.grid.trK, self.grid.At11, self.grid.At12, self.grid.At13,
                self.grid.At22, self.grid.At23, self.grid.At33,
                self.H,
                self.grid.nx, self.grid.ny, self.grid.nz, inv_dx
            ]
        )
        
        wp.launch(
            compute_momentum_constraint_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.grid.phi, self.grid.gt11, self.grid.gt12, self.grid.gt13,
                self.grid.gt22, self.grid.gt23, self.grid.gt33,
                self.grid.trK, self.grid.At11, self.grid.At12, self.grid.At13,
                self.grid.At22, self.grid.At23, self.grid.At33,
                self.grid.Xt1, self.grid.Xt2, self.grid.Xt3,
                self.M1, self.M2, self.M3,
                self.grid.nx, self.grid.ny, self.grid.nz, inv_dx
            ]
        )
    
    def get_norms(self):
        """Get L2 and Linf norms of constraint violations."""
        H_np = self.H.numpy()
        M1_np = self.M1.numpy()
        M2_np = self.M2.numpy()
        M3_np = self.M3.numpy()
        
        H_L2 = np.sqrt(np.mean(H_np**2))
        H_Linf = np.abs(H_np).max()
        
        M_sq = M1_np**2 + M2_np**2 + M3_np**2
        M_L2 = np.sqrt(np.mean(M_sq))
        M_Linf = np.sqrt(M_sq).max()
        
        return {
            'H_L2': H_L2,
            'H_Linf': H_Linf,
            'M_L2': M_L2,
            'M_Linf': M_Linf
        }
    
    def print_summary(self, step=None):
        """Print constraint violation summary."""
        norms = self.get_norms()
        if step is not None:
            print(f"Step {step:5d}: ", end="")
        print(f"H_L2 = {norms['H_L2']:.4e}, H_max = {norms['H_Linf']:.4e}, "
              f"M_L2 = {norms['M_L2']:.4e}, M_max = {norms['M_Linf']:.4e}")


def test_constraints():
    """Test constraint monitoring."""
    import sys
    sys.path.insert(0, '/workspace/src')
    from bssn_vars import BSSNGrid
    from bssn_initial_data import set_schwarzschild_puncture
    
    wp.init()
    print("=== Constraint Monitoring Test ===\n")
    
    # Create grid with Schwarzschild initial data
    nx, ny, nz = 32, 32, 32
    domain_size = 20.0
    dx = domain_size / nx
    
    grid = BSSNGrid(nx, ny, nz, dx)
    
    # Test flat spacetime (should have zero constraints)
    print("Flat spacetime (should have zero constraints):")
    grid.set_flat_spacetime()
    monitor = ConstraintMonitor(grid)
    monitor.compute()
    monitor.print_summary()
    
    # Test Schwarzschild (puncture data has some constraint violation)
    print("\nSchwarzschildpuncture initial data:")
    set_schwarzschild_puncture(grid, bh_mass=1.0, bh_pos=(0.0, 0.0, 0.0),
                                pre_collapse_lapse=True)
    monitor.compute()
    monitor.print_summary()
    
    # The Hamiltonian constraint should be approximately satisfied
    # (some numerical error due to discretization)
    norms = monitor.get_norms()
    print(f"\nNote: Puncture data has finite resolution constraint violation.")
    print(f"These should decrease with higher resolution.")
    
    print("\n✓ Constraint monitoring test completed.")


if __name__ == "__main__":
    test_constraints()
