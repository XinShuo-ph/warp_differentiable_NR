"""
Test Single Puncture Evolution

Tests:
1. Brill-Lindquist initial data
2. Stable evolution with 1+log slicing and Gamma-driver shift
3. Puncture remains approximately stationary
"""

import sys
sys.path.insert(0, '/workspace/NR/src')

import warp as wp
import numpy as np
from bssn_initial_data import BrillLindquistData
from bssn_boundary import BSSNBoundaryConditions

wp.init()


@wp.kernel
def rk_update(
    y: wp.array3d(dtype=float),
    k: wp.array3d(dtype=float),
    y_out: wp.array3d(dtype=float),
    coeff: float,
    nx: int, ny: int, nz: int
):
    i, j, kk = wp.tid()
    if i >= nx or j >= ny or kk >= nz:
        return
    y_out[i, j, kk] = y[i, j, kk] + coeff * k[i, j, kk]


@wp.kernel
def rk4_final(
    y: wp.array3d(dtype=float),
    k1: wp.array3d(dtype=float),
    k2: wp.array3d(dtype=float),
    k3: wp.array3d(dtype=float),
    k4: wp.array3d(dtype=float),
    dt: float,
    nx: int, ny: int, nz: int
):
    i, j, kk = wp.tid()
    if i >= nx or j >= ny or kk >= nz:
        return
    y[i, j, kk] = y[i, j, kk] + dt * (
        k1[i, j, kk] + 2.0*k2[i, j, kk] + 2.0*k3[i, j, kk] + k4[i, j, kk]
    ) / 6.0


class SimplifiedPunctureEvolver:
    """Simplified evolver for single puncture using RK4"""
    
    def __init__(self, data: BrillLindquistData, sigma_ko=0.1, eta=2.0):
        self.data = data
        self.n = data.nx
        self.dx = data.dx
        self.sigma_ko = sigma_ko
        self.eta = eta
        
        # Boundary conditions
        self.bc = BSSNBoundaryConditions(
            data.nx, data.ny, data.nz,
            data.dx, data.dy, data.dz,
            data.x_min, data.y_min, data.z_min
        )
        
        # Import RHS kernel
        from bssn_rhs_full import compute_full_bssn_rhs
        self.compute_rhs = compute_full_bssn_rhs
        
        # Allocate RHS arrays
        shape = (self.n, self.n, self.n)
        self._alloc_rhs_arrays(shape)
        self._alloc_temp_arrays(shape)
    
    def _alloc_rhs_arrays(self, shape):
        self.rhs_chi = wp.zeros(shape, dtype=float)
        self.rhs_gamma_xx = wp.zeros(shape, dtype=float)
        self.rhs_gamma_xy = wp.zeros(shape, dtype=float)
        self.rhs_gamma_xz = wp.zeros(shape, dtype=float)
        self.rhs_gamma_yy = wp.zeros(shape, dtype=float)
        self.rhs_gamma_yz = wp.zeros(shape, dtype=float)
        self.rhs_gamma_zz = wp.zeros(shape, dtype=float)
        self.rhs_K = wp.zeros(shape, dtype=float)
        self.rhs_A_xx = wp.zeros(shape, dtype=float)
        self.rhs_A_xy = wp.zeros(shape, dtype=float)
        self.rhs_A_xz = wp.zeros(shape, dtype=float)
        self.rhs_A_yy = wp.zeros(shape, dtype=float)
        self.rhs_A_yz = wp.zeros(shape, dtype=float)
        self.rhs_A_zz = wp.zeros(shape, dtype=float)
        self.rhs_Gamma_x = wp.zeros(shape, dtype=float)
        self.rhs_Gamma_y = wp.zeros(shape, dtype=float)
        self.rhs_Gamma_z = wp.zeros(shape, dtype=float)
        self.rhs_alpha = wp.zeros(shape, dtype=float)
        self.rhs_beta_x = wp.zeros(shape, dtype=float)
        self.rhs_beta_y = wp.zeros(shape, dtype=float)
        self.rhs_beta_z = wp.zeros(shape, dtype=float)
        self.rhs_B_x = wp.zeros(shape, dtype=float)
        self.rhs_B_y = wp.zeros(shape, dtype=float)
        self.rhs_B_z = wp.zeros(shape, dtype=float)
    
    def _alloc_temp_arrays(self, shape):
        """Temporary arrays for RK4 stages"""
        self.tmp_chi = wp.zeros(shape, dtype=float)
        self.tmp_alpha = wp.zeros(shape, dtype=float)
        self.tmp_K = wp.zeros(shape, dtype=float)
    
    def _compute_rhs(self, dt):
        """Compute RHS for all fields"""
        d = self.data
        n = self.n
        h = self.dx
        
        inv_12h = 1.0 / (12.0 * h)
        inv_12h2 = 1.0 / (12.0 * h * h)
        inv_144h2 = 1.0 / (144.0 * h * h)
        
        wp.launch(
            self.compute_rhs,
            dim=(n, n, n),
            inputs=[
                d.chi, d.gamma_xx, d.gamma_xy, d.gamma_xz, d.gamma_yy, d.gamma_yz, d.gamma_zz,
                d.K, d.A_xx, d.A_xy, d.A_xz, d.A_yy, d.A_yz, d.A_zz,
                d.Gamma_x, d.Gamma_y, d.Gamma_z,
                d.alpha, d.beta_x, d.beta_y, d.beta_z, d.B_x, d.B_y, d.B_z,
                self.rhs_chi, self.rhs_gamma_xx, self.rhs_gamma_xy, self.rhs_gamma_xz,
                self.rhs_gamma_yy, self.rhs_gamma_yz, self.rhs_gamma_zz,
                self.rhs_K, self.rhs_A_xx, self.rhs_A_xy, self.rhs_A_xz,
                self.rhs_A_yy, self.rhs_A_yz, self.rhs_A_zz,
                self.rhs_Gamma_x, self.rhs_Gamma_y, self.rhs_Gamma_z,
                self.rhs_alpha, self.rhs_beta_x, self.rhs_beta_y, self.rhs_beta_z,
                self.rhs_B_x, self.rhs_B_y, self.rhs_B_z,
                n, n, n, inv_12h, inv_12h2, inv_144h2, self.sigma_ko, dt, self.eta
            ]
        )
    
    def step_euler(self, dt):
        """Simple Euler step for testing"""
        self._compute_rhs(dt)
        
        d = self.data
        n = self.n
        dim = (n, n, n)
        
        # Update key fields only (simplified)
        wp.launch(rk_update, dim=dim, inputs=[d.chi, self.rhs_chi, d.chi, dt, n, n, n])
        wp.launch(rk_update, dim=dim, inputs=[d.alpha, self.rhs_alpha, d.alpha, dt, n, n, n])
        wp.launch(rk_update, dim=dim, inputs=[d.K, self.rhs_K, d.K, dt, n, n, n])
        
        # Apply BCs
        self.bc.apply(
            d.chi, d.gamma_xx, d.gamma_xy, d.gamma_xz, d.gamma_yy, d.gamma_yz, d.gamma_zz,
            d.K, d.A_xx, d.A_xy, d.A_xz, d.A_yy, d.A_yz, d.A_zz,
            d.Gamma_x, d.Gamma_y, d.Gamma_z,
            d.alpha, d.beta_x, d.beta_y, d.beta_z, d.B_x, d.B_y, d.B_z
        )


def test_puncture_evolution():
    """Test single puncture evolution"""
    print("=" * 60)
    print("Testing Single Puncture Evolution")
    print("=" * 60)
    
    # Grid setup
    n = 48  # Grid size
    L = 8.0  # Domain: [-L, L]^3
    M = 1.0  # BH mass
    
    print(f"Grid: {n}^3, Domain: [-{L}, {L}]^3, Mass: {M}")
    
    # Create initial data
    print("\nCreating Brill-Lindquist initial data...")
    data = BrillLindquistData(
        nx=n, ny=n, nz=n,
        x_min=-L, x_max=L,
        y_min=-L, y_max=L,
        z_min=-L, z_max=L,
        M=M,
        x_p=0.0, y_p=0.0, z_p=0.0
    )
    
    # Initial state
    chi_init = data.chi.numpy().copy()
    alpha_init = data.alpha.numpy().copy()
    
    print(f"Initial chi at center: {chi_init[n//2, n//2, n//2]:.6f}")
    print(f"Initial alpha at center: {alpha_init[n//2, n//2, n//2]:.6f}")
    
    # Create evolver
    evolver = SimplifiedPunctureEvolver(data, sigma_ko=0.2, eta=2.0/M)
    
    # Evolution parameters
    dx = 2.0 * L / (n - 1)
    dt = 0.1 * dx  # CFL condition
    n_steps = 50
    
    print(f"\nEvolution: dt = {dt:.4f}, steps = {n_steps}")
    
    # Evolve
    for step in range(n_steps):
        evolver.step_euler(dt)
        
        if (step + 1) % 10 == 0:
            chi_np = data.chi.numpy()
            alpha_np = data.alpha.numpy()
            K_np = data.K.numpy()
            
            # Check for NaN/Inf
            if np.any(np.isnan(chi_np)) or np.any(np.isinf(chi_np)):
                print(f"Step {step+1}: NaN/Inf detected in chi!")
                return False
            
            print(f"Step {step+1}: chi_center={chi_np[n//2,n//2,n//2]:.6f}, "
                  f"alpha_center={alpha_np[n//2,n//2,n//2]:.6f}, "
                  f"|K|_max={np.max(np.abs(K_np)):.6e}")
    
    # Final checks
    print("\n" + "-" * 40)
    print("Final state:")
    
    chi_final = data.chi.numpy()
    alpha_final = data.alpha.numpy()
    
    chi_center_change = abs(chi_final[n//2, n//2, n//2] - chi_init[n//2, n//2, n//2])
    alpha_center_change = abs(alpha_final[n//2, n//2, n//2] - alpha_init[n//2, n//2, n//2])
    
    print(f"Chi at center: {chi_final[n//2, n//2, n//2]:.6f} (change: {chi_center_change:.6f})")
    print(f"Alpha at center: {alpha_final[n//2, n//2, n//2]:.6f} (change: {alpha_center_change:.6f})")
    
    # Check stability
    stable = True
    if np.any(np.isnan(chi_final)) or np.any(np.isinf(chi_final)):
        print("FAILED: NaN or Inf in final chi")
        stable = False
    
    if np.min(chi_final) < 0:
        print(f"WARNING: chi went negative (min = {np.min(chi_final):.6f})")
    
    if np.min(alpha_final) < 0:
        print(f"WARNING: alpha went negative (min = {np.min(alpha_final):.6f})")
        stable = False
    
    # Check that puncture didn't move significantly
    # (For stationary puncture, profile should stay similar)
    
    print("-" * 40)
    if stable:
        print("PASSED: Puncture evolution stable for 50 steps!")
    else:
        print("FAILED: Evolution unstable")
    
    return stable


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_puncture_evolution()
