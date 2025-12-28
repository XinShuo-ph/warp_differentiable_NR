"""
BSSN Black Hole Evolution Test

Tests the evolution of a single Schwarzschild black hole using the full BSSN
implementation with:
- Schwarzschild puncture initial data
- Complete BSSN RHS
- Sommerfeld radiative boundary conditions
- 1+log slicing and Gamma-driver shift
- Constraint monitoring
"""

import sys
sys.path.insert(0, '/workspace/NR/src')

import warp as wp
import numpy as np
from bssn_vars import BSSNGrid
from bssn_initial_data import set_schwarzschild_puncture
from bssn_rhs_full import compute_bssn_rhs_full_kernel
from bssn_boundary import apply_standard_bssn_boundaries
from bssn_constraints import ConstraintMonitor


@wp.kernel
def rk4_update_kernel(
    u: wp.array(dtype=wp.float32),
    u0: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),
    dt: float,
    coeff: float
):
    """u = u0 + coeff * dt * k"""
    tid = wp.tid()
    u[tid] = u0[tid] + coeff * dt * k[tid]


@wp.kernel
def rk4_accumulate_kernel(
    u_final: wp.array(dtype=wp.float32),
    u0: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),
    dt: float,
    weight: float
):
    """u_final = u0 + weight * dt * k (accumulates)"""
    tid = wp.tid()
    u_final[tid] = u_final[tid] + weight * dt * k[tid]


@wp.kernel
def copy_kernel(
    dst: wp.array(dtype=wp.float32),
    src: wp.array(dtype=wp.float32)
):
    """dst = src"""
    tid = wp.tid()
    dst[tid] = src[tid]


class BSSNEvolver:
    """
    Full BSSN time evolution with RK4 integration.
    """
    def __init__(self, grid, eps_diss=0.2):
        self.grid = grid
        self.eps_diss = eps_diss * grid.dx
        self.inv_dx = 1.0 / grid.dx
        
        # Storage for RK4 stages
        self.var_names = [
            'phi', 'gt11', 'gt12', 'gt13', 'gt22', 'gt23', 'gt33',
            'trK', 'At11', 'At12', 'At13', 'At22', 'At23', 'At33',
            'Xt1', 'Xt2', 'Xt3', 'alpha', 'beta1', 'beta2', 'beta3'
        ]
        
        # Allocate storage for initial values
        self.u0 = {}
        self.u_acc = {}
        for name in self.var_names:
            self.u0[name] = wp.zeros(grid.n_points, dtype=wp.float32)
            self.u_acc[name] = wp.zeros(grid.n_points, dtype=wp.float32)
    
    def compute_rhs(self):
        """Compute BSSN RHS and apply boundary conditions."""
        wp.launch(
            compute_bssn_rhs_full_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.grid.phi, self.grid.gt11, self.grid.gt12, self.grid.gt13,
                self.grid.gt22, self.grid.gt23, self.grid.gt33,
                self.grid.trK, self.grid.At11, self.grid.At12, self.grid.At13,
                self.grid.At22, self.grid.At23, self.grid.At33,
                self.grid.Xt1, self.grid.Xt2, self.grid.Xt3,
                self.grid.alpha, self.grid.beta1, self.grid.beta2, self.grid.beta3,
                self.grid.phi_rhs, self.grid.gt11_rhs, self.grid.gt12_rhs, self.grid.gt13_rhs,
                self.grid.gt22_rhs, self.grid.gt23_rhs, self.grid.gt33_rhs,
                self.grid.trK_rhs, self.grid.At11_rhs, self.grid.At12_rhs, self.grid.At13_rhs,
                self.grid.At22_rhs, self.grid.At23_rhs, self.grid.At33_rhs,
                self.grid.Xt1_rhs, self.grid.Xt2_rhs, self.grid.Xt3_rhs,
                self.grid.alpha_rhs, self.grid.beta1_rhs, self.grid.beta2_rhs, self.grid.beta3_rhs,
                self.grid.nx, self.grid.ny, self.grid.nz, self.inv_dx, self.eps_diss
            ]
        )
        
        # Apply boundary conditions
        apply_standard_bssn_boundaries(self.grid)
    
    def _get_var(self, name):
        return getattr(self.grid, name)
    
    def _get_rhs(self, name):
        return getattr(self.grid, name + '_rhs')
    
    def step(self, dt):
        """Perform one RK4 time step."""
        # Save initial state
        for name in self.var_names:
            wp.launch(copy_kernel, dim=self.grid.n_points,
                      inputs=[self.u0[name], self._get_var(name)])
            # Initialize accumulator to u0
            wp.launch(copy_kernel, dim=self.grid.n_points,
                      inputs=[self.u_acc[name], self.u0[name]])
        
        # k1
        self.compute_rhs()
        for name in self.var_names:
            # u_acc += (1/6)*dt*k1
            wp.launch(rk4_accumulate_kernel, dim=self.grid.n_points,
                      inputs=[self.u_acc[name], self.u0[name], self._get_rhs(name), dt, 1.0/6.0])
            # u = u0 + 0.5*dt*k1
            wp.launch(rk4_update_kernel, dim=self.grid.n_points,
                      inputs=[self._get_var(name), self.u0[name], self._get_rhs(name), dt, 0.5])
        
        # k2
        self.compute_rhs()
        for name in self.var_names:
            # u_acc += (1/3)*dt*k2
            wp.launch(rk4_accumulate_kernel, dim=self.grid.n_points,
                      inputs=[self.u_acc[name], self.u0[name], self._get_rhs(name), dt, 1.0/3.0])
            # u = u0 + 0.5*dt*k2
            wp.launch(rk4_update_kernel, dim=self.grid.n_points,
                      inputs=[self._get_var(name), self.u0[name], self._get_rhs(name), dt, 0.5])
        
        # k3
        self.compute_rhs()
        for name in self.var_names:
            # u_acc += (1/3)*dt*k3
            wp.launch(rk4_accumulate_kernel, dim=self.grid.n_points,
                      inputs=[self.u_acc[name], self.u0[name], self._get_rhs(name), dt, 1.0/3.0])
            # u = u0 + dt*k3
            wp.launch(rk4_update_kernel, dim=self.grid.n_points,
                      inputs=[self._get_var(name), self.u0[name], self._get_rhs(name), dt, 1.0])
        
        # k4
        self.compute_rhs()
        for name in self.var_names:
            # u_acc += (1/6)*dt*k4
            wp.launch(rk4_accumulate_kernel, dim=self.grid.n_points,
                      inputs=[self.u_acc[name], self.u0[name], self._get_rhs(name), dt, 1.0/6.0])
            # Copy final result
            wp.launch(copy_kernel, dim=self.grid.n_points,
                      inputs=[self._get_var(name), self.u_acc[name]])


def test_single_bh_evolution():
    """Test single Schwarzschild black hole evolution."""
    wp.init()
    print("=" * 60)
    print("Single Schwarzschild Black Hole Evolution Test")
    print("=" * 60)
    
    # Grid setup - use smaller domain and finer resolution
    nx, ny, nz = 48, 48, 48
    domain_size = 16.0  # 8M on each side, keep BH away from boundary
    dx = domain_size / nx
    
    print(f"\nGrid: {nx}x{ny}x{nz}")
    print(f"Domain: [-{domain_size/2:.1f}M, +{domain_size/2:.1f}M]³")
    print(f"Resolution: dx = {dx:.4f}M")
    
    # CFL condition: dt < dx / (characteristic speed)
    # For BSSN, characteristic speed ~ 1 (gauge waves)
    # Use smaller CFL for stability with puncture data
    cfl = 0.1
    dt = cfl * dx
    print(f"Time step: dt = {dt:.4f}M (CFL = {cfl})")
    
    # Create grid and initial data
    grid = BSSNGrid(nx, ny, nz, dx)
    bh_mass = 1.0
    set_schwarzschild_puncture(grid, bh_mass=bh_mass, bh_pos=(0.0, 0.0, 0.0),
                                pre_collapse_lapse=True)
    
    print(f"\nInitial data: Schwarzschild puncture, M = {bh_mass}")
    print("Gauge: 1+log slicing, Gamma-driver shift")
    
    # Evolution with higher dissipation for puncture stability
    evolver = BSSNEvolver(grid, eps_diss=0.5)
    monitor = ConstraintMonitor(grid)
    
    n_steps = 100
    report_interval = 10
    
    print(f"\nEvolving for {n_steps} steps (T = {n_steps * dt:.2f}M)...")
    print("-" * 60)
    print("Step  |  Time  |   α_min  |   α_max  |  H_L2    |  H_max")
    print("-" * 60)
    
    # Initial state
    alpha_np = grid.alpha.numpy()
    monitor.compute()
    norms = monitor.get_norms()
    print(f"    0 | {0.0:6.2f}M | {alpha_np.min():.4f} | {alpha_np.max():.4f} | "
          f"{norms['H_L2']:.2e} | {norms['H_Linf']:.2e}")
    
    times = [0.0]
    alpha_mins = [alpha_np.min()]
    alpha_maxs = [alpha_np.max()]
    H_L2s = [norms['H_L2']]
    
    for step in range(1, n_steps + 1):
        evolver.step(dt)
        t = step * dt
        
        if step % report_interval == 0 or step == n_steps:
            alpha_np = grid.alpha.numpy()
            monitor.compute()
            norms = monitor.get_norms()
            
            print(f"{step:5d} | {t:6.2f}M | {alpha_np.min():.4f} | {alpha_np.max():.4f} | "
                  f"{norms['H_L2']:.2e} | {norms['H_Linf']:.2e}")
            
            times.append(t)
            alpha_mins.append(alpha_np.min())
            alpha_maxs.append(alpha_np.max())
            H_L2s.append(norms['H_L2'])
    
    print("-" * 60)
    
    # Check stability
    final_alpha = grid.alpha.numpy()
    final_phi = grid.phi.numpy()
    
    alpha_stable = np.isfinite(final_alpha).all() and final_alpha.min() > 0
    phi_stable = np.isfinite(final_phi).all()
    
    print("\nStability check:")
    print(f"  α finite and positive: {alpha_stable}")
    print(f"  φ finite: {phi_stable}")
    
    if alpha_stable and phi_stable:
        print("\n✓ Single black hole evolution stable!")
    else:
        print("\n✗ Evolution became unstable!")
    
    # Check lapse collapse behavior
    print("\nLapse collapse analysis:")
    print(f"  Initial α_min: {alpha_mins[0]:.4f}")
    print(f"  Final α_min:   {alpha_mins[-1]:.4f}")
    
    if alpha_mins[-1] < alpha_mins[0]:
        print("  → Lapse is collapsing near the black hole (expected behavior)")
    
    return alpha_stable and phi_stable


if __name__ == "__main__":
    success = test_single_bh_evolution()
    sys.exit(0 if success else 1)
