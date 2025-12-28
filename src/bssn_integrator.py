"""
RK4 Time Integration for BSSN Evolution

Implements 4th order Runge-Kutta time stepping for the BSSN equations.
"""

import warp as wp
import numpy as np


@wp.kernel
def rk4_update_kernel(
    u: wp.array(dtype=wp.float32),      # Variable to update
    u0: wp.array(dtype=wp.float32),     # Initial value
    k: wp.array(dtype=wp.float32),      # RHS value
    dt: float,
    coeff: float                         # RK coefficient
):
    """Update u = u0 + coeff * dt * k"""
    tid = wp.tid()
    u[tid] = u0[tid] + coeff * dt * k[tid]


@wp.kernel
def rk4_accumulate_kernel(
    u_final: wp.array(dtype=wp.float32),  # Final value (accumulated)
    u0: wp.array(dtype=wp.float32),       # Initial value
    k: wp.array(dtype=wp.float32),        # RHS value
    dt: float,
    weight: float                          # RK weight (1/6, 2/6, 2/6, 1/6)
):
    """Accumulate: u_final += weight * dt * k (and initialize from u0 if weight is first)"""
    tid = wp.tid()
    u_final[tid] = u_final[tid] + weight * dt * k[tid]


@wp.kernel
def copy_kernel(dst: wp.array(dtype=wp.float32), src: wp.array(dtype=wp.float32)):
    """Copy array: dst = src"""
    tid = wp.tid()
    dst[tid] = src[tid]


class RK4Integrator:
    """
    4th order Runge-Kutta integrator for BSSN evolution.
    
    RK4 scheme:
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    
    def __init__(self, grid):
        """
        Initialize RK4 integrator with temporary storage.
        
        Args:
            grid: BSSNGrid object
        """
        self.grid = grid
        n = grid.n_points
        
        # Store initial values for each RK stage
        self.phi_0 = wp.zeros(n, dtype=wp.float32)
        self.gt11_0 = wp.zeros(n, dtype=wp.float32)
        self.gt12_0 = wp.zeros(n, dtype=wp.float32)
        self.gt13_0 = wp.zeros(n, dtype=wp.float32)
        self.gt22_0 = wp.zeros(n, dtype=wp.float32)
        self.gt23_0 = wp.zeros(n, dtype=wp.float32)
        self.gt33_0 = wp.zeros(n, dtype=wp.float32)
        self.trK_0 = wp.zeros(n, dtype=wp.float32)
        self.At11_0 = wp.zeros(n, dtype=wp.float32)
        self.At12_0 = wp.zeros(n, dtype=wp.float32)
        self.At13_0 = wp.zeros(n, dtype=wp.float32)
        self.At22_0 = wp.zeros(n, dtype=wp.float32)
        self.At23_0 = wp.zeros(n, dtype=wp.float32)
        self.At33_0 = wp.zeros(n, dtype=wp.float32)
        self.Xt1_0 = wp.zeros(n, dtype=wp.float32)
        self.Xt2_0 = wp.zeros(n, dtype=wp.float32)
        self.Xt3_0 = wp.zeros(n, dtype=wp.float32)
        self.alpha_0 = wp.zeros(n, dtype=wp.float32)
        self.beta1_0 = wp.zeros(n, dtype=wp.float32)
        self.beta2_0 = wp.zeros(n, dtype=wp.float32)
        self.beta3_0 = wp.zeros(n, dtype=wp.float32)
        
        # Accumulated result
        self.phi_acc = wp.zeros(n, dtype=wp.float32)
        self.gt11_acc = wp.zeros(n, dtype=wp.float32)
        self.gt12_acc = wp.zeros(n, dtype=wp.float32)
        self.gt13_acc = wp.zeros(n, dtype=wp.float32)
        self.gt22_acc = wp.zeros(n, dtype=wp.float32)
        self.gt23_acc = wp.zeros(n, dtype=wp.float32)
        self.gt33_acc = wp.zeros(n, dtype=wp.float32)
        self.trK_acc = wp.zeros(n, dtype=wp.float32)
        self.At11_acc = wp.zeros(n, dtype=wp.float32)
        self.At12_acc = wp.zeros(n, dtype=wp.float32)
        self.At13_acc = wp.zeros(n, dtype=wp.float32)
        self.At22_acc = wp.zeros(n, dtype=wp.float32)
        self.At23_acc = wp.zeros(n, dtype=wp.float32)
        self.At33_acc = wp.zeros(n, dtype=wp.float32)
        self.Xt1_acc = wp.zeros(n, dtype=wp.float32)
        self.Xt2_acc = wp.zeros(n, dtype=wp.float32)
        self.Xt3_acc = wp.zeros(n, dtype=wp.float32)
        self.alpha_acc = wp.zeros(n, dtype=wp.float32)
        self.beta1_acc = wp.zeros(n, dtype=wp.float32)
        self.beta2_acc = wp.zeros(n, dtype=wp.float32)
        self.beta3_acc = wp.zeros(n, dtype=wp.float32)
        
    def _save_initial(self):
        """Save initial values before RK stages."""
        n = self.grid.n_points
        wp.launch(copy_kernel, dim=n, inputs=[self.phi_0, self.grid.phi])
        wp.launch(copy_kernel, dim=n, inputs=[self.gt11_0, self.grid.gt11])
        wp.launch(copy_kernel, dim=n, inputs=[self.gt12_0, self.grid.gt12])
        wp.launch(copy_kernel, dim=n, inputs=[self.gt13_0, self.grid.gt13])
        wp.launch(copy_kernel, dim=n, inputs=[self.gt22_0, self.grid.gt22])
        wp.launch(copy_kernel, dim=n, inputs=[self.gt23_0, self.grid.gt23])
        wp.launch(copy_kernel, dim=n, inputs=[self.gt33_0, self.grid.gt33])
        wp.launch(copy_kernel, dim=n, inputs=[self.trK_0, self.grid.trK])
        wp.launch(copy_kernel, dim=n, inputs=[self.At11_0, self.grid.At11])
        wp.launch(copy_kernel, dim=n, inputs=[self.At12_0, self.grid.At12])
        wp.launch(copy_kernel, dim=n, inputs=[self.At13_0, self.grid.At13])
        wp.launch(copy_kernel, dim=n, inputs=[self.At22_0, self.grid.At22])
        wp.launch(copy_kernel, dim=n, inputs=[self.At23_0, self.grid.At23])
        wp.launch(copy_kernel, dim=n, inputs=[self.At33_0, self.grid.At33])
        wp.launch(copy_kernel, dim=n, inputs=[self.Xt1_0, self.grid.Xt1])
        wp.launch(copy_kernel, dim=n, inputs=[self.Xt2_0, self.grid.Xt2])
        wp.launch(copy_kernel, dim=n, inputs=[self.Xt3_0, self.grid.Xt3])
        wp.launch(copy_kernel, dim=n, inputs=[self.alpha_0, self.grid.alpha])
        wp.launch(copy_kernel, dim=n, inputs=[self.beta1_0, self.grid.beta1])
        wp.launch(copy_kernel, dim=n, inputs=[self.beta2_0, self.grid.beta2])
        wp.launch(copy_kernel, dim=n, inputs=[self.beta3_0, self.grid.beta3])
        
    def _update_vars(self, coeff, dt):
        """Update grid variables: u = u0 + coeff * dt * rhs."""
        n = self.grid.n_points
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.phi, self.phi_0, self.grid.phi_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.gt11, self.gt11_0, self.grid.gt11_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.gt12, self.gt12_0, self.grid.gt12_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.gt13, self.gt13_0, self.grid.gt13_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.gt22, self.gt22_0, self.grid.gt22_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.gt23, self.gt23_0, self.grid.gt23_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.gt33, self.gt33_0, self.grid.gt33_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.trK, self.trK_0, self.grid.trK_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.At11, self.At11_0, self.grid.At11_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.At12, self.At12_0, self.grid.At12_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.At13, self.At13_0, self.grid.At13_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.At22, self.At22_0, self.grid.At22_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.At23, self.At23_0, self.grid.At23_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.At33, self.At33_0, self.grid.At33_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.Xt1, self.Xt1_0, self.grid.Xt1_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.Xt2, self.Xt2_0, self.grid.Xt2_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.Xt3, self.Xt3_0, self.grid.Xt3_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.alpha, self.alpha_0, self.grid.alpha_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.beta1, self.beta1_0, self.grid.beta1_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.beta2, self.beta2_0, self.grid.beta2_rhs, dt, coeff])
        wp.launch(rk4_update_kernel, dim=n, inputs=[self.grid.beta3, self.beta3_0, self.grid.beta3_rhs, dt, coeff])
        
    def _accumulate(self, weight, dt, first=False):
        """Accumulate weighted RHS contribution."""
        n = self.grid.n_points
        if first:
            # Initialize accumulators with u0
            wp.launch(copy_kernel, dim=n, inputs=[self.phi_acc, self.phi_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.gt11_acc, self.gt11_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.gt12_acc, self.gt12_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.gt13_acc, self.gt13_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.gt22_acc, self.gt22_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.gt23_acc, self.gt23_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.gt33_acc, self.gt33_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.trK_acc, self.trK_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.At11_acc, self.At11_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.At12_acc, self.At12_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.At13_acc, self.At13_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.At22_acc, self.At22_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.At23_acc, self.At23_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.At33_acc, self.At33_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.Xt1_acc, self.Xt1_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.Xt2_acc, self.Xt2_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.Xt3_acc, self.Xt3_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.alpha_acc, self.alpha_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.beta1_acc, self.beta1_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.beta2_acc, self.beta2_0])
            wp.launch(copy_kernel, dim=n, inputs=[self.beta3_acc, self.beta3_0])
        
        # Add weighted contribution
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.phi_acc, self.phi_0, self.grid.phi_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.gt11_acc, self.gt11_0, self.grid.gt11_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.gt12_acc, self.gt12_0, self.grid.gt12_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.gt13_acc, self.gt13_0, self.grid.gt13_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.gt22_acc, self.gt22_0, self.grid.gt22_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.gt23_acc, self.gt23_0, self.grid.gt23_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.gt33_acc, self.gt33_0, self.grid.gt33_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.trK_acc, self.trK_0, self.grid.trK_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.At11_acc, self.At11_0, self.grid.At11_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.At12_acc, self.At12_0, self.grid.At12_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.At13_acc, self.At13_0, self.grid.At13_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.At22_acc, self.At22_0, self.grid.At22_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.At23_acc, self.At23_0, self.grid.At23_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.At33_acc, self.At33_0, self.grid.At33_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.Xt1_acc, self.Xt1_0, self.grid.Xt1_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.Xt2_acc, self.Xt2_0, self.grid.Xt2_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.Xt3_acc, self.Xt3_0, self.grid.Xt3_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.alpha_acc, self.alpha_0, self.grid.alpha_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.beta1_acc, self.beta1_0, self.grid.beta1_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.beta2_acc, self.beta2_0, self.grid.beta2_rhs, dt, weight])
        wp.launch(rk4_accumulate_kernel, dim=n, inputs=[self.beta3_acc, self.beta3_0, self.grid.beta3_rhs, dt, weight])
        
    def _copy_acc_to_grid(self):
        """Copy accumulated result to grid variables."""
        n = self.grid.n_points
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.phi, self.phi_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.gt11, self.gt11_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.gt12, self.gt12_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.gt13, self.gt13_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.gt22, self.gt22_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.gt23, self.gt23_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.gt33, self.gt33_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.trK, self.trK_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.At11, self.At11_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.At12, self.At12_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.At13, self.At13_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.At22, self.At22_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.At23, self.At23_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.At33, self.At33_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.Xt1, self.Xt1_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.Xt2, self.Xt2_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.Xt3, self.Xt3_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.alpha, self.alpha_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.beta1, self.beta1_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.beta2, self.beta2_acc])
        wp.launch(copy_kernel, dim=n, inputs=[self.grid.beta3, self.beta3_acc])

    def step(self, dt, compute_rhs_func):
        """
        Take one RK4 time step.
        
        Args:
            dt: Time step size
            compute_rhs_func: Function that computes RHS (takes grid as argument)
        """
        # Save initial state
        self._save_initial()
        
        # Stage 1: k1 = f(t, y)
        compute_rhs_func(self.grid)
        self._accumulate(1.0/6.0, dt, first=True)
        
        # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        self._update_vars(0.5, dt)
        compute_rhs_func(self.grid)
        self._accumulate(2.0/6.0, dt)
        
        # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        self._update_vars(0.5, dt)
        compute_rhs_func(self.grid)
        self._accumulate(2.0/6.0, dt)
        
        # Stage 4: k4 = f(t + dt, y + dt * k3)
        self._update_vars(1.0, dt)
        compute_rhs_func(self.grid)
        self._accumulate(1.0/6.0, dt)
        
        # Copy final result to grid
        self._copy_acc_to_grid()


def test_rk4_integrator():
    """Test RK4 integrator with flat spacetime evolution."""
    import sys
    sys.path.insert(0, '/workspace/src')
    from bssn_vars import BSSNGrid
    from bssn_rhs import compute_bssn_rhs_kernel
    
    wp.init()
    print("=== RK4 Integrator Test (Flat Spacetime) ===\n")
    
    nx, ny, nz = 20, 20, 20
    dx = 0.1
    dt = 0.01  # CFL condition: dt < dx/c, here c ~ 1
    
    grid = BSSNGrid(nx, ny, nz, dx)
    grid.set_flat_spacetime()
    
    integrator = RK4Integrator(grid)
    inv_dx = 1.0 / dx
    eps_diss = 0.2 * dx  # Kreiss-Oliger dissipation coefficient
    
    def compute_rhs(g):
        wp.launch(
            compute_bssn_rhs_kernel,
            dim=g.n_points,
            inputs=[
                g.phi, g.gt11, g.gt12, g.gt13, g.gt22, g.gt23, g.gt33,
                g.trK, g.At11, g.At12, g.At13, g.At22, g.At23, g.At33,
                g.Xt1, g.Xt2, g.Xt3,
                g.alpha, g.beta1, g.beta2, g.beta3,
                g.phi_rhs, g.gt11_rhs, g.gt12_rhs, g.gt13_rhs,
                g.gt22_rhs, g.gt23_rhs, g.gt33_rhs,
                g.trK_rhs, g.At11_rhs, g.At12_rhs, g.At13_rhs,
                g.At22_rhs, g.At23_rhs, g.At33_rhs,
                g.Xt1_rhs, g.Xt2_rhs, g.Xt3_rhs,
                g.alpha_rhs, g.beta1_rhs, g.beta2_rhs, g.beta3_rhs,
                nx, ny, nz, inv_dx, eps_diss
            ]
        )
    
    # Initial values
    alpha_init = grid.alpha.numpy().mean()
    phi_init = grid.phi.numpy().mean()
    gt11_init = grid.gt11.numpy().mean()
    
    print(f"Initial: alpha={alpha_init:.6f}, phi={phi_init:.6f}, gt11={gt11_init:.6f}")
    
    # Evolve for 100 timesteps
    n_steps = 100
    for step in range(n_steps):
        integrator.step(dt, compute_rhs)
    
    # Check final values (should remain flat)
    alpha_final = grid.alpha.numpy().mean()
    phi_final = grid.phi.numpy().mean()
    gt11_final = grid.gt11.numpy().mean()
    
    print(f"After {n_steps} steps (t={n_steps*dt:.2f}):")
    print(f"  alpha: {alpha_final:.6f} (change: {alpha_final-alpha_init:.6e})")
    print(f"  phi:   {phi_final:.6f} (change: {phi_final-phi_init:.6e})")
    print(f"  gt11:  {gt11_final:.6f} (change: {gt11_final-gt11_init:.6e})")
    
    # Check det(gt) = 1 is preserved
    gt11 = grid.gt11.numpy()
    gt22 = grid.gt22.numpy()
    gt33 = grid.gt33.numpy()
    gt12 = grid.gt12.numpy()
    gt13 = grid.gt13.numpy()
    gt23 = grid.gt23.numpy()
    
    det = (gt11 * (gt22 * gt33 - gt23**2)
           - gt12 * (gt12 * gt33 - gt23 * gt13)
           + gt13 * (gt12 * gt23 - gt22 * gt13))
    
    print(f"  det(gt): {det.mean():.6f} (should be 1)")
    
    print("\n✓ RK4 integrator test completed.")


if __name__ == "__main__":
    test_rk4_integrator()
