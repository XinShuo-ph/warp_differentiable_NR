"""
RK4 time integration for BSSN evolution.
"""

import warp as wp
from bssn_variables import BSSNVariables, BSSNRHSVariables
from bssn_rhs import compute_bssn_rhs_simple


@wp.kernel
def rk4_update_kernel(
    var: wp.array3d(dtype=float),
    var_init: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    dt: float,
    stage: int  # 0, 1, 2, 3 for RK4 stages
):
    """
    RK4 update kernel.
    
    Stage 0: var = var_init + 0.5 * dt * rhs  (for k1)
    Stage 1: var = var_init + 0.5 * dt * rhs  (for k2)
    Stage 2: var = var_init + dt * rhs        (for k3)
    Stage 3: var = var_init + dt/6 * (k1 + 2*k2 + 2*k3 + k4)  (final)
    """
    i, j, k = wp.tid()
    
    if stage == 0:  # After k1, prepare for k2
        var[i, j, k] = var_init[i, j, k] + 0.5 * dt * rhs[i, j, k]
    elif stage == 1:  # After k2, prepare for k3
        var[i, j, k] = var_init[i, j, k] + 0.5 * dt * rhs[i, j, k]
    elif stage == 2:  # After k3, prepare for k4
        var[i, j, k] = var_init[i, j, k] + dt * rhs[i, j, k]


@wp.kernel
def rk4_final_kernel(
    var: wp.array3d(dtype=float),
    var_init: wp.array3d(dtype=float),
    k1: wp.array3d(dtype=float),
    k2: wp.array3d(dtype=float),
    k3: wp.array3d(dtype=float),
    k4: wp.array3d(dtype=float),
    dt: float
):
    """Final RK4 update: u^{n+1} = u^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)"""
    i, j, k = wp.tid()
    
    var[i, j, k] = var_init[i, j, k] + (dt / 6.0) * (
        k1[i, j, k] + 2.0 * k2[i, j, k] + 2.0 * k3[i, j, k] + k4[i, j, k]
    )


class BSSNEvolver:
    """BSSN evolution using RK4 time integration"""
    
    def __init__(self, nx, ny, nz, dx, dy, dz, dt, eps_diss=0.1):
        """
        Initialize BSSN evolver.
        
        Args:
            nx, ny, nz: Grid dimensions
            dx, dy, dz: Grid spacing
            dt: Time step
            eps_diss: Dissipation strength
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.eps_diss = eps_diss
        
        # Inverse grid spacing
        self.idx = 1.0 / dx
        self.idy = 1.0 / dy
        self.idz = 1.0 / dz
        
        # Current state
        self.vars = BSSNVariables(nx, ny, nz)
        
        # Storage for RK4 stages
        self.vars_init = BSSNVariables(nx, ny, nz)  # Initial state for RK step
        self.vars_temp = BSSNVariables(nx, ny, nz)  # Temp state during RK substeps
        
        # RHS storage for each stage
        self.k1 = BSSNRHSVariables(nx, ny, nz)
        self.k2 = BSSNRHSVariables(nx, ny, nz)
        self.k3 = BSSNRHSVariables(nx, ny, nz)
        self.k4 = BSSNRHSVariables(nx, ny, nz)
        
        self.time = 0.0
        self.step_count = 0
    
    def compute_rhs(self, vars, rhs):
        """Compute RHS for given state"""
        wp.launch(
            compute_bssn_rhs_simple,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                # Input variables
                vars.phi, vars.gt_xx, vars.gt_xy, vars.gt_xz, vars.gt_yy, vars.gt_yz, vars.gt_zz,
                vars.At_xx, vars.At_xy, vars.At_xz, vars.At_yy, vars.At_yz, vars.At_zz,
                vars.Gamma_x, vars.Gamma_y, vars.Gamma_z,
                vars.K, vars.alpha, vars.beta_x, vars.beta_y, vars.beta_z,
                # Output RHS
                rhs.phi_rhs, rhs.gt_xx_rhs, rhs.gt_xy_rhs, rhs.gt_xz_rhs,
                rhs.gt_yy_rhs, rhs.gt_yz_rhs, rhs.gt_zz_rhs,
                rhs.At_xx_rhs, rhs.At_xy_rhs, rhs.At_xz_rhs,
                rhs.At_yy_rhs, rhs.At_yz_rhs, rhs.At_zz_rhs,
                rhs.Gamma_x_rhs, rhs.Gamma_y_rhs, rhs.Gamma_z_rhs,
                rhs.K_rhs, rhs.alpha_rhs, rhs.beta_x_rhs, rhs.beta_y_rhs, rhs.beta_z_rhs,
                # Grid parameters
                self.idx, self.idy, self.idz, self.eps_diss
            ]
        )
    
    def rk4_step(self):
        """Perform one RK4 time step"""
        
        # Save initial state
        self.vars_init.copy_from(self.vars)
        
        # Stage 1: Compute k1 at t^n
        self.compute_rhs(self.vars, self.k1)
        
        # Update to intermediate state for k2
        for var, var_init, k1_rhs in zip(
            self.vars.get_all_vars(),
            self.vars_init.get_all_vars(),
            self.k1.get_all_vars()
        ):
            wp.launch(rk4_update_kernel, dim=var.shape,
                     inputs=[var, var_init, k1_rhs, self.dt, 0])
        
        # Stage 2: Compute k2 at t^n + dt/2
        self.compute_rhs(self.vars, self.k2)
        
        # Update to intermediate state for k3
        for var, var_init, k2_rhs in zip(
            self.vars.get_all_vars(),
            self.vars_init.get_all_vars(),
            self.k2.get_all_vars()
        ):
            wp.launch(rk4_update_kernel, dim=var.shape,
                     inputs=[var, var_init, k2_rhs, self.dt, 1])
        
        # Stage 3: Compute k3 at t^n + dt/2
        self.compute_rhs(self.vars, self.k3)
        
        # Update to intermediate state for k4
        for var, var_init, k3_rhs in zip(
            self.vars.get_all_vars(),
            self.vars_init.get_all_vars(),
            self.k3.get_all_vars()
        ):
            wp.launch(rk4_update_kernel, dim=var.shape,
                     inputs=[var, var_init, k3_rhs, self.dt, 2])
        
        # Stage 4: Compute k4 at t^n + dt
        self.compute_rhs(self.vars, self.k4)
        
        # Final update: u^{n+1} = u^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for var, var_init, k1_rhs, k2_rhs, k3_rhs, k4_rhs in zip(
            self.vars.get_all_vars(),
            self.vars_init.get_all_vars(),
            self.k1.get_all_vars(),
            self.k2.get_all_vars(),
            self.k3.get_all_vars(),
            self.k4.get_all_vars()
        ):
            wp.launch(rk4_final_kernel, dim=var.shape,
                     inputs=[var, var_init, k1_rhs, k2_rhs, k3_rhs, k4_rhs, self.dt])
        
        self.time += self.dt
        self.step_count += 1
    
    def evolve(self, num_steps):
        """Evolve for multiple time steps"""
        for step in range(num_steps):
            self.rk4_step()
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{num_steps}, t = {self.time:.4f}")
