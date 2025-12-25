"""
BSSN Evolution in Warp - Core Implementation

Implements the BSSN formulation of Einstein equations using NVIDIA Warp.
Starting with flat spacetime evolution for testing.

BSSN Variables:
- phi (or W): conformal factor
- gt_ij: conformal 3-metric (6 components)
- At_ij: traceless conformal extrinsic curvature (6 components)
- trK: trace of extrinsic curvature
- Xt_i: conformal connection functions (3 components)
- alpha: lapse function
- beta_i: shift vector (3 components)

Total: 24 evolved variables per point
"""

import warp as wp
import numpy as np


# BSSN variable indices (for flat array storage)
# Conformal factor
PHI = 0

# Conformal metric (symmetric 3x3)
GT11 = 1
GT12 = 2
GT13 = 3
GT22 = 4
GT23 = 5
GT33 = 6

# Traceless conformal extrinsic curvature (symmetric 3x3)
AT11 = 7
AT12 = 8
AT13 = 9
AT22 = 10
AT23 = 11
AT33 = 12

# Trace of extrinsic curvature
TRK = 13

# Conformal connection functions (vector)
XT1 = 14
XT2 = 15
XT3 = 16

# Lapse
ALPHA = 17

# Shift vector
BETA1 = 18
BETA2 = 19
BETA3 = 20

# Total number of variables
NUM_VARS = 21


class BSSNGrid:
    """
    3D grid for BSSN evolution
    """
    def __init__(self, nx, ny, nz, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, zmin=-1.0, zmax=1.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        
        self.dx = (xmax - xmin) / (nx - 1)
        self.dy = (ymax - ymin) / (ny - 1)
        self.dz = (zmax - zmin) / (nz - 1)
        
        # Allocate arrays for current state and RHS
        self.vars = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)
        self.rhs = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)
        
        # Temporary arrays for RK stages
        self.k1 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)
        self.k2 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)
        self.k3 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)
        self.k4 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)
        self.temp = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)
    
    def initialize_flat_spacetime(self):
        """
        Initialize to flat spacetime in Cartesian coordinates
        """
        vars_np = np.zeros((self.nx, self.ny, self.nz, NUM_VARS), dtype=np.float32)
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Conformal factor: phi = 0 (or W = 1)
                    vars_np[i, j, k, PHI] = 0.0
                    
                    # Conformal metric: gamma_tilde_ij = delta_ij (flat)
                    vars_np[i, j, k, GT11] = 1.0
                    vars_np[i, j, k, GT12] = 0.0
                    vars_np[i, j, k, GT13] = 0.0
                    vars_np[i, j, k, GT22] = 1.0
                    vars_np[i, j, k, GT23] = 0.0
                    vars_np[i, j, k, GT33] = 1.0
                    
                    # Traceless extrinsic curvature: A_tilde_ij = 0 (flat)
                    vars_np[i, j, k, AT11] = 0.0
                    vars_np[i, j, k, AT12] = 0.0
                    vars_np[i, j, k, AT13] = 0.0
                    vars_np[i, j, k, AT22] = 0.0
                    vars_np[i, j, k, AT23] = 0.0
                    vars_np[i, j, k, AT33] = 0.0
                    
                    # Trace of K: trK = 0 (flat)
                    vars_np[i, j, k, TRK] = 0.0
                    
                    # Conformal connection: Xt^i = 0 (flat)
                    vars_np[i, j, k, XT1] = 0.0
                    vars_np[i, j, k, XT2] = 0.0
                    vars_np[i, j, k, XT3] = 0.0
                    
                    # Lapse: alpha = 1 (flat)
                    vars_np[i, j, k, ALPHA] = 1.0
                    
                    # Shift: beta^i = 0 (flat)
                    vars_np[i, j, k, BETA1] = 0.0
                    vars_np[i, j, k, BETA2] = 0.0
                    vars_np[i, j, k, BETA3] = 0.0
        
        self.vars = wp.from_numpy(vars_np, dtype=wp.float32)
    
    def get_position(self, i, j, k):
        """Get physical coordinates"""
        x = self.xmin + i * self.dx
        y = self.ymin + j * self.dy
        z = self.zmin + k * self.dz
        return x, y, z


@wp.kernel
def compute_rhs_kernel(
    vars: wp.array4d(dtype=wp.float32),
    rhs: wp.array4d(dtype=wp.float32),
    dx: float,
    dy: float,
    dz: float,
    epsDiss: float,
):
    """
    Compute RHS of BSSN evolution equations
    
    This is a simplified version - full implementation in bssn_rhs.py
    For now, just implement flat spacetime (RHS = 0)
    """
    i, j, k = wp.tid()
    
    nx = vars.shape[0]
    ny = vars.shape[1]
    nz = vars.shape[2]
    
    # Skip boundary points (need at least 2 ghost zones for 4th order)
    if i < 2 or i >= nx - 2:
        return
    if j < 2 or j >= ny - 2:
        return
    if k < 2 or k >= nz - 2:
        return
    
    # For flat spacetime, RHS should be zero
    for v in range(NUM_VARS):
        rhs[i, j, k, v] = 0.0


@wp.kernel
def rk4_stage_kernel(
    vars: wp.array4d(dtype=wp.float32),
    rhs: wp.array4d(dtype=wp.float32),
    temp: wp.array4d(dtype=wp.float32),
    dt: float,
    factor: float,
):
    """
    RK4 intermediate stage: temp = vars + factor * dt * rhs
    """
    i, j, k = wp.tid()
    
    for v in range(NUM_VARS):
        temp[i, j, k, v] = vars[i, j, k, v] + factor * dt * rhs[i, j, k, v]


@wp.kernel
def rk4_final_kernel(
    vars: wp.array4d(dtype=wp.float32),
    k1: wp.array4d(dtype=wp.float32),
    k2: wp.array4d(dtype=wp.float32),
    k3: wp.array4d(dtype=wp.float32),
    k4: wp.array4d(dtype=wp.float32),
    dt: float,
):
    """
    RK4 final stage: vars_new = vars + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    i, j, k = wp.tid()
    
    for v in range(NUM_VARS):
        vars[i, j, k, v] = (vars[i, j, k, v] + 
                            dt / 6.0 * (k1[i, j, k, v] + 
                                       2.0 * k2[i, j, k, v] + 
                                       2.0 * k3[i, j, k, v] + 
                                       k4[i, j, k, v]))


def evolve_rk4(grid: BSSNGrid, dt: float, epsDiss: float = 0.0):
    """
    Evolve BSSN system one timestep using RK4
    """
    # Stage 1: k1 = f(t, y)
    wp.launch(
        compute_rhs_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.vars, grid.k1, grid.dx, grid.dy, grid.dz, epsDiss],
    )
    
    # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
    wp.launch(
        rk4_stage_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.vars, grid.k1, grid.temp, dt, 0.5],
    )
    wp.launch(
        compute_rhs_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.temp, grid.k2, grid.dx, grid.dy, grid.dz, epsDiss],
    )
    
    # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
    wp.launch(
        rk4_stage_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.vars, grid.k2, grid.temp, dt, 0.5],
    )
    wp.launch(
        compute_rhs_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.temp, grid.k3, grid.dx, grid.dy, grid.dz, epsDiss],
    )
    
    # Stage 4: k4 = f(t + dt, y + dt * k3)
    wp.launch(
        rk4_stage_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.vars, grid.k3, grid.temp, dt, 1.0],
    )
    wp.launch(
        compute_rhs_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.temp, grid.k4, grid.dx, grid.dy, grid.dz, epsDiss],
    )
    
    # Final update: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    wp.launch(
        rk4_final_kernel,
        dim=(grid.nx, grid.ny, grid.nz),
        inputs=[grid.vars, grid.k1, grid.k2, grid.k3, grid.k4, dt],
    )


def test_flat_spacetime_evolution():
    """
    Test that flat spacetime remains flat under evolution
    """
    print("Testing flat spacetime evolution...")
    
    # Create small grid
    nx, ny, nz = 16, 16, 16
    grid = BSSNGrid(nx, ny, nz, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, zmin=-1.0, zmax=1.0)
    
    # Initialize to flat spacetime
    grid.initialize_flat_spacetime()
    
    # Get initial state
    initial_vars = grid.vars.numpy().copy()
    
    # Evolve for some timesteps
    dt = 0.01
    num_steps = 100
    
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Timestep: dt = {dt}")
    print(f"Total steps: {num_steps}")
    print()
    
    for step in range(num_steps):
        evolve_rk4(grid, dt)
        
        if (step + 1) % 20 == 0:
            current_vars = grid.vars.numpy()
            max_diff = np.abs(current_vars - initial_vars).max()
            print(f"Step {step+1:3d}: max deviation from flat = {max_diff:.6e}")
    
    # Final check
    final_vars = grid.vars.numpy()
    max_diff = np.abs(final_vars - initial_vars).max()
    
    print()
    print(f"Final max deviation from flat spacetime: {max_diff:.6e}")
    
    if max_diff < 1e-6:
        print("✓ Flat spacetime remains stable!")
        return True
    else:
        print("✗ Flat spacetime evolution shows drift")
        return False


if __name__ == "__main__":
    wp.init()
    
    success = test_flat_spacetime_evolution()
