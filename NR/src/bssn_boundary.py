"""
BSSN Boundary Conditions

Implements Sommerfeld (radiative) boundary conditions for BSSN evolution.
These allow outgoing waves to leave the domain without reflection.
"""

import warp as wp
import numpy as np


@wp.kernel
def apply_sommerfeld_bc_kernel(
    # Variable and its RHS
    u: wp.array(dtype=wp.float32),
    u_rhs: wp.array(dtype=wp.float32),
    # Asymptotic value
    u0: float,
    # Wave speed (typically 1)
    v: float,
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float
):
    """
    Apply Sommerfeld radiative boundary condition.
    
    At the boundary: ∂ₜu = -v * ∂ᵣu - (u - u0) / r
    
    This condition allows outgoing waves to propagate out of the domain.
    """
    tid = wp.tid()
    
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Only apply at boundary points
    at_boundary = (i < 3 or i >= nx - 3 or 
                   j < 3 or j >= ny - 3 or 
                   k < 3 or k >= nz - 3)
    
    if not at_boundary:
        return
    
    # Grid coordinates (cell-centered)
    x = (float(i) + 0.5) * dx - float(nx) * dx / 2.0
    y = (float(j) + 0.5) * dx - float(ny) * dx / 2.0
    z = (float(k) + 0.5) * dx - float(nz) * dx / 2.0
    
    r = wp.sqrt(x*x + y*y + z*z)
    r = wp.max(r, dx)  # Avoid division by zero
    
    # Radial direction
    nx_r = x / r
    ny_r = y / r
    nz_r = z / r
    
    # Compute radial derivative using one-sided difference
    u_L = u[tid]
    
    # Get interior neighbor for radial derivative
    # Find direction towards interior
    di = 0
    dj = 0
    dk = 0
    if i < 3:
        di = 1
    elif i >= nx - 3:
        di = -1
    if j < 3:
        dj = 1
    elif j >= ny - 3:
        dj = -1
    if k < 3:
        dk = 1
    elif k >= nz - 3:
        dk = -1
    
    # Interior neighbor index
    i_int = i + di
    j_int = j + dj
    k_int = k + dk
    
    # Clamp to valid range
    i_int = wp.max(0, wp.min(i_int, nx - 1))
    j_int = wp.max(0, wp.min(j_int, ny - 1))
    k_int = wp.max(0, wp.min(k_int, nz - 1))
    
    tid_int = i_int + nx * (j_int + ny * k_int)
    u_int = u[tid_int]
    
    # Radial derivative (one-sided)
    # Distance to interior point
    dr = wp.sqrt(float(di*di + dj*dj + dk*dk)) * dx
    dr = wp.max(dr, 0.1 * dx)
    
    # ∂u/∂r ≈ (u - u_int) / dr (outward derivative)
    du_dr = (u_L - u_int) / dr
    
    # Sommerfeld condition: ∂ₜu = -v * ∂ᵣu - (u - u0) / r
    u_rhs[tid] = -v * du_dr - (u_L - u0) / r


def apply_sommerfeld_boundaries(grid, var_info):
    """
    Apply Sommerfeld boundary conditions to all variables.
    
    Args:
        grid: BSSNGrid object
        var_info: List of (variable, rhs, u0, v) tuples
                  - variable: wp.array of the field
                  - rhs: wp.array of the RHS
                  - u0: asymptotic value (e.g., 0 for phi, 1 for alpha)
                  - v: wave speed (typically 1.0)
    """
    for var, rhs, u0, v in var_info:
        wp.launch(
            apply_sommerfeld_bc_kernel,
            dim=grid.n_points,
            inputs=[var, rhs, u0, v, grid.nx, grid.ny, grid.nz, grid.dx]
        )


def apply_standard_bssn_boundaries(grid):
    """
    Apply standard Sommerfeld BCs to all BSSN variables.
    
    Asymptotic values:
    - phi → 0 (flat spacetime)
    - gt_ij → δ_ij (flat conformal metric)
    - trK → 0
    - At_ij → 0
    - Xt^i → 0
    - alpha → 1
    - beta^i → 0
    """
    v = 1.0  # Wave speed
    
    var_info = [
        (grid.phi, grid.phi_rhs, 0.0, v),
        (grid.gt11, grid.gt11_rhs, 1.0, v),
        (grid.gt12, grid.gt12_rhs, 0.0, v),
        (grid.gt13, grid.gt13_rhs, 0.0, v),
        (grid.gt22, grid.gt22_rhs, 1.0, v),
        (grid.gt23, grid.gt23_rhs, 0.0, v),
        (grid.gt33, grid.gt33_rhs, 1.0, v),
        (grid.trK, grid.trK_rhs, 0.0, v),
        (grid.At11, grid.At11_rhs, 0.0, v),
        (grid.At12, grid.At12_rhs, 0.0, v),
        (grid.At13, grid.At13_rhs, 0.0, v),
        (grid.At22, grid.At22_rhs, 0.0, v),
        (grid.At23, grid.At23_rhs, 0.0, v),
        (grid.At33, grid.At33_rhs, 0.0, v),
        (grid.Xt1, grid.Xt1_rhs, 0.0, v),
        (grid.Xt2, grid.Xt2_rhs, 0.0, v),
        (grid.Xt3, grid.Xt3_rhs, 0.0, v),
        (grid.alpha, grid.alpha_rhs, 1.0, v),
        (grid.beta1, grid.beta1_rhs, 0.0, v),
        (grid.beta2, grid.beta2_rhs, 0.0, v),
        (grid.beta3, grid.beta3_rhs, 0.0, v),
    ]
    
    apply_sommerfeld_boundaries(grid, var_info)


def test_boundary_conditions():
    """Test Sommerfeld boundary conditions."""
    import sys
    sys.path.insert(0, '/workspace/NR/src')
    from bssn_vars import BSSNGrid
    from bssn_initial_data import set_schwarzschild_puncture
    from bssn_rhs_full import compute_bssn_rhs_full_kernel
    
    wp.init()
    print("=== Sommerfeld Boundary Conditions Test ===\n")
    
    nx, ny, nz = 32, 32, 32
    domain_size = 20.0
    dx = domain_size / nx
    
    grid = BSSNGrid(nx, ny, nz, dx)
    set_schwarzschild_puncture(grid, bh_mass=1.0, bh_pos=(0.0, 0.0, 0.0),
                                pre_collapse_lapse=True)
    
    inv_dx = 1.0 / dx
    eps_diss = 0.2 * dx
    
    # Compute RHS
    wp.launch(
        compute_bssn_rhs_full_kernel,
        dim=grid.n_points,
        inputs=[
            grid.phi, grid.gt11, grid.gt12, grid.gt13, grid.gt22, grid.gt23, grid.gt33,
            grid.trK, grid.At11, grid.At12, grid.At13, grid.At22, grid.At23, grid.At33,
            grid.Xt1, grid.Xt2, grid.Xt3,
            grid.alpha, grid.beta1, grid.beta2, grid.beta3,
            grid.phi_rhs, grid.gt11_rhs, grid.gt12_rhs, grid.gt13_rhs,
            grid.gt22_rhs, grid.gt23_rhs, grid.gt33_rhs,
            grid.trK_rhs, grid.At11_rhs, grid.At12_rhs, grid.At13_rhs,
            grid.At22_rhs, grid.At23_rhs, grid.At33_rhs,
            grid.Xt1_rhs, grid.Xt2_rhs, grid.Xt3_rhs,
            grid.alpha_rhs, grid.beta1_rhs, grid.beta2_rhs, grid.beta3_rhs,
            nx, ny, nz, inv_dx, eps_diss
        ]
    )
    
    print("Before boundary conditions:")
    alpha_rhs_np = grid.alpha_rhs.numpy().reshape(nx, ny, nz)
    print(f"  alpha_rhs at corner (0,0,0): {alpha_rhs_np[0,0,0]:.6f}")
    print(f"  alpha_rhs at corner (nx-1,0,0): {alpha_rhs_np[nx-1,0,0]:.6f}")
    
    # Apply boundary conditions
    apply_standard_bssn_boundaries(grid)
    
    print("\nAfter Sommerfeld boundary conditions:")
    alpha_rhs_np = grid.alpha_rhs.numpy().reshape(nx, ny, nz)
    print(f"  alpha_rhs at corner (0,0,0): {alpha_rhs_np[0,0,0]:.6f}")
    print(f"  alpha_rhs at corner (nx-1,0,0): {alpha_rhs_np[nx-1,0,0]:.6f}")
    
    # Check that boundary values are non-zero (Sommerfeld should set them)
    boundary_rhs_max = max(
        abs(alpha_rhs_np[0, :, :]).max(),
        abs(alpha_rhs_np[nx-1, :, :]).max(),
        abs(alpha_rhs_np[:, 0, :]).max(),
        abs(alpha_rhs_np[:, ny-1, :]).max(),
        abs(alpha_rhs_np[:, :, 0]).max(),
        abs(alpha_rhs_np[:, :, nz-1]).max()
    )
    print(f"\nMax boundary alpha_rhs: {boundary_rhs_max:.6e}")
    
    print("\n✓ Sommerfeld boundary conditions test completed.")


if __name__ == "__main__":
    test_boundary_conditions()
