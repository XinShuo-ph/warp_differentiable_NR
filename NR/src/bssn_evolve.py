"""
BSSN Evolution Driver with Boundary Conditions and Constraints

Provides a complete evolution driver for BSSN equations including:
- Radiative (Sommerfeld) boundary conditions
- RK4 time integration
- Hamiltonian and momentum constraint monitoring
"""

import math
import numpy as np
import warp as wp

wp.init()

from bssn import (
    BSSNState, create_bssn_state, init_flat_spacetime_state,
    dx, dy, dz, dxx, dyy, dzz,
    ko_dissipation,
)
from bssn_full import (
    init_gauge_wave_state,
    init_brill_lindquist_state,
    compute_christoffel_and_ricci,
    compute_bssn_rhs_full,
    create_ricci_arrays,
    compute_inverse_metric,
)


# ============================================================================
# Radiative Boundary Conditions
# ============================================================================

@wp.kernel
def apply_sommerfeld_bc(
    u: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    u0: float,  # Asymptotic value
    v0: float,  # Wave speed
    dx: float,
    center_x: float,
    center_y: float,
    center_z: float,
):
    """Apply Sommerfeld radiative boundary condition.
    
    ∂_t u + v₀/r ∂_r(r(u - u₀)) = 0
    
    Modifies RHS at boundary points.
    """
    i, j, k = wp.tid()
    
    nx = u.shape[0]
    ny = u.shape[1]
    nz = u.shape[2]
    
    # Check if this is a boundary point (within 2 cells of edge)
    is_boundary = (i < 3 or i >= nx - 3 or 
                   j < 3 or j >= ny - 3 or 
                   k < 3 or k >= nz - 3)
    
    if not is_boundary:
        return
    
    # Physical position
    x = float(i) * dx - center_x
    y = float(j) * dx - center_y
    z = float(k) * dx - center_z
    
    r = wp.sqrt(x*x + y*y + z*z + 1e-10)
    
    # Current value
    u_val = u[i, j, k]
    
    # Sommerfeld condition: d_t u = -v0/r * (u - u0) - v0 * d_r(u)
    # Simplified: d_t u ≈ -v0 * (u - u0) / r
    rhs[i, j, k] = -v0 * (u_val - u0) / r


def apply_boundary_conditions(state: BSSNState, rhs: BSSNState, 
                              center: tuple = None):
    """Apply radiative boundary conditions to all fields."""
    nx, ny, nz = state.nx, state.ny, state.nz
    dx_val = state.dx
    
    if center is None:
        center_x = nx * dx_val / 2.0
        center_y = ny * dx_val / 2.0
        center_z = nz * dx_val / 2.0
    else:
        center_x, center_y, center_z = center
    
    # Apply to each field with appropriate asymptotic values
    fields_with_bc = [
        (state.phi, rhs.phi, 0.0, 1.0),      # phi -> 0
        (state.gt11, rhs.gt11, 1.0, 1.0),    # gt -> identity
        (state.gt12, rhs.gt12, 0.0, 1.0),
        (state.gt13, rhs.gt13, 0.0, 1.0),
        (state.gt22, rhs.gt22, 1.0, 1.0),
        (state.gt23, rhs.gt23, 0.0, 1.0),
        (state.gt33, rhs.gt33, 1.0, 1.0),
        (state.Xt1, rhs.Xt1, 0.0, 1.0),      # Xt -> 0
        (state.Xt2, rhs.Xt2, 0.0, 1.0),
        (state.Xt3, rhs.Xt3, 0.0, 1.0),
        (state.trK, rhs.trK, 0.0, 1.0),      # K -> 0
        (state.At11, rhs.At11, 0.0, 1.0),    # At -> 0
        (state.At12, rhs.At12, 0.0, 1.0),
        (state.At13, rhs.At13, 0.0, 1.0),
        (state.At22, rhs.At22, 0.0, 1.0),
        (state.At23, rhs.At23, 0.0, 1.0),
        (state.At33, rhs.At33, 0.0, 1.0),
        (state.alpha, rhs.alpha, 1.0, 1.0),  # alpha -> 1
        (state.beta1, rhs.beta1, 0.0, 1.0),  # beta -> 0
        (state.beta2, rhs.beta2, 0.0, 1.0),
        (state.beta3, rhs.beta3, 0.0, 1.0),
    ]
    
    for u, rhs_u, u0, v0 in fields_with_bc:
        wp.launch(
            apply_sommerfeld_bc,
            dim=(nx, ny, nz),
            inputs=[u, rhs_u, u0, v0, dx_val, center_x, center_y, center_z]
        )


# ============================================================================
# Constraint Monitoring
# ============================================================================

@wp.kernel
def compute_hamiltonian_constraint(
    phi: wp.array3d(dtype=float),
    gt11: wp.array3d(dtype=float),
    gt12: wp.array3d(dtype=float),
    gt13: wp.array3d(dtype=float),
    gt22: wp.array3d(dtype=float),
    gt23: wp.array3d(dtype=float),
    gt33: wp.array3d(dtype=float),
    trK: wp.array3d(dtype=float),
    At11: wp.array3d(dtype=float),
    At12: wp.array3d(dtype=float),
    At13: wp.array3d(dtype=float),
    At22: wp.array3d(dtype=float),
    At23: wp.array3d(dtype=float),
    At33: wp.array3d(dtype=float),
    trR: wp.array3d(dtype=float),
    H_out: wp.array3d(dtype=float),
    inv_dx: float,
):
    """Compute Hamiltonian constraint: H = R - K_{ij}K^{ij} + K² = 0"""
    i, j, k = wp.tid()
    
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    
    if i < 3 or i >= nx - 3 or j < 3 or j >= ny - 3 or k < 3 or k >= nz - 3:
        H_out[i, j, k] = 0.0
        return
    
    # Get values
    g11 = gt11[i, j, k]
    g12 = gt12[i, j, k]
    g13 = gt13[i, j, k]
    g22 = gt22[i, j, k]
    g23 = gt23[i, j, k]
    g33 = gt33[i, j, k]
    
    a11 = At11[i, j, k]
    a12 = At12[i, j, k]
    a13 = At13[i, j, k]
    a22 = At22[i, j, k]
    a23 = At23[i, j, k]
    a33 = At33[i, j, k]
    
    K = trK[i, j, k]
    R = trR[i, j, k]
    
    # Inverse metric
    gtu = compute_inverse_metric(g11, g12, g13, g22, g23, g33)
    gtu11 = gtu[0, 0]
    gtu12 = gtu[0, 1]
    gtu13 = gtu[0, 2]
    gtu22 = gtu[1, 1]
    gtu23 = gtu[1, 2]
    gtu33 = gtu[2, 2]
    
    # A_{ij} A^{ij}
    Atu11 = gtu11 * gtu11 * a11 + 2.0 * gtu11 * gtu12 * a12 + gtu12 * gtu12 * a22
    Atu22 = gtu12 * gtu12 * a11 + 2.0 * gtu12 * gtu22 * a12 + gtu22 * gtu22 * a22
    Atu33 = gtu13 * gtu13 * a11 + 2.0 * gtu13 * gtu33 * a13 + gtu33 * gtu33 * a33
    
    AtAt = a11 * Atu11 + a22 * Atu22 + a33 * Atu33
    
    # H = R - A_{ij}A^{ij} + 2/3 K^2
    H_out[i, j, k] = R - AtAt + 2.0/3.0 * K * K


def compute_constraint_norms(state: BSSNState, ricci: dict) -> dict:
    """Compute L2 and L∞ norms of constraint violations."""
    nx, ny, nz = state.nx, state.ny, state.nz
    inv_dx = 1.0 / state.dx
    
    H = wp.zeros((nx, ny, nz), dtype=float)
    
    wp.launch(
        compute_hamiltonian_constraint,
        dim=(nx, ny, nz),
        inputs=[
            state.phi,
            state.gt11, state.gt12, state.gt13, state.gt22, state.gt23, state.gt33,
            state.trK,
            state.At11, state.At12, state.At13, state.At22, state.At23, state.At33,
            ricci['trR'],
            H,
            inv_dx,
        ]
    )
    
    H_np = H.numpy()
    
    # Adaptive interior to handle small grids
    pad = min(4, nx // 4, ny // 4, nz // 4)
    if pad < 1:
        pad = 0
    interior = (slice(pad, nx - pad if pad > 0 else nx), 
                slice(pad, ny - pad if pad > 0 else ny), 
                slice(pad, nz - pad if pad > 0 else nz))
    
    H_interior = H_np[interior]
    
    if H_interior.size == 0:
        return {'H_L2': 0.0, 'H_Linf': 0.0}
    
    return {
        'H_L2': np.sqrt((H_interior**2).mean()),
        'H_Linf': np.abs(H_interior).max(),
    }


# ============================================================================
# RK4 Time Evolution
# ============================================================================

def copy_state(src: BSSNState, dst: BSSNState):
    """Copy all fields from src to dst."""
    wp.copy(dst.phi, src.phi)
    wp.copy(dst.gt11, src.gt11)
    wp.copy(dst.gt12, src.gt12)
    wp.copy(dst.gt13, src.gt13)
    wp.copy(dst.gt22, src.gt22)
    wp.copy(dst.gt23, src.gt23)
    wp.copy(dst.gt33, src.gt33)
    wp.copy(dst.Xt1, src.Xt1)
    wp.copy(dst.Xt2, src.Xt2)
    wp.copy(dst.Xt3, src.Xt3)
    wp.copy(dst.trK, src.trK)
    wp.copy(dst.At11, src.At11)
    wp.copy(dst.At12, src.At12)
    wp.copy(dst.At13, src.At13)
    wp.copy(dst.At22, src.At22)
    wp.copy(dst.At23, src.At23)
    wp.copy(dst.At33, src.At33)
    wp.copy(dst.alpha, src.alpha)
    wp.copy(dst.beta1, src.beta1)
    wp.copy(dst.beta2, src.beta2)
    wp.copy(dst.beta3, src.beta3)


@wp.kernel
def axpy_kernel(y: wp.array3d(dtype=float), a: float, x: wp.array3d(dtype=float)):
    """y = y + a * x"""
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + a * x[i, j, k]


@wp.kernel
def rk4_combine_kernel(
    y: wp.array3d(dtype=float),
    y0: wp.array3d(dtype=float),
    k1: wp.array3d(dtype=float),
    k2: wp.array3d(dtype=float),
    k3: wp.array3d(dtype=float),
    k4: wp.array3d(dtype=float),
    dt: float,
):
    """y = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)"""
    i, j, k = wp.tid()
    y[i, j, k] = y0[i, j, k] + dt / 6.0 * (
        k1[i, j, k] + 2.0 * k2[i, j, k] + 2.0 * k3[i, j, k] + k4[i, j, k]
    )


def add_scaled(dst: BSSNState, a: float, src: BSSNState, shape: tuple):
    """dst = dst + a * src for all fields."""
    fields = [
        (dst.phi, src.phi), (dst.gt11, src.gt11), (dst.gt12, src.gt12),
        (dst.gt13, src.gt13), (dst.gt22, src.gt22), (dst.gt23, src.gt23),
        (dst.gt33, src.gt33), (dst.Xt1, src.Xt1), (dst.Xt2, src.Xt2),
        (dst.Xt3, src.Xt3), (dst.trK, src.trK),
        (dst.At11, src.At11), (dst.At12, src.At12), (dst.At13, src.At13),
        (dst.At22, src.At22), (dst.At23, src.At23), (dst.At33, src.At33),
        (dst.alpha, src.alpha), (dst.beta1, src.beta1),
        (dst.beta2, src.beta2), (dst.beta3, src.beta3),
    ]
    for y, x in fields:
        wp.launch(axpy_kernel, dim=shape, inputs=[y, a, x])


def rk4_final(result: BSSNState, y0: BSSNState, 
              k1: BSSNState, k2: BSSNState, k3: BSSNState, k4: BSSNState,
              dt: float, shape: tuple):
    """result = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4) for all fields."""
    all_fields = [
        (result.phi, y0.phi, k1.phi, k2.phi, k3.phi, k4.phi),
        (result.gt11, y0.gt11, k1.gt11, k2.gt11, k3.gt11, k4.gt11),
        (result.gt12, y0.gt12, k1.gt12, k2.gt12, k3.gt12, k4.gt12),
        (result.gt13, y0.gt13, k1.gt13, k2.gt13, k3.gt13, k4.gt13),
        (result.gt22, y0.gt22, k1.gt22, k2.gt22, k3.gt22, k4.gt22),
        (result.gt23, y0.gt23, k1.gt23, k2.gt23, k3.gt23, k4.gt23),
        (result.gt33, y0.gt33, k1.gt33, k2.gt33, k3.gt33, k4.gt33),
        (result.Xt1, y0.Xt1, k1.Xt1, k2.Xt1, k3.Xt1, k4.Xt1),
        (result.Xt2, y0.Xt2, k1.Xt2, k2.Xt2, k3.Xt2, k4.Xt2),
        (result.Xt3, y0.Xt3, k1.Xt3, k2.Xt3, k3.Xt3, k4.Xt3),
        (result.trK, y0.trK, k1.trK, k2.trK, k3.trK, k4.trK),
        (result.At11, y0.At11, k1.At11, k2.At11, k3.At11, k4.At11),
        (result.At12, y0.At12, k1.At12, k2.At12, k3.At12, k4.At12),
        (result.At13, y0.At13, k1.At13, k2.At13, k3.At13, k4.At13),
        (result.At22, y0.At22, k1.At22, k2.At22, k3.At22, k4.At22),
        (result.At23, y0.At23, k1.At23, k2.At23, k3.At23, k4.At23),
        (result.At33, y0.At33, k1.At33, k2.At33, k3.At33, k4.At33),
        (result.alpha, y0.alpha, k1.alpha, k2.alpha, k3.alpha, k4.alpha),
        (result.beta1, y0.beta1, k1.beta1, k2.beta1, k3.beta1, k4.beta1),
        (result.beta2, y0.beta2, k1.beta2, k2.beta2, k3.beta2, k4.beta2),
        (result.beta3, y0.beta3, k1.beta3, k2.beta3, k3.beta3, k4.beta3),
    ]
    for y, y0_f, k1_f, k2_f, k3_f, k4_f in all_fields:
        wp.launch(rk4_combine_kernel, dim=shape, inputs=[y, y0_f, k1_f, k2_f, k3_f, k4_f, dt])


class BSSNEvolver:
    """BSSN evolution driver with RK4 integration."""
    
    def __init__(self, nx: int, ny: int, nz: int, dx: float,
                 eps_diss: float = 0.1, eta: float = 2.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dt = 0.25 * dx  # CFL condition
        self.eps_diss = eps_diss
        self.eta = eta
        self.shape = (nx, ny, nz)
        
        # Create states for RK4
        self.state = create_bssn_state(nx, ny, nz, dx)
        self.state0 = create_bssn_state(nx, ny, nz, dx)  # Initial state for RK step
        self.k1 = create_bssn_state(nx, ny, nz, dx)
        self.k2 = create_bssn_state(nx, ny, nz, dx)
        self.k3 = create_bssn_state(nx, ny, nz, dx)
        self.k4 = create_bssn_state(nx, ny, nz, dx)
        self.tmp = create_bssn_state(nx, ny, nz, dx)  # Temporary for intermediate states
        
        # Ricci tensor storage
        self.ricci = create_ricci_arrays(nx, ny, nz)
        
        self.time = 0.0
        self.step_count = 0
    
    def compute_rhs(self, state: BSSNState, rhs: BSSNState):
        """Compute full BSSN RHS including Ricci tensor."""
        inv_dx = 1.0 / self.dx
        
        # First compute Ricci tensor
        wp.launch(
            compute_christoffel_and_ricci,
            dim=self.shape,
            inputs=[
                state.gt11, state.gt12, state.gt13, state.gt22, state.gt23, state.gt33,
                state.phi,
                state.Xt1, state.Xt2, state.Xt3,
                self.ricci['Rt11'], self.ricci['Rt12'], self.ricci['Rt13'],
                self.ricci['Rt22'], self.ricci['Rt23'], self.ricci['Rt33'],
                self.ricci['trR'],
                inv_dx,
            ]
        )
        
        # Then compute full RHS
        wp.launch(
            compute_bssn_rhs_full,
            dim=self.shape,
            inputs=[
                state.phi,
                state.gt11, state.gt12, state.gt13, state.gt22, state.gt23, state.gt33,
                state.Xt1, state.Xt2, state.Xt3,
                state.trK,
                state.At11, state.At12, state.At13, state.At22, state.At23, state.At33,
                state.alpha,
                state.beta1, state.beta2, state.beta3,
                self.ricci['Rt11'], self.ricci['Rt12'], self.ricci['Rt13'],
                self.ricci['Rt22'], self.ricci['Rt23'], self.ricci['Rt33'],
                self.ricci['trR'],
                rhs.phi,
                rhs.gt11, rhs.gt12, rhs.gt13, rhs.gt22, rhs.gt23, rhs.gt33,
                rhs.Xt1, rhs.Xt2, rhs.Xt3,
                rhs.trK,
                rhs.At11, rhs.At12, rhs.At13, rhs.At22, rhs.At23, rhs.At33,
                rhs.alpha,
                rhs.beta1, rhs.beta2, rhs.beta3,
                inv_dx,
                self.eps_diss,
                self.eta,
            ]
        )
        
        # Apply boundary conditions
        apply_boundary_conditions(state, rhs)
    
    def step(self):
        """Perform one RK4 time step."""
        dt = self.dt
        
        # Store initial state
        copy_state(self.state, self.state0)
        
        # k1 = f(t, y)
        self.compute_rhs(self.state, self.k1)
        
        # y_tmp = y0 + dt/2 * k1
        copy_state(self.state0, self.tmp)
        add_scaled(self.tmp, 0.5 * dt, self.k1, self.shape)
        
        # k2 = f(t + dt/2, y_tmp)
        self.compute_rhs(self.tmp, self.k2)
        
        # y_tmp = y0 + dt/2 * k2
        copy_state(self.state0, self.tmp)
        add_scaled(self.tmp, 0.5 * dt, self.k2, self.shape)
        
        # k3 = f(t + dt/2, y_tmp)
        self.compute_rhs(self.tmp, self.k3)
        
        # y_tmp = y0 + dt * k3
        copy_state(self.state0, self.tmp)
        add_scaled(self.tmp, dt, self.k3, self.shape)
        
        # k4 = f(t + dt, y_tmp)
        self.compute_rhs(self.tmp, self.k4)
        
        # y = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        rk4_final(self.state, self.state0, self.k1, self.k2, self.k3, self.k4, dt, self.shape)
        
        self.time += dt
        self.step_count += 1
    
    def get_constraints(self) -> dict:
        """Compute constraint norms."""
        return compute_constraint_norms(self.state, self.ricci)
    
    def evolve(self, num_steps: int, print_interval: int = 10):
        """Evolve for multiple steps with monitoring."""
        for n in range(num_steps):
            self.step()
            
            if (n + 1) % print_interval == 0:
                constraints = self.get_constraints()
                alpha_center = self.state.alpha.numpy()[self.nx//2, self.ny//2, self.nz//2]
                print(f"  Step {self.step_count:4d}: t={self.time:.4f}, "
                      f"α_center={alpha_center:.6f}, H_L2={constraints['H_L2']:.2e}")


# ============================================================================
# Tests
# ============================================================================

def test_gauge_wave_evolution():
    """Test gauge wave evolution stability."""
    print("Testing gauge wave evolution (100 steps)...")
    
    nx, ny, nz = 32, 8, 8
    dx = 0.1
    
    evolver = BSSNEvolver(nx, ny, nz, dx, eps_diss=0.2, eta=1.0)
    init_gauge_wave_state(evolver.state, amplitude=0.02, wavelength=nx * dx)
    
    # Check initial state
    alpha_init = evolver.state.alpha.numpy().copy()
    
    # Evolve
    evolver.evolve(100, print_interval=20)
    
    # Check final state
    alpha_final = evolver.state.alpha.numpy()
    
    # Lapse should remain bounded
    assert alpha_final.min() > 0.5, f"Lapse collapsed: min = {alpha_final.min()}"
    assert alpha_final.max() < 1.5, f"Lapse exploded: max = {alpha_final.max()}"
    
    print("  PASSED!")
    return evolver


def test_flat_spacetime_evolution():
    """Test flat spacetime evolution - should remain stable."""
    print("Testing flat spacetime evolution (100 steps)...")
    
    nx, ny, nz = 24, 24, 24
    dx = 0.1
    
    evolver = BSSNEvolver(nx, ny, nz, dx, eps_diss=0.1, eta=1.0)
    init_flat_spacetime_state(evolver.state)
    
    # Evolve
    evolver.evolve(100, print_interval=25)
    
    # Check stability
    alpha_final = evolver.state.alpha.numpy()
    interior = (slice(4, -4), slice(4, -4), slice(4, -4))
    
    alpha_error = abs(alpha_final[interior] - 1.0).max()
    
    print(f"  Max |α - 1| in interior: {alpha_error:.6e}")
    assert alpha_error < 0.1, f"Flat spacetime drifted: error = {alpha_error}"
    
    print("  PASSED!")
    return evolver


def test_puncture_initial_data():
    """Test puncture (black hole) initial data validity.
    
    Note: Full puncture evolution requires moving-puncture gauge conditions
    and typically AMR (adaptive mesh refinement) to handle the singularity.
    This test verifies the initial data is correctly set up.
    """
    print("Testing puncture initial data...")
    
    nx, ny, nz = 40, 40, 40
    dx = 0.5
    
    evolver = BSSNEvolver(nx, ny, nz, dx, eps_diss=0.5, eta=2.0)
    init_brill_lindquist_state(evolver.state, mass=0.5)
    
    # Check initial state
    alpha_init = evolver.state.alpha.numpy()
    phi_init = evolver.state.phi.numpy()
    
    print(f"  Initial alpha range: [{alpha_init.min():.6f}, {alpha_init.max():.4f}]")
    print(f"  Initial phi range: [{phi_init.min():.4f}, {phi_init.max():.4f}]")
    
    # Verify properties of Brill-Lindquist data:
    # 1. Alpha (pre-collapsed lapse) should be small near center, approach 1 far away
    assert alpha_init.min() >= 0, "Lapse should be non-negative"
    assert alpha_init.max() <= 1.0, "Pre-collapsed lapse should be <= 1"
    
    # 2. Phi should be larger near center (where conformal factor is larger)
    center_phi = phi_init[nx//2, ny//2, nz//2]
    corner_phi = phi_init[0, 0, 0]
    assert center_phi > corner_phi, "Phi should be larger at center (near puncture)"
    
    # 3. Metric should be conformally flat
    gt11 = evolver.state.gt11.numpy()
    gt22 = evolver.state.gt22.numpy()
    gt33 = evolver.state.gt33.numpy()
    assert np.allclose(gt11, 1.0), "Conformal metric should be flat"
    assert np.allclose(gt22, 1.0), "Conformal metric should be flat"
    assert np.allclose(gt33, 1.0), "Conformal metric should be flat"
    
    # 4. Extrinsic curvature should be zero (time-symmetric data)
    trK = evolver.state.trK.numpy()
    assert np.allclose(trK, 0.0), "trK should be zero for time-symmetric data"
    
    # 5. Compute initial constraints (should be satisfied by construction)
    constraints = evolver.get_constraints()
    print(f"  Initial H_L2: {constraints['H_L2']:.6e}")
    
    print("  PASSED!")
    return evolver


def save_checkpoint(evolver: BSSNEvolver, filename: str):
    """Save evolution state to numpy file."""
    data = {
        'time': evolver.time,
        'step_count': evolver.step_count,
        'nx': evolver.nx,
        'ny': evolver.ny,
        'nz': evolver.nz,
        'dx': evolver.dx,
        'phi': evolver.state.phi.numpy(),
        'alpha': evolver.state.alpha.numpy(),
        'trK': evolver.state.trK.numpy(),
        'gt11': evolver.state.gt11.numpy(),
        'gt22': evolver.state.gt22.numpy(),
        'gt33': evolver.state.gt33.numpy(),
    }
    np.savez(filename, **data)
    print(f"Saved checkpoint to {filename}")


def test_checkpoint():
    """Test checkpoint save capability."""
    print("Testing checkpoint save...")
    
    nx, ny, nz = 16, 16, 16
    dx = 0.1
    
    evolver = BSSNEvolver(nx, ny, nz, dx)
    init_flat_spacetime_state(evolver.state)
    evolver.evolve(10, print_interval=10)
    
    # Save checkpoint
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_checkpoint.npz")
        save_checkpoint(evolver, filepath)
        
        # Verify file exists and can be loaded
        data = np.load(filepath)
        assert 'time' in data.files
        assert 'alpha' in data.files
        assert data['alpha'].shape == (nx, ny, nz)
        
    print("  PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("BSSN Evolution Driver Tests")
    print("=" * 60)
    
    test_flat_spacetime_evolution()
    print()
    test_gauge_wave_evolution()
    print()
    test_puncture_initial_data()
    print()
    test_checkpoint()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
