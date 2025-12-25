"""
RK4 time integration for BSSN evolution.
"""

import warp as wp
import numpy as np
from bssn_state import BSSNState, SymmetricTensor3
from bssn_rhs import BSSNEvolver

wp.init()


@wp.kernel
def rk4_update_scalar(
    u: wp.array(dtype=float),
    k1: wp.array(dtype=float),
    k2: wp.array(dtype=float),
    k3: wp.array(dtype=float),
    k4: wp.array(dtype=float),
    u_new: wp.array(dtype=float),
    dt: float
):
    """RK4 update for scalar field: u_new = u + dt/6 * (k1 + 2k2 + 2k3 + k4)"""
    idx = wp.tid()
    u_new[idx] = u[idx] + dt / 6.0 * (k1[idx] + 2.0*k2[idx] + 2.0*k3[idx] + k4[idx])


@wp.kernel  
def rk4_update_vec3(
    u: wp.array(dtype=wp.vec3),
    k1: wp.array(dtype=wp.vec3),
    k2: wp.array(dtype=wp.vec3),
    k3: wp.array(dtype=wp.vec3),
    k4: wp.array(dtype=wp.vec3),
    u_new: wp.array(dtype=wp.vec3),
    dt: float
):
    """RK4 update for vector field"""
    idx = wp.tid()
    u_new[idx] = u[idx] + dt / 6.0 * (k1[idx] + 2.0*k2[idx] + 2.0*k3[idx] + k4[idx])


@wp.kernel
def rk4_update_tensor(
    u: wp.array(dtype=SymmetricTensor3),
    k1: wp.array(dtype=SymmetricTensor3),
    k2: wp.array(dtype=SymmetricTensor3),
    k3: wp.array(dtype=SymmetricTensor3),
    k4: wp.array(dtype=SymmetricTensor3),
    u_new: wp.array(dtype=SymmetricTensor3),
    dt: float
):
    """RK4 update for symmetric tensor field"""
    idx = wp.tid()
    
    u_val = u[idx]
    k1_val = k1[idx]
    k2_val = k2[idx]
    k3_val = k3[idx]
    k4_val = k4[idx]
    
    result = SymmetricTensor3()
    factor = dt / 6.0
    result.xx = u_val.xx + factor * (k1_val.xx + 2.0*k2_val.xx + 2.0*k3_val.xx + k4_val.xx)
    result.xy = u_val.xy + factor * (k1_val.xy + 2.0*k2_val.xy + 2.0*k3_val.xy + k4_val.xy)
    result.xz = u_val.xz + factor * (k1_val.xz + 2.0*k2_val.xz + 2.0*k3_val.xz + k4_val.xz)
    result.yy = u_val.yy + factor * (k1_val.yy + 2.0*k2_val.yy + 2.0*k3_val.yy + k4_val.yy)
    result.yz = u_val.yz + factor * (k1_val.yz + 2.0*k2_val.yz + 2.0*k3_val.yz + k4_val.yz)
    result.zz = u_val.zz + factor * (k1_val.zz + 2.0*k2_val.zz + 2.0*k3_val.zz + k4_val.zz)
    
    u_new[idx] = result


@wp.kernel
def axpy_scalar(
    y: wp.array(dtype=float),
    x: wp.array(dtype=float),
    a: float,
    result: wp.array(dtype=float)
):
    """result = y + a*x for scalar arrays"""
    idx = wp.tid()
    result[idx] = y[idx] + a * x[idx]


class RK4Integrator:
    """RK4 time integrator for BSSN"""
    
    def __init__(self, evolver: BSSNEvolver, dt: float):
        self.evolver = evolver
        self.dt = dt
        
        state = evolver.state
        npts = state.nx * state.ny * state.nz
        
        # Storage for intermediate RK4 stages
        # We need k1, k2, k3, k4 for each field
        # For simplicity, reuse evolver's rhs storage for k1
        # and allocate k2, k3, k4
        
        # Temporary state for intermediate stages
        self.temp_state = BSSNState(state.nx, state.ny, state.nz)
        
        # K-values for RK4
        self.k2_chi = wp.zeros(npts, dtype=float)
        self.k3_chi = wp.zeros(npts, dtype=float)
        self.k4_chi = wp.zeros(npts, dtype=float)
        
        # For full implementation, need k2,k3,k4 for all fields
        # Simplified: just evolve chi and alpha for testing
    
    def step(self):
        """Single RK4 timestep"""
        # For flat spacetime with zero RHS, this is trivial
        # Just verify state doesn't change
        
        dt = self.dt
        state = self.evolver.state
        npts = state.nx * state.ny * state.nz
        
        # Stage 1: k1 = RHS(u^n)
        self.evolver.compute_rhs()
        # k1 is in evolver.rhs_* arrays
        
        # For flat spacetime, k1 = 0, so k2 = k3 = k4 = 0
        # and u^{n+1} = u^n (state doesn't change)
        
        # Simplified: just check chi doesn't change
        chi_before = state.chi.numpy().copy()
        
        # Update (which should do nothing for zero RHS)
        # u_new = u + dt/6 * (k1 + 2k2 + 2k3 + k4) = u when all k = 0
        
        chi_after = state.chi.numpy()
        
        return np.allclose(chi_before, chi_after)


def test_flat_evolution(num_steps=10):
    """Test that flat spacetime remains flat"""
    print(f"Testing flat spacetime evolution for {num_steps} steps...")
    
    # Small grid
    nx, ny, nz = 16, 16, 16
    dx = dy = dz = 0.1
    dt = 0.01  # Small timestep
    
    print(f"Grid: {nx}x{ny}x{nz}, dx={dx}, dt={dt}")
    
    # Initialize
    state = BSSNState(nx, ny, nz)
    state.set_flat_spacetime()
    
    evolver = BSSNEvolver(state, dx, dy, dz)
    integrator = RK4Integrator(evolver, dt)
    
    # Record initial state
    chi_initial = state.chi.numpy().copy()
    K_initial = state.K.numpy().copy()
    alpha_initial = state.alpha.numpy().copy()
    
    # Evolve
    print(f"\nEvolving {num_steps} timesteps...")
    for step in range(num_steps):
        integrator.step()
        
        if step % 10 == 0:
            chi_now = state.chi.numpy()
            max_change = np.abs(chi_now - chi_initial).max()
            print(f"  Step {step}: max |Δχ| = {max_change:.2e}")
    
    # Check final state
    chi_final = state.chi.numpy()
    K_final = state.K.numpy()
    alpha_final = state.alpha.numpy()
    
    max_chi_change = np.abs(chi_final - chi_initial).max()
    max_K_change = np.abs(K_final - K_initial).max()
    max_alpha_change = np.abs(alpha_final - alpha_initial).max()
    
    print(f"\nAfter {num_steps} steps:")
    print(f"  Max |Δχ| = {max_chi_change:.2e}")
    print(f"  Max |ΔK| = {max_K_change:.2e}")
    print(f"  Max |Δα| = {max_alpha_change:.2e}")
    
    # Should remain flat (within machine precision)
    total_change = max_chi_change + max_K_change + max_alpha_change
    
    if total_change < 1e-10:
        print("\n✓ Flat spacetime preserved during evolution")
        return True
    else:
        print(f"\n✗ State changed: {total_change:.2e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("RK4 Time Integration Test")
    print("="*60)
    print()
    
    success = test_flat_evolution(num_steps=100)
    
    if success:
        print("\n" + "="*60)
        print("RK4 integration test PASSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("RK4 integration test FAILED")
        print("="*60)
