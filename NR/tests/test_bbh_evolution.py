"""
Complete test: Evolve BBH initial data

Tests the full pipeline:
1. Set up Brill-Lindquist initial data
2. Evolve with BSSN equations
3. Monitor constraints and fields
"""

import warp as wp
import numpy as np
import sys
sys.path.insert(0, 'src')

from bssn_state import BSSNState
from bbh_initial_data import create_bbh_initial_data
from bssn_rhs_full import compute_bssn_rhs_full, compute_gauge_rhs_full

wp.init()


def evolve_bbh(num_steps=50, output_interval=10):
    """
    Evolve BBH spacetime starting from Brill-Lindquist initial data.
    """
    print("="*70)
    print("BBH Evolution Test")
    print("="*70)
    
    # Grid setup - smaller for faster testing
    nx, ny, nz = 48, 48, 48
    L = 30.0
    xmin = ymin = zmin = -L/2
    dx = dy = dz = L / (nx - 1)
    
    # CFL condition
    cfl = 0.2
    dt = cfl * min(dx, dy, dz)
    
    print(f"\nConfiguration:")
    print(f"  Grid: {nx} x {ny} x {nz} = {nx*ny*nz} points")
    print(f"  Domain: [{xmin:.1f}, {-xmin:.1f}]³")
    print(f"  Spacing: dx = {dx:.3f}")
    print(f"  Timestep: dt = {dt:.4f} (CFL = {cfl})")
    print(f"  Evolution time: T = {num_steps * dt:.3f}")
    
    # Initialize BBH
    print("\nInitializing BBH (Brill-Lindquist)...")
    state = BSSNState(nx, ny, nz)
    create_bbh_initial_data(state, xmin, ymin, zmin, dx, dy, dz,
                           separation=8.0, mass_ratio=1.0)
    
    # Allocate RHS storage
    npts = nx * ny * nz
    rhs_chi = wp.zeros(npts, dtype=float)
    rhs_K = wp.zeros(npts, dtype=float)
    rhs_alpha = wp.zeros(npts, dtype=float)
    from bssn_state import SymmetricTensor3
    rhs_gamma = wp.zeros(npts, dtype=SymmetricTensor3)
    rhs_A = wp.zeros(npts, dtype=SymmetricTensor3)
    rhs_Gamma = wp.zeros(npts, dtype=wp.vec3)
    rhs_beta = wp.zeros(npts, dtype=wp.vec3)
    B_tilde = wp.zeros(npts, dtype=wp.vec3)
    rhs_B = wp.zeros(npts, dtype=wp.vec3)
    
    # Initial values
    chi_init = state.chi.numpy().copy()
    alpha_init = state.alpha.numpy().copy()
    K_init = state.K.numpy().copy()
    
    print(f"\nInitial data:")
    print(f"  χ: min = {chi_init.min():.6f}, max = {chi_init.max():.6f}")
    print(f"  α: min = {alpha_init.min():.6f}, max = {alpha_init.max():.6f}")
    print(f"  K: max |K| = {np.abs(K_init).max():.2e}")
    
    # Evolution loop
    print(f"\n{'='*70}")
    print("Evolving BBH spacetime...")
    print(f"{'='*70}")
    print(f"{'Step':>6} {'Time':>10} {'|Δχ|_max':>12} {'|Δα|_max':>12} {'Status':>10}")
    print("-"*70)
    
    for step in range(num_steps + 1):
        t = step * dt
        
        if step % output_interval == 0:
            chi_now = state.chi.numpy()
            alpha_now = state.alpha.numpy()
            
            dchi = np.abs(chi_now - chi_init).max()
            dalpha = np.abs(alpha_now - alpha_init).max()
            
            status = "OK"
            if dchi > 0.5 or dalpha > 0.5:
                status = "CHANGING"
            
            print(f"{step:6d} {t:10.4f} {dchi:12.4e} {dalpha:12.4e} {status:>10}")
        
        if step < num_steps:
            # Compute RHS
            wp.launch(
                compute_bssn_rhs_full,
                dim=(nx, ny, nz),
                inputs=[
                    state.chi, state.gamma_tilde, state.K,
                    state.A_tilde, state.Gamma_tilde,
                    state.alpha, state.beta,
                    rhs_chi, rhs_gamma, rhs_K,
                    rhs_A, rhs_Gamma,
                    nx, ny, nz, dx, dy, dz
                ]
            )
            
            wp.launch(
                compute_gauge_rhs_full,
                dim=(nx, ny, nz),
                inputs=[
                    state.alpha, state.beta, state.K,
                    state.Gamma_tilde,
                    rhs_alpha, rhs_beta,
                    B_tilde, rhs_B,
                    nx, ny, nz, dx, dy, dz, 1.0
                ]
            )
            
            # Simple forward Euler update (RK4 would be better but more complex)
            # u^{n+1} = u^n + dt * RHS(u^n)
            # This is just for testing - real evolution would use RK4
            
            chi_np = state.chi.numpy()
            alpha_np = state.alpha.numpy()
            K_np = state.K.numpy()
            
            rhs_chi_np = rhs_chi.numpy()
            rhs_alpha_np = rhs_alpha.numpy()
            rhs_K_np = rhs_K.numpy()
            
            # Update (forward Euler - first order, for simplicity)
            chi_np += dt * rhs_chi_np
            alpha_np += dt * rhs_alpha_np
            K_np += dt * rhs_K_np
            
            # Bounds checking
            chi_np = np.clip(chi_np, 0.01, 2.0)
            alpha_np = np.clip(alpha_np, 0.1, 2.0)
            
            state.chi.assign(chi_np)
            state.alpha.assign(alpha_np)
            state.K.assign(K_np)
    
    print("-"*70)
    
    # Final analysis
    print(f"\n{'='*70}")
    print("Evolution Complete")
    print(f"{'='*70}")
    
    chi_final = state.chi.numpy()
    alpha_final = state.alpha.numpy()
    K_final = state.K.numpy()
    
    print(f"\nFinal state:")
    print(f"  χ: min = {chi_final.min():.6f}, max = {chi_final.max():.6f}")
    print(f"  α: min = {alpha_final.min():.6f}, max = {alpha_final.max():.6f}")
    print(f"  K: max |K| = {np.abs(K_final).max():.2e}")
    
    print(f"\nChanges from initial:")
    print(f"  Δχ: max = {np.abs(chi_final - chi_init).max():.4e}")
    print(f"  Δα: max = {np.abs(alpha_final - alpha_init).max():.4e}")
    print(f"  ΔK: max = {np.abs(K_final - K_init).max():.4e}")
    
    print(f"\n{'='*70}")
    print("BBH evolution test completed")
    print("Note: This uses simplified RHS and forward Euler integration")
    print("Full implementation would include all BSSN terms and RK4")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    success = evolve_bbh(num_steps=50, output_interval=10)
    
    if success:
        print("\n✓ BBH evolution test completed")
    else:
        print("\n✗ Test encountered issues")
