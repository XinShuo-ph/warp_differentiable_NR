"""
Test BSSN evolution on flat spacetime.

Flat spacetime should remain stable under evolution.
"""

import sys
sys.path.insert(0, '/workspace/NR/src')

import warp as wp
import numpy as np
from bssn_evolver import BSSNEvolver

wp.init()


def test_flat_spacetime_evolution():
    """Test that flat spacetime remains flat under evolution"""
    
    print("=" * 70)
    print("BSSN Flat Spacetime Evolution Test")
    print("=" * 70)
    
    # Grid parameters
    nx, ny, nz = 32, 32, 32
    xmin, xmax = -5.0, 5.0
    dx = (xmax - xmin) / (nx - 1)
    dy = dx
    dz = dx
    
    # Time stepping
    cfl = 0.25
    dt = cfl * dx
    num_steps = 100
    
    print(f"\nGrid: {nx} x {ny} x {nz}")
    print(f"Domain: [{xmin}, {xmax}]³")
    print(f"dx = {dx:.4f}")
    print(f"dt = {dt:.4f} (CFL = {cfl})")
    print(f"Steps: {num_steps}")
    print(f"Final time: {num_steps * dt:.4f}")
    
    # Create evolver
    evolver = BSSNEvolver(nx, ny, nz, dx, dy, dz, dt, eps_diss=0.1)
    
    # Initialize to flat spacetime
    print("\nInitializing flat spacetime...")
    evolver.vars.set_flat_spacetime()
    
    # Check initial values
    phi_init = evolver.vars.phi.numpy()
    gt_xx_init = evolver.vars.gt_xx.numpy()
    alpha_init = evolver.vars.alpha.numpy()
    
    print(f"  phi: min={np.min(phi_init):.6f}, max={np.max(phi_init):.6f}")
    print(f"  gt_xx: min={np.min(gt_xx_init):.6f}, max={np.max(gt_xx_init):.6f}")
    print(f"  alpha: min={np.min(alpha_init):.6f}, max={np.max(alpha_init):.6f}")
    
    # Evolve
    print(f"\nEvolving for {num_steps} steps...")
    evolver.evolve(num_steps)
    
    # Check final values
    phi_final = evolver.vars.phi.numpy()
    gt_xx_final = evolver.vars.gt_xx.numpy()
    alpha_final = evolver.vars.alpha.numpy()
    K_final = evolver.vars.K.numpy()
    
    print(f"\nFinal values:")
    print(f"  phi: min={np.min(phi_final):.6f}, max={np.max(phi_final):.6f}")
    print(f"  gt_xx: min={np.min(gt_xx_final):.6f}, max={np.max(gt_xx_final):.6f}")
    print(f"  alpha: min={np.min(alpha_final):.6f}, max={np.max(alpha_final):.6f}")
    print(f"  K: min={np.min(K_final):.6f}, max={np.max(K_final):.6f}")
    
    # Compute deviations from flat spacetime
    interior = np.s_[5:-5, 5:-5, 5:-5]  # Skip boundaries
    
    phi_dev = np.abs(phi_final[interior] - 0.0)
    gt_xx_dev = np.abs(gt_xx_final[interior] - 1.0)
    alpha_dev = np.abs(alpha_final[interior] - 1.0)
    K_dev = np.abs(K_final[interior])
    
    print(f"\nDeviations from flat spacetime (interior):")
    print(f"  phi: max={np.max(phi_dev):.6e}, rms={np.sqrt(np.mean(phi_dev**2)):.6e}")
    print(f"  gt_xx: max={np.max(gt_xx_dev):.6e}, rms={np.sqrt(np.mean(gt_xx_dev**2)):.6e}")
    print(f"  alpha: max={np.max(alpha_dev):.6e}, rms={np.sqrt(np.mean(alpha_dev**2)):.6e}")
    print(f"  K: max={np.max(K_dev):.6e}, rms={np.sqrt(np.mean(K_dev**2)):.6e}")
    
    # Check stability
    print(f"\n" + "=" * 70)
    if np.max(phi_dev) < 0.01 and np.max(gt_xx_dev) < 0.01:
        print("✓ Evolution stable - flat spacetime preserved")
        success = True
    else:
        print("✗ Evolution unstable - deviations too large")
        success = False
    
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    test_flat_spacetime_evolution()
