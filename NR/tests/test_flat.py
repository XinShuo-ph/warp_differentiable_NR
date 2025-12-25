import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import warp as wp
import numpy as np
from bssn import BSSNState
from integrator import RK4Integrator

def test_flat_spacetime_evolution():
    wp.init()
    
    # 1. Setup flat spacetime
    res = (32, 32, 32)
    bounds_lo = (0.0, 0.0, 0.0)
    bounds_hi = (1.0, 1.0, 1.0)
    
    state = BSSNState(res, bounds_lo, bounds_hi)
    state.init_flat_spacetime()
    
    integrator = RK4Integrator(state)
    
    # 2. Evolve for 100 steps
    dt = 0.25 * state.dx # CFL = 0.25
    steps = 100
    
    print(f"Evolving flat spacetime for {steps} steps with dt={dt}")
    
    for i in range(steps):
        integrator.step(dt)
        if (i+1) % 10 == 0:
            # Check constraints (flat space should stay flat)
            # K should be 0, phi should be 0, etc.
            max_K = np.max(np.abs(state.K.numpy()))
            max_phi = np.max(np.abs(state.phi.numpy()))
            print(f"Step {i+1}: Max |K| = {max_K}, Max |phi| = {max_phi}")
            
            if max_K > 1e-10 or max_phi > 1e-10:
                print("Stability check failed!")
                exit(1)
                
    print("Flat spacetime evolution stable.")

if __name__ == "__main__":
    test_flat_spacetime_evolution()
