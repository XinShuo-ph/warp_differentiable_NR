"""Test constraint preservation during BSSN evolution."""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import warp as wp

wp.init()

from src.bssn_evol import BSSNEvolver


def test_flat_spacetime_constraints():
    """Test that constraints remain zero on flat spacetime."""
    nx = 16
    dx = 1.0 / nx
    dt = 0.25 * dx
    n_steps = 50
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.0)
    
    # Initial constraints
    H_init, M_init = evolver.compute_constraints()
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    # Final constraints
    H_final, M_final = evolver.compute_constraints()
    
    assert H_init < 1e-10, f"Initial H too large: {H_init}"
    assert M_init < 1e-10, f"Initial M too large: {M_init}"
    assert H_final < 1e-10, f"Final H too large: {H_final}"
    assert M_final < 1e-10, f"Final M too large: {M_final}"
    
    print(f"PASS: Flat spacetime constraints preserved (H={H_final:.2e}, M={M_final:.2e})")


def test_gauge_wave_constraints():
    """Test constraint preservation with gauge wave."""
    nx = 16
    dx = 1.0 / nx
    dt = 0.25 * dx
    n_steps = 50
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.1)
    evolver.init_gauge_wave(amplitude=0.01, wavelength=1.0)
    
    # Initial constraints
    H_init, M_init = evolver.compute_constraints()
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    # Final constraints
    H_final, M_final = evolver.compute_constraints()
    
    # Gauge wave may have small constraint violations but should be bounded
    assert np.isfinite(H_final), "H is not finite"
    assert np.isfinite(M_final), "M is not finite"
    assert H_final < 1.0, f"H too large: {H_final}"
    assert M_final < 1.0, f"M too large: {M_final}"
    
    print(f"PASS: Gauge wave constraints bounded (H={H_final:.2e}, M={M_final:.2e})")


def test_bh_constraints_bounded():
    """Test that BH constraints remain bounded during evolution."""
    nx = 16
    L = 10.0
    dx = L / nx
    dt = 0.1 * dx
    n_steps = 20
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.2, use_sommerfeld=True)
    evolver.init_brill_lindquist(mass=1.0)
    
    initial_H, _ = evolver.compute_constraints()
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    final_H, final_M = evolver.compute_constraints()
    
    # For BH, constraints may grow but should not blow up
    assert np.isfinite(final_H), "H is not finite for BH"
    assert np.isfinite(final_M), "M is not finite for BH"
    
    print(f"PASS: BH constraints bounded (H={final_H:.2e}, M={final_M:.2e})")


if __name__ == "__main__":
    print("Running constraint tests...\n")
    
    test_flat_spacetime_constraints()
    test_gauge_wave_constraints()
    test_bh_constraints_bounded()
    
    print("\nAll constraint tests passed!")
