"""Tests for complete BSSN evolution."""

import numpy as np
import warp as wp

wp.init()

import sys
sys.path.insert(0, '/workspace/NR')

from src.bssn_evol import BSSNEvolver


def test_flat_spacetime_rk4():
    """Test that flat spacetime remains stable with RK4."""
    nx = 16
    dx = 1.0 / nx
    dt = 0.25 * dx
    n_steps = 50
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.0)
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    # Check flat spacetime preservation
    phi_max = np.max(np.abs(evolver.phi.numpy()))
    K_max = np.max(np.abs(evolver.K.numpy()))
    alpha_err = np.max(np.abs(evolver.alpha.numpy() - 1.0))
    
    assert phi_max < 1e-10, f"φ deviated: {phi_max}"
    assert K_max < 1e-10, f"K deviated: {K_max}"
    assert alpha_err < 1e-10, f"α deviated: {alpha_err}"
    
    print(f"PASS: Flat spacetime stable with RK4 (|φ|={phi_max:.2e}, |K|={K_max:.2e})")


def test_gauge_wave_stability():
    """Test that gauge wave evolution is stable."""
    nx = 16
    dx = 1.0 / nx
    dt = 0.25 * dx
    n_steps = 100
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.1)
    evolver.init_gauge_wave(amplitude=0.01, wavelength=1.0)
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    alpha_np = evolver.alpha.numpy()
    
    # Check stability (lapse should remain bounded)
    assert np.all(np.isfinite(alpha_np)), "NaN/Inf in solution"
    assert alpha_np.min() > 0.5, f"Lapse collapsed: min={alpha_np.min()}"
    assert alpha_np.max() < 2.0, f"Lapse blew up: max={alpha_np.max()}"
    
    print(f"PASS: Gauge wave stable (α∈[{alpha_np.min():.4f},{alpha_np.max():.4f}])")


def test_constraint_monitoring():
    """Test that constraint monitoring works."""
    nx = 8
    dx = 1.0 / nx
    
    evolver = BSSNEvolver(nx, nx, nx, dx)
    
    # For flat spacetime, constraints should be zero
    H_max, M_max = evolver.compute_constraints()
    
    assert np.isfinite(H_max), "Hamiltonian constraint returned NaN"
    assert np.isfinite(M_max), "Momentum constraint returned NaN"
    # For flat spacetime with our setup, constraints should be small
    print(f"PASS: Constraint monitoring works (H={H_max:.2e}, M={M_max:.2e})")


def test_rk4_consistency():
    """Test that two RK4 runs give same result."""
    nx = 8
    dx = 1.0 / nx
    dt = 0.25 * dx
    n_steps = 20
    
    results = []
    for run in range(2):
        evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.1)
        evolver.init_gauge_wave(amplitude=0.01, wavelength=1.0)
        
        for step in range(n_steps):
            evolver.step_rk4(dt)
        
        results.append(evolver.alpha.numpy().copy())
    
    diff = np.max(np.abs(results[0] - results[1]))
    assert diff < 1e-12, f"Results not consistent: diff = {diff}"
    print(f"PASS: RK4 consistent (diff = {diff:.2e})")


def test_sommerfeld_boundary():
    """Test Sommerfeld boundary conditions."""
    nx = 16
    dx = 1.0 / nx
    dt = 0.25 * dx
    n_steps = 50
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.1, use_sommerfeld=True)
    evolver.init_gauge_wave(amplitude=0.01, wavelength=1.0)
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    alpha_np = evolver.alpha.numpy()
    
    # Check stability with Sommerfeld BCs
    assert np.all(np.isfinite(alpha_np)), "NaN/Inf in solution with Sommerfeld BCs"
    assert alpha_np.min() > 0.5, f"Lapse collapsed with Sommerfeld: min={alpha_np.min()}"
    
    print(f"PASS: Sommerfeld BCs stable (α∈[{alpha_np.min():.4f},{alpha_np.max():.4f}])")


def test_brill_lindquist():
    """Test Brill-Lindquist (puncture) black hole initial data."""
    nx = 16
    L = 10.0
    dx = L / nx
    dt = 0.1 * dx
    n_steps = 20
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.2, use_sommerfeld=True)
    evolver.init_brill_lindquist(mass=1.0)
    
    # Check initial data makes sense
    phi_np = evolver.phi.numpy()
    alpha_np = evolver.alpha.numpy()
    
    assert phi_np.max() > 0.5, "φ should be positive near puncture"
    assert alpha_np.min() < 0.5, "α should be small near puncture (pre-collapsed)"
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    # Check stability
    alpha_np = evolver.alpha.numpy()
    assert np.all(np.isfinite(alpha_np)), "NaN in Brill-Lindquist evolution"
    assert alpha_np.min() > 0.0, f"Lapse went negative: {alpha_np.min()}"
    
    print(f"PASS: Brill-Lindquist stable (α_min={alpha_np.min():.4f})")


def test_binary_black_hole():
    """Test binary black hole (two punctures) initial data."""
    nx = 16
    L = 16.0
    dx = L / nx
    dt = 0.1 * dx
    n_steps = 10
    
    evolver = BSSNEvolver(nx, nx, nx, dx, sigma=0.2, use_sommerfeld=True)
    evolver.init_binary_bh(mass1=0.5, mass2=0.5, separation=4.0)
    
    # Check initial data
    phi_np = evolver.phi.numpy()
    assert phi_np.max() > 0.3, "φ should show two punctures"
    
    for step in range(n_steps):
        evolver.step_rk4(dt)
    
    alpha_np = evolver.alpha.numpy()
    assert np.all(np.isfinite(alpha_np)), "NaN in binary BH evolution"
    assert alpha_np.min() > 0.0, f"Lapse went negative: {alpha_np.min()}"
    
    print(f"PASS: Binary BH stable (α_min={alpha_np.min():.4f})")


if __name__ == "__main__":
    print("Running BSSN evolution tests...\n")
    
    test_flat_spacetime_rk4()
    test_gauge_wave_stability()
    test_constraint_monitoring()
    test_rk4_consistency()
    test_sommerfeld_boundary()
    test_brill_lindquist()
    test_binary_black_hole()
    
    print("\nAll tests passed!")
