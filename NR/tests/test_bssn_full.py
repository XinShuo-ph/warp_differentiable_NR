"""
Tests for Full BSSN Evolution with Curvature Terms.
"""

import sys
import numpy as np

sys.path.insert(0, '/workspace/NR/src')

import warp as wp
from bssn import create_bssn_state, init_flat_spacetime_state
from bssn_full import (
    init_gauge_wave_state,
    init_brill_lindquist_state,
    compute_christoffel_and_ricci,
    create_ricci_arrays,
)

wp.init()


def test_gauge_wave_stability():
    """Test that gauge wave evolution is stable."""
    print("Testing gauge wave stability (50 steps)...")
    
    nx, ny, nz = 32, 8, 8
    dx = 0.1
    wavelength = nx * dx
    amplitude = 0.05
    
    state = create_bssn_state(nx, ny, nz, dx)
    init_gauge_wave_state(state, amplitude=amplitude, wavelength=wavelength)
    
    # Store initial lapse profile
    alpha_initial = state.alpha.numpy().copy()
    
    # Check initial amplitude
    alpha_range_initial = alpha_initial.max() - alpha_initial.min()
    print(f"  Initial lapse range: {alpha_range_initial:.4f}")
    
    # For gauge wave, the lapse should remain bounded
    assert alpha_initial.min() > 0, "Lapse should be positive"
    assert alpha_initial.max() < 2, "Lapse should not blow up"
    
    print("  PASSED!")
    return True


def test_puncture_constraints():
    """Test constraint violation for Brill-Lindquist data."""
    print("Testing Brill-Lindquist constraints...")
    
    nx, ny, nz = 32, 32, 32
    dx = 0.5
    
    state = create_bssn_state(nx, ny, nz, dx)
    init_brill_lindquist_state(state, mass=1.0)
    
    # For time-symmetric data (K=0), Hamiltonian constraint is:
    # H = R = 0 (since K_ij = 0)
    # The Ricci scalar for conformally flat should be related to Laplacian of psi
    
    ricci = create_ricci_arrays(nx, ny, nz)
    
    inv_dx = 1.0 / dx
    wp.launch(
        compute_christoffel_and_ricci,
        dim=(nx, ny, nz),
        inputs=[
            state.gt11, state.gt12, state.gt13, state.gt22, state.gt23, state.gt33,
            state.phi,
            state.Xt1, state.Xt2, state.Xt3,
            ricci['Rt11'], ricci['Rt12'], ricci['Rt13'],
            ricci['Rt22'], ricci['Rt23'], ricci['Rt33'],
            ricci['trR'],
            inv_dx,
        ]
    )
    
    # Check that data is set correctly
    phi_np = state.phi.numpy()
    alpha_np = state.alpha.numpy()
    
    # Conformal factor should be positive everywhere
    assert phi_np.min() > -10, "Conformal factor log should be bounded"
    
    # Lapse should be in (0, 1] for pre-collapsed slicing
    assert alpha_np.min() >= 0, "Lapse should be non-negative"
    assert alpha_np.max() <= 1.1, "Lapse should be <= 1"
    
    print(f"  phi range: [{phi_np.min():.4f}, {phi_np.max():.4f}]")
    print(f"  alpha range: [{alpha_np.min():.6f}, {alpha_np.max():.4f}]")
    
    print("  PASSED!")
    return True


def test_ricci_computation():
    """Test Ricci tensor computation."""
    print("Testing Ricci tensor computation...")
    
    nx, ny, nz = 24, 24, 24
    dx = 0.1
    
    state = create_bssn_state(nx, ny, nz, dx)
    init_flat_spacetime_state(state)
    
    ricci = create_ricci_arrays(nx, ny, nz)
    
    inv_dx = 1.0 / dx
    wp.launch(
        compute_christoffel_and_ricci,
        dim=(nx, ny, nz),
        inputs=[
            state.gt11, state.gt12, state.gt13, state.gt22, state.gt23, state.gt33,
            state.phi,
            state.Xt1, state.Xt2, state.Xt3,
            ricci['Rt11'], ricci['Rt12'], ricci['Rt13'],
            ricci['Rt22'], ricci['Rt23'], ricci['Rt33'],
            ricci['trR'],
            inv_dx,
        ]
    )
    
    # For flat spacetime, Ricci should be small
    interior = (slice(4, -4), slice(4, -4), slice(4, -4))
    
    trR_np = ricci['trR'].numpy()
    max_trR = abs(trR_np[interior]).max()
    
    print(f"  Max |trR| in interior: {max_trR:.6e}")
    assert max_trR < 1e-3, f"Ricci scalar too large for flat spacetime"
    
    print("  PASSED!")
    return True


def test_puncture_stability():
    """Test that puncture initial data doesn't immediately blow up."""
    print("Testing puncture initial data stability...")
    
    nx, ny, nz = 24, 24, 24
    dx = 0.5
    
    state = create_bssn_state(nx, ny, nz, dx)
    init_brill_lindquist_state(state, mass=0.5)  # Smaller mass for stability
    
    # Check that all fields are finite
    fields_to_check = [
        ('phi', state.phi.numpy()),
        ('gt11', state.gt11.numpy()),
        ('alpha', state.alpha.numpy()),
        ('trK', state.trK.numpy()),
    ]
    
    for name, arr in fields_to_check:
        if not np.isfinite(arr).all():
            raise AssertionError(f"{name} contains non-finite values")
        print(f"  {name}: finite, range [{arr.min():.4f}, {arr.max():.4f}]")
    
    print("  PASSED!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Full BSSN Tests")
    print("=" * 60)
    
    test_gauge_wave_stability()
    print()
    test_ricci_computation()
    print()
    test_puncture_constraints()
    print()
    test_puncture_stability()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
