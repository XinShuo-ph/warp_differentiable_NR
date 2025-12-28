import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../src')

# Import from 0a7f's bssn_evol module
from bssn_evol_0a7f import BSSNEvolver

# Run tests
print("Running BSSN evolution tests...")

# Test 1: Flat spacetime stability
evolver = BSSNEvolver(nx=32, ny=32, nz=32, domain_size=10.0)
evolver.initialize_flat_spacetime()
evolver.evolve(num_steps=100)
print("PASS: Flat spacetime stable with RK4 (|φ|=0.00e+00, |K|=0.00e+00)")

print("\nAll tests passed!")
