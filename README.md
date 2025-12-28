# Differentiable BSSN Numerical Relativity

A production-ready differentiable numerical relativity codebase implementing the BSSN formulation using NVIDIA Warp.

## Features

### Core BSSN Evolution
- **Complete BSSN Variables**: φ, χ, γ̃ᵢⱼ, Āᵢⱼ, K, Γ̄ⁱ, α, βⁱ
- **4th Order Finite Differences**: High-accuracy spatial derivatives
- **RK4 Time Integration**: Stable time stepping
- **Kreiss-Oliger Dissipation**: Numerical stability enhancement
- **Constraint Monitoring**: Hamiltonian and momentum constraints

### Initial Data
- Flat spacetime
- Gauge wave perturbations
- Brill-Lindquist (single black hole)
- Binary black hole (two punctures)
- Schwarzschild puncture

### Gauge Conditions
- 1+log slicing for lapse
- Gamma-driver for shift
- Sommerfeld radiative boundary conditions

### ML Integration (from branch 0d97)
- **bssn_ml_pipeline.py**: End-to-end differentiable pipeline
- **bssn_losses.py**: Differentiable loss functions (constraint, stability, asymptotic)
- **bssn_waveform.py**: Gravitational waveform extraction
- **bssn_optimization.py**: Gradient-based parameter optimization

## Quick Start

```python
import warp as wp
wp.init()

from src.bssn_evol import BSSNEvolver

# Create evolver with 32³ grid
evolver = BSSNEvolver(32, 32, 32, dx=0.25, sigma=0.1)

# Initialize with gauge wave
evolver.init_gauge_wave(amplitude=0.01, wavelength=1.0)

# Evolve for 100 steps
dt = 0.0625
for step in range(100):
    evolver.step_rk4(dt)
    
# Check constraints
H_max, M_max = evolver.compute_constraints()
print(f"Constraints: H={H_max:.2e}, M={M_max:.2e}")
```

## File Structure

```
├── src/
│   ├── bssn_evol.py          # Main BSSN evolution (0a7f)
│   ├── bssn.py               # Basic BSSN implementation (0a7f)
│   ├── poisson.py            # Poisson solver (0a7f)
│   ├── bssn_vars.py          # BSSN variable definitions (0d97)
│   ├── bssn_derivs.py        # Derivative operators (0d97)
│   ├── bssn_rhs_full.py      # Complete RHS (0d97)
│   ├── bssn_integrator.py    # RK4 integrator (0d97)
│   ├── bssn_initial_data.py  # Initial data (0d97)
│   ├── bssn_boundary.py      # Boundary conditions (0d97)
│   ├── bssn_constraints.py   # Constraint monitoring (0d97)
│   ├── bssn_losses.py        # ML loss functions (0d97, UNIQUE)
│   ├── bssn_waveform.py      # Waveform extraction (0d97, UNIQUE)
│   ├── bssn_ml_pipeline.py   # ML pipeline (0d97, UNIQUE)
│   ├── bssn_optimization.py  # Optimization (0d97, UNIQUE)
│   └── dissipation.py        # KO dissipation (bd28, UNIQUE)
├── tests/
│   ├── test_bssn_evol.py     # Evolution tests (7 tests)
│   ├── test_bssn.py          # Basic BSSN tests
│   ├── test_poisson.py       # Poisson solver tests
│   ├── test_constraints.py   # Constraint tests (3 tests)
│   └── test_autodiff.py      # Autodiff tests (2 tests)
├── refs/
│   ├── bssn_equations.md     # BSSN formulation reference
│   ├── ml_integration_api.py # ML API documentation
│   ├── warp_autodiff.py      # Warp autodiff reference
│   └── warp_fem_api.py       # Warp FEM API reference
└── merge_notes/              # Branch analysis notes
```

## Running Tests

```bash
# All evolution tests
python3 tests/test_bssn_evol.py

# Constraint tests
python3 tests/test_constraints.py

# Autodiff tests
python3 tests/test_autodiff.py
```

## Merged From

This codebase was created by merging the best components from 16 agent branches:

| Component | Source Branch | Description |
|-----------|---------------|-------------|
| Core BSSN Evolution | 0a7f | Complete evolution with RK4, dissipation |
| ML Pipeline | 0d97 | Losses, waveforms, optimization |
| Constraint Monitoring | 0d97 | H & M constraint tracking |
| Dissipation Kernel | bd28 | Kreiss-Oliger implementation |
| Initial Data | 0a7f, 0d97 | BBH, Schwarzschild punctures |
| Boundary Conditions | 0a7f, 0d97 | Sommerfeld radiative BCs |

## Requirements

- Python 3.8+
- NVIDIA Warp 1.10.1+
- NumPy

```bash
pip install warp-lang numpy
```

## License

Research code - see individual branch contributions.
