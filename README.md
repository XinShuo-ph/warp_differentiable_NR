# Differentiable BSSN Numerical Relativity

A production-ready differentiable BSSN numerical relativity implementation using NVIDIA Warp.

## Overview

This codebase implements the BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation of Einstein's equations using NVIDIA Warp for GPU-accelerated computation with automatic differentiation support.

## Features

### Core Evolution
- **BSSN Variables**: Complete set (φ, χ, γ̄ᵢⱼ, Āᵢⱼ, K, Γ̄ⁱ)
- **4th Order Finite Differences**: Spatial derivatives with 6th order Kreiss-Oliger dissipation
- **RK4 Time Integration**: Stable time stepping
- **Constraint Monitoring**: Hamiltonian and momentum constraint tracking

### Initial Data
- Flat spacetime (test case)
- Gauge wave (analytic test)
- Brill-Lindquist single puncture black hole
- Binary black hole (two punctures)

### Boundary Conditions
- Sommerfeld radiative boundary conditions
- 1+log slicing for lapse
- Gamma-driver for shift

### ML Integration (Unique)
- `bssn_ml_pipeline.py`: End-to-end differentiable pipeline
- `bssn_losses.py`: Physics-informed loss functions
- `bssn_optimization.py`: Gradient-based parameter optimization
- `bssn_waveform.py`: Gravitational waveform extraction
- Full autodiff support via wp.Tape()

## Installation

```bash
pip install warp-lang numpy pytest
```

## Quick Start

```python
import warp as wp
from src.bssn_evol import BSSNEvolver

wp.init()

# Create evolver with 32^3 grid
evolver = BSSNEvolver(N=32, domain_size=10.0)

# Initialize with flat spacetime
evolver.init_flat()

# Evolve for 100 steps
evolver.evolve(n_steps=100, dt=0.01)
```

## Running Tests

```bash
# Run all tests
python tests/test_bssn_evol.py      # Main BSSN evolution tests (7 tests)
python tests/test_bssn_autodiff.py  # Autodiff verification (2 tests)

# Individual tests
python tests/test_poisson.py        # Poisson solver test
python tests/test_bssn.py           # Basic BSSN tests
```

## Test Results

All 9 tests passing:

```
BSSN Evolution Tests:
  ✓ Flat spacetime stable with RK4
  ✓ Gauge wave stable
  ✓ Constraint monitoring works
  ✓ RK4 consistent
  ✓ Sommerfeld BCs stable
  ✓ Brill-Lindquist stable
  ✓ Binary BH stable

Autodiff Tests:
  ✓ Autodiff infrastructure works
  ✓ Gradients through evolution steps
```

## File Structure

```
├── src/
│   ├── bssn_evol.py          # Complete BSSN evolution system (main)
│   ├── bssn.py               # Basic BSSN implementation
│   ├── bssn_ml_pipeline.py   # ML integration pipeline
│   ├── bssn_losses.py        # Differentiable loss functions
│   ├── bssn_optimization.py  # Gradient-based optimization
│   ├── bssn_waveform.py      # Waveform extraction
│   ├── bssn_constraints.py   # Constraint monitoring
│   ├── dissipation.py        # Kreiss-Oliger dissipation
│   ├── dissipation_kernel.py # Dissipation kernel
│   └── poisson.py            # Poisson solver
├── tests/
│   ├── test_bssn_evol.py     # Main evolution tests
│   ├── test_bssn_autodiff.py # Autodiff tests
│   ├── test_bssn.py          # Basic BSSN tests
│   └── test_poisson.py       # Poisson solver tests
├── refs/
│   ├── bssn_equations.md     # BSSN formulation reference
│   ├── warp_autodiff.py      # Warp autodiff documentation
│   └── warp_fem_api.py       # Warp FEM API reference
└── merge_notes/              # Merge documentation
```

## BSSN Components

| Component | File | Description |
|-----------|------|-------------|
| Variables | `bssn_evol.py` | φ, χ, γ̄ᵢⱼ, Āᵢⱼ, K, Γ̄ⁱ, α, βⁱ |
| Derivatives | `bssn_evol.py` | 4th order FD + 6th order KO dissipation |
| RHS | `bssn_evol.py` | Complete BSSN evolution equations |
| Integration | `bssn_evol.py` | RK4 time stepping |
| Constraints | `bssn_constraints.py` | Hamiltonian & momentum |
| ML Pipeline | `bssn_ml_pipeline.py` | Differentiable end-to-end pipeline |

## Physics

### BSSN Formulation

The code implements the standard BSSN decomposition:

- **Conformal factor**: χ = e^{-4φ}
- **Conformal metric**: γ̄ᵢⱼ with det(γ̄) = 1
- **Traceless extrinsic curvature**: Āᵢⱼ
- **Conformal connection**: Γ̄ⁱ = γ̄ʲᵏΓ̄ⁱⱼₖ

### Gauge Conditions

- **Lapse**: 1+log slicing: ∂ₜα = -2αK
- **Shift**: Gamma-driver: ∂ₜβⁱ = (3/4)Γ̄ⁱ - ηβⁱ

### Dissipation

Kreiss-Oliger dissipation for stability:
```
D_KO = -σ/dx * (f_{i+2} - 4f_{i+1} + 6f_i - 4f_{i-1} + f_{i-2})
```

## Merge Summary

This codebase was created by merging the best components from 16 development branches:

- **Base**: Branch 0a7f (complete BSSN evolution, 14 tests)
- **ML Pipeline**: Branch 0d97 (unique differentiable pipeline)
- **Dissipation**: Branch bd28 (modular KO dissipation)
- **Tests**: Branches c633, 9052 (comprehensive validation)

## License

MIT License
