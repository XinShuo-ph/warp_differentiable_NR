# Differentiable Numerical Relativity with Warp - cursor/instructions-wrapup-completion-7ab8

## Progress Summary
- **Milestone reached**: M5 (Task 6 of 7)
- **Key deliverables**:
  - Complete BSSN formulation with RK4 time integration
  - Poisson solver using Warp FEM
  - Black hole initial data (single and binary punctures)
  - Autodifferentiation support through time evolution
  - 14 passing tests covering all components

## What Works
- [x] **Poisson Solver**: FEM-based solver using `warp.fem` for Dirichlet problems
- [x] **BSSN State Management**: Full 25 evolution variables (φ, K, α, γ̃ᵢⱼ, Ãᵢⱼ, Γ̃ⁱ, βⁱ, Bⁱ)
- [x] **4th Order Finite Differences**: Spatial derivatives with boundary clamping
- [x] **RK4 Time Integration**: Stable 4th order Runge-Kutta time stepping
- [x] **Kreiss-Oliger Dissipation**: 5th order dissipation for stability
- [x] **1+log Slicing**: Standard gauge condition for lapse
- [x] **Gamma-driver Shift**: Evolution of shift vector
- [x] **Sommerfeld Boundary Conditions**: Radiative outgoing-wave BCs
- [x] **Constraint Monitoring**: Hamiltonian and momentum constraints
- [x] **Gauge Wave Initial Data**: Exact solution test case
- [x] **Brill-Lindquist (Single BH)**: Puncture initial data for single black hole
- [x] **Binary Black Hole Initial Data**: Two-puncture Brill-Lindquist data
- [x] **Autodiff Support**: Verified gradient propagation through timesteps

## Requirements
```bash
pip install warp-lang numpy pytest
```

## Quick Start
```bash
# Run all tests (14 tests)
cd NR
python3 -m pytest tests/ -v

# Run specific test files
python3 -m pytest tests/test_poisson.py -v      # Poisson solver tests
python3 -m pytest tests/test_bssn.py -v         # Basic BSSN tests
python3 -m pytest tests/test_bssn_evol.py -v    # Full evolution tests

# Run individual modules (they have main blocks)
python3 src/poisson.py          # Poisson convergence test
python3 src/bssn.py             # Flat spacetime test
python3 src/bssn_evol.py        # Gauge wave evolution
```

## File Structure
```
NR/
├── src/
│   ├── __init__.py
│   ├── poisson.py         # FEM Poisson solver (M1)
│   ├── bssn.py            # BSSN state & basic RHS (M3)
│   └── bssn_evol.py       # Complete BSSN evolution (M4/M5)
├── tests/
│   ├── __init__.py
│   ├── test_poisson.py    # 3 tests: convergence, consistency, BCs
│   ├── test_bssn.py       # 4 tests: stability, derivatives, autodiff
│   └── test_bssn_evol.py  # 7 tests: RK4, gauge wave, BH initial data
├── refs/
│   ├── bssn_equations.md  # BSSN formulation reference
│   ├── warp_autodiff.py   # Autodiff API reference
│   └── warp_fem_api.py    # FEM API reference
├── STATE.md               # Development state tracker
├── WRAPUP_STATE.md        # Wrapup progress tracker
└── README.md              # This file
```

## Implementation Details

### BSSN Variables
The implementation uses the standard BSSN decomposition:

| Variable | Description | Array |
|----------|-------------|-------|
| φ | Conformal factor (e^{4φ} = ψ⁴) | `phi` |
| γ̃ᵢⱼ | Conformal metric (6 components) | `gtxx, gtxy, gtxz, gtyy, gtyz, gtzz` |
| K | Trace of extrinsic curvature | `K` |
| Ãᵢⱼ | Traceless conformal extrinsic curvature | `Atxx, Atxy, ...` |
| Γ̃ⁱ | Conformal connection functions | `Gtx, Gty, Gtz` |
| α | Lapse function | `alpha` |
| βⁱ | Shift vector | `betax, betay, betaz` |
| Bⁱ | Gamma-driver auxiliary | `Bx, By, Bz` |

### Numerical Methods
- **Spatial derivatives**: 4th order centered finite differences
  - First derivative: `(-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12h)`
  - Second derivative: `(-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2}) / (12h²)`
- **Time integration**: 4th order Runge-Kutta (RK4)
- **Dissipation**: 5th order Kreiss-Oliger, σ ∈ [0.1, 0.2]
- **Boundary conditions**: 
  - Clamped (copy nearest interior value)
  - Sommerfeld radiative (outgoing wave condition)

### Gauge Conditions
- **Lapse**: 1+log slicing `∂ₜα = -2αK + βⁱ∂ᵢα`
- **Shift**: Gamma-driver `∂ₜβⁱ = (3/4)Bⁱ`, `∂ₜBⁱ = ∂ₜΓ̃ⁱ - ηBⁱ`

### Test Results
| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓ | Errors remain < 10⁻¹⁰ for 50+ steps |
| Flat spacetime (RK4) | ✓ | Consistent results across runs |
| 4th order derivative accuracy | ✓ | Interior error < 10⁻⁴ |
| Gauge wave stability | ✓ | Lapse bounded [0.5, 2.0] for 100 steps |
| Constraint monitoring | ✓ | H, Mⁱ finite and small |
| RK4 consistency | ✓ | Identical results across runs |
| Sommerfeld BCs | ✓ | Stable with radiative boundaries |
| Brill-Lindquist (single BH) | ✓ | Pre-collapsed lapse, stable short evolution |
| Binary black hole | ✓ | Two-puncture data, stable short evolution |
| Autodiff through timestep | ✓ | Gradients propagate correctly |
| Poisson convergence | ✓ | Converges to analytical solution |
| Poisson consistency | ✓ | Reproducible results |
| Poisson BCs | ✓ | Dirichlet conditions satisfied |

## Autodifferentiation Example

```python
import warp as wp
from src.bssn_evol import BSSNEvolver

wp.init()

evolver = BSSNEvolver(nx=8, ny=8, nz=8, dx=0.125)

# Create arrays with requires_grad=True for autodiff
phi = wp.zeros((8, 8, 8), dtype=float, requires_grad=True)
loss = wp.zeros(1, dtype=float, requires_grad=True)

@wp.kernel
def compute_loss(rhs: wp.array3d(dtype=float), loss: wp.array(dtype=float)):
    i, j, k = wp.tid()
    wp.atomic_add(loss, 0, rhs[i, j, k] * rhs[i, j, k])

# Record operations on tape
tape = wp.Tape()
with tape:
    # ... compute BSSN RHS ...
    wp.launch(compute_loss, dim=(8,8,8), inputs=[rhs, loss])

# Backward pass
tape.backward(loss)
grad_phi = tape.gradients[phi]  # Gradients available!
```

## Known Issues / TODOs
- [ ] M5 Task 7: Boosted black hole initial data (not yet implemented)
- [ ] Full Γ̃ⁱ evolution RHS (currently simplified)
- [ ] Mixed second derivatives in Ãᵢⱼ RHS (currently omitted)
- [ ] GPU backend support (see `notes/gpu_analysis.md`)
- [ ] Long-term evolution stability (tested < 200 steps)
- [ ] Convergence testing with Richardson extrapolation

## References
- Baumgarte, T. W., & Shapiro, S. L. (2010). *Numerical Relativity*. Cambridge University Press.
- Alcubierre, M. (2008). *Introduction to 3+1 Numerical Relativity*. Oxford University Press.
- [NVIDIA Warp Documentation](https://nvidia.github.io/warp/)
