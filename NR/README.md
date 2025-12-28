# Differentiable Numerical Relativity with Warp - instructions-wrapup-completion Branch

## Progress Summary
- **Milestone reached**: M3 Complete, M4 In Progress (38%)
- **Key deliverables**:
  - Complete BSSN evolution for flat spacetime (100+ steps, machine precision)
  - 4th order finite difference spatial derivatives
  - RK4 time integration with CFL control
  - Brill-Lindquist BBH initial data
  - Full autodiff support through PDE evolution
  - FEM Poisson solver for validation

## What Works
- [x] **Poisson Solver**: FEM-based solver with analytical validation (error < 1e-4)
- [x] **BSSN State Management**: All 22 BSSN fields (χ, γ̃ᵢⱼ, K, Ãᵢⱼ, Γ̃ⁱ, α, βⁱ)
- [x] **4th Order Derivatives**: Centered FD with boundary handling
- [x] **RK4 Integration**: Stable evolution with dt = CFL × dx
- [x] **Flat Spacetime Evolution**: 100+ steps, constraints = 0.00e+00
- [x] **Autodiff**: Gradients flow through evolution (wp.Tape verified)
- [x] **BBH Initial Data**: Brill-Lindquist punctures with physical values
- [x] **Curved RHS Framework**: BSSN RHS for non-trivial geometry (simplified)
- [ ] **Full BSSN RHS**: Complete curved spacetime terms (in progress)
- [ ] **Sommerfeld Boundaries**: Outgoing wave boundary conditions
- [ ] **Wave Extraction**: ψ₄ computation

## Requirements

```bash
pip install warp-lang numpy pytest
```

## Quick Start

```bash
# Navigate to NR directory
cd NR

# Run all tests (pytest-compatible)
python3 -m pytest tests/ -v

# Run individual test files
python3 tests/test_bssn_complete.py      # Full evolution test
python3 tests/test_bssn_autodiff.py      # Gradient verification
python3 tests/test_bbh_evolution.py      # BBH framework test
python3 tests/test_poisson_verification.py  # Poisson solver

# Run source files directly
python3 src/poisson_solver.py       # FEM Poisson test
python3 src/bssn_rk4.py             # Flat spacetime evolution
python3 src/bbh_initial_data.py     # BBH initial data
```

## File Structure

```
NR/
├── src/
│   ├── poisson_solver.py       # FEM Poisson solver (M1 validation)
│   ├── bssn_state.py           # BSSN field definitions (χ, γ̃, K, Ã, Γ̃, α, β)
│   ├── bssn_derivatives.py     # 4th order centered FD operators
│   ├── bssn_rhs.py             # BSSN RHS for flat spacetime
│   ├── bssn_rhs_full.py        # BSSN RHS for curved spacetime (M4)
│   ├── bssn_rk4.py             # RK4 time integrator
│   └── bbh_initial_data.py     # Brill-Lindquist puncture data (M4)
├── tests/
│   ├── test_poisson_verification.py  # Poisson convergence tests
│   ├── test_bssn_complete.py         # 100-step evolution test
│   ├── test_bssn_autodiff.py         # Autodiff gradient test
│   ├── test_diffusion_autodiff.py    # FEM autodiff verification
│   └── test_bbh_evolution.py         # BBH evolution framework
├── refs/
│   ├── bssn_equations.md       # Complete BSSN formulation
│   ├── grid_structure.md       # Grid and boundary documentation
│   ├── time_integration.md     # RK4 and dissipation schemes
│   ├── diffusion_autodiff.py   # Autodiff pattern reference
│   ├── mesh_field_apis.py      # Warp FEM API reference
│   └── refinement_apis.py      # Adaptive grid APIs
├── notes/
│   └── gpu_analysis.md         # GPU readiness analysis
├── STATE.md                    # Detailed progress tracking
├── PROGRESS.md                 # Milestone summaries
├── WRAPUP_STATE.md             # Wrapup validation results
└── README.md                   # This file
```

## Implementation Details

### BSSN Variables
The BSSN formulation uses conformal decomposition:
- **χ** (1 field): Conformal factor χ = e^{-4φ}
- **γ̃ᵢⱼ** (6 fields): Conformal 3-metric (symmetric tensor)
- **K** (1 field): Trace of extrinsic curvature
- **Ãᵢⱼ** (6 fields): Traceless conformal extrinsic curvature
- **Γ̃ⁱ** (3 fields): Conformal connection functions
- **α** (1 field): Lapse function
- **βⁱ** (3 fields): Shift vector

Total: 22 evolved fields per grid point

### Numerical Methods
- **Spatial derivatives**: 4th order centered finite differences
  - Interior: `(-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12h)`
  - Boundary: 1st order one-sided stencils
- **Time integration**: Classical RK4
  - CFL factor: 0.25
  - Timestep: dt = 0.25 × min(dx, dy, dz)
- **Gauge conditions**: 1+log lapse, Gamma-driver shift
- **Dissipation**: Not yet implemented (Kreiss-Oliger ready)

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓ Pass | 100 steps, |Δχ| = 0.00e+00 |
| Constraint preservation | ✓ Pass | H = M = 0.00e+00 |
| Autodiff gradient flow | ✓ Pass | wp.Tape() verified |
| Poisson solver accuracy | ✓ Pass | Error < 1e-4 |
| BBH initial data | ✓ Pass | χ ∈ [0.17, 0.93], physical |
| BBH evolution framework | ✓ Pass | RHS computes non-zero values |

### Performance (CPU)
- Grid: 32³ (32,768 points)
- Single RHS evaluation: ~10ms
- 100-step evolution: ~1s
- BBH initialization (48³): ~1.3s compile + instant run

## Known Issues / TODOs

### In Progress (M4)
- [ ] Complete BSSN RHS terms for curved spacetime
- [ ] Bowen-York momentum for spinning BHs
- [ ] Sommerfeld outgoing wave boundaries
- [ ] Gravitational wave extraction (ψ₄)
- [ ] Long-term BBH evolution validation

### Future Work
- [ ] Kreiss-Oliger dissipation for high-frequency noise
- [ ] Constraint damping terms
- [ ] Adaptive mesh refinement
- [ ] Multi-GPU parallelization
- [ ] Einstein Toolkit comparison

## Usage Examples

### Flat Spacetime Evolution
```python
from bssn_state import BSSNState
from bssn_rhs import BSSNEvolver
from bssn_rk4 import RK4Integrator

# Setup grid
nx, ny, nz = 32, 32, 32
dx = dy = dz = 0.3
dt = 0.08  # CFL = 0.25

# Initialize flat spacetime
state = BSSNState(nx, ny, nz)
state.set_flat_spacetime()

# Evolve
evolver = BSSNEvolver(state, dx, dy, dz)
integrator = RK4Integrator(evolver, dt)

for step in range(100):
    integrator.step()
    
# Result: state unchanged (machine precision)
```

### BBH Initial Data
```python
from bssn_state import BSSNState
from bbh_initial_data import create_bbh_initial_data

# Create grid centered at origin
nx, ny, nz = 64, 64, 64
L = 40.0
xmin = ymin = zmin = -L/2
dx = dy = dz = L / (nx - 1)

state = BSSNState(nx, ny, nz)
create_bbh_initial_data(state, xmin, ymin, zmin, dx, dy, dz,
                        separation=10.0, mass_ratio=1.0)

# Result: Two BH punctures with physical conformal factor
```

### Autodiff Through Evolution
```python
import warp as wp

with wp.Tape() as tape:
    evolver.compute_rhs()
    loss = compute_loss(state)

tape.backward(loss)
# Gradients available: tape.gradients[state.chi], etc.
```

## References

### Numerical Relativity
- Baumgarte & Shapiro, "Numerical Relativity" (2010)
- Alcubierre, "Introduction to 3+1 Numerical Relativity" (2008)

### BSSN Formulation
- Shibata & Nakamura, PRD 52, 5428 (1995)
- Baumgarte & Shapiro, PRD 59, 024007 (1999)

### NVIDIA Warp
- https://github.com/NVIDIA/warp
- Warp 1.10.1 documentation

---

**Branch**: `cursor/instructions-wrapup-completion-fdd9`
**Status**: M1-M3 complete, M4 in progress. All tests passing.
**Last Updated**: December 28, 2025
