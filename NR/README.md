# Differentiable Numerical Relativity with Warp - Branch cursor/instructions-wrapup-completion-ef5d

## Progress Summary
- **Milestone reached**: M4 (starting BBH Evolution)
- **M3 (Core BSSN) complete**
- **Key deliverables**:
  - Full BSSN variable representation (conformal decomposition)
  - 4th order finite difference operators with Kreiss-Oliger dissipation
  - RK4 time integrator with autodiff support
  - 2D Poisson solver using warp.fem
  - Flat spacetime stability verified
  - End-to-end autodiff through time evolution

## What Works
- [x] **BSSNState class**: Complete BSSN variable storage (φ, γ̃ᵢⱼ, K, Ãᵢⱼ, Γ̃ⁱ, α, βⁱ, B)
- [x] **Flat spacetime initialization**: Minkowski initial data
- [x] **4th order finite differences**: First and second derivatives in x, y, z
- [x] **Kreiss-Oliger dissipation**: 5th order numerical dissipation
- [x] **RK4 integrator**: 4th order Runge-Kutta time stepping
- [x] **Poisson solver**: FEM-based solver with Dirichlet BCs
- [x] **Autodiff**: Gradients flow through time evolution
- [ ] **Full BSSN RHS**: Currently flat-space approximation (missing curvature terms)
- [ ] **BBH initial data**: Not yet implemented
- [ ] **Constraint monitoring**: Hamiltonian/momentum constraints not computed

## Requirements
```bash
pip install warp-lang numpy
```

## Quick Start
```bash
# Run all tests
cd NR
python3 -m pytest tests/ -v

# Or run specific components
python3 tests/test_poisson.py      # Poisson equation convergence test
python3 tests/test_flat.py         # BSSN flat spacetime evolution
python3 tests/test_autodiff.py     # Autodiff gradient verification
```

## File Structure
```
NR/
├── src/
│   ├── bssn.py          # BSSNState class with all BSSN variables
│   ├── derivs.py        # 4th order finite difference operators & KO dissipation
│   ├── rhs.py           # BSSN RHS kernel (flat-space approximation)
│   ├── integrator.py    # RK4 time integrator
│   └── poisson.py       # FEM-based 2D Poisson solver
├── tests/
│   ├── test_flat.py     # Flat spacetime stability test
│   ├── test_poisson.py  # Poisson solver convergence test
│   └── test_autodiff.py # Autodiff through RK4 step test
├── refs/
│   ├── bssn_equations.md    # BSSN equation reference
│   ├── grid_structure.md    # Grid layout documentation
│   ├── time_integration.md  # Time stepping details
│   └── warp_fem_api.md      # Warp FEM API reference
├── STATE.md             # Development state tracker
├── WRAPUP_STATE.md      # Wrapup phase progress
└── README.md            # This file
```

## Implementation Details

### BSSN Variables
All variables stored on a 3D grid with symmetric tensors as 6-component arrays:

| Variable | Type | Description |
|----------|------|-------------|
| φ (phi) | scalar | Conformal factor: γᵢⱼ = e^{4φ} γ̃ᵢⱼ |
| γ̃ᵢⱼ (gamma_tilde) | sym. tensor (6) | Conformal metric (det = 1) |
| K | scalar | Trace of extrinsic curvature |
| Ãᵢⱼ (A_tilde) | sym. tensor (6) | Traceless conformal extrinsic curvature |
| Γ̃ⁱ (Gam_tilde) | vector (3) | Conformal connection functions |
| α (alpha) | scalar | Lapse function |
| βⁱ (beta) | vector (3) | Shift vector |
| Bⁱ | vector (3) | Shift auxiliary variable (Gamma-driver) |

### Numerical Methods
- **Spatial derivatives**: 4th order centered finite differences
  - First derivatives: 5-point stencil (-1, +8, -8, +1)/(12h)
  - Second derivatives: 5-point stencil (-1, +16, -30, +16, -1)/(12h²)
  - Mixed derivatives: 2nd order cross-stencil
- **Time integration**: 4th order Runge-Kutta (RK4)
- **Dissipation**: 5th order Kreiss-Oliger (σ=0.01)
- **Boundary handling**: Skip 2-point boundary layer in RHS computation

### Gauge Conditions
- **Lapse evolution**: 1+log slicing: ∂ₜα = -2αK
- **Shift evolution**: Gamma-driver: ∂ₜβⁱ = ¾Bⁱ

### Test Results
| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓ | Max \|K\|, \|φ\| = 0.0 over 100 steps |
| Poisson convergence | ✓ | 4th order convergence verified |
| Autodiff | ✓ | Gradient norm = 64.0 through RK4 step |

### Poisson Solver Details
- Uses `warp.fem` for FEM discretization
- 2D Grid2D geometry
- Polynomial basis (configurable degree)
- Conjugate gradient solve with BSR matrix format
- Dirichlet boundary conditions via projection

## Known Issues / TODOs
- **BSSN RHS incomplete**: Current RHS is a flat-space approximation. Missing:
  - Ricci tensor terms in K evolution
  - Atilde^ij * Atilde_ij term
  - Full Gamma_tilde evolution
  - Advection terms (beta^i ∂_i terms)
- **No constraint monitoring**: Hamiltonian/momentum constraints not computed
- **No BBH initial data**: Would need Bowen-York or puncture data
- **Boundary conditions**: Currently simple boundary skip (not radiative/Sommerfeld)
- **3D Poisson**: Current solver is 2D only

## Device Support
- Currently runs on CPU (`device="cpu"`)
- Warp arrays created without explicit device specification (defaults to CPU)
- See `notes/gpu_analysis.md` for GPU migration plan
