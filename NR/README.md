# Differentiable Numerical Relativity with Warp

## Progress Summary
- **Milestone reached**: M5 (Complete)
- **Key deliverables**:
  - Full BSSN formulation with 24 evolved fields
  - 4th order finite difference spatial derivatives
  - 6th order Kreiss-Oliger dissipation
  - RK4 time integration
  - 1+log slicing and Gamma-driver shift conditions
  - Brill-Lindquist puncture initial data
  - Sommerfeld radiative boundary conditions
  - Hamiltonian constraint monitoring
  - Full autodiff support through evolution

## What Works
- [x] Poisson solver: FEM-based elliptic solver with CG iteration
- [x] BSSN variable storage: 24 fields (chi, γ̃ᵢⱼ, K, Ãᵢⱼ, Γ̃ⁱ, α, βⁱ, Bⁱ)
- [x] 4th order FD derivatives: ∂₁, ∂₂, mixed partials
- [x] 6th order KO dissipation: Numerical stability filtering
- [x] RK4 integration: 4th order accurate time stepping
- [x] Flat spacetime evolution: Stable 200+ timesteps
- [x] Puncture initial data: Brill-Lindquist single BH
- [x] 1+log slicing: ∂ₜα = -2αK
- [x] Gamma-driver shift: ∂ₜβⁱ = ¾Bⁱ, ∂ₜBⁱ = ∂ₜΓ̃ⁱ - ηBⁱ
- [x] Sommerfeld BCs: Radiative outer boundaries
- [x] Constraint monitor: H_L2 and H_Linf norms
- [x] Autodiff: Gradients through full BSSN RHS (wp.Tape)
- [x] Single puncture evolution: Stable 50+ timesteps

## Requirements

```bash
pip install warp-lang numpy pytest
```

Tested with:
- Python 3.12
- Warp 1.10.1
- NumPy 2.3.5

## Quick Start

```bash
# Run all tests
cd NR
python3 -m pytest tests/ -v

# Run specific test suites
python3 -m pytest tests/test_warp_basic.py -v          # Basic Warp functionality
python3 -m pytest tests/test_poisson_analytical.py -v  # Poisson solver
python3 -m pytest tests/test_bssn_evolution.py -v      # BSSN flat space evolution
python3 -m pytest tests/test_puncture_evolution.py -v  # Puncture BH evolution
python3 -m pytest tests/test_long_evolution.py -v      # Long evolution + autodiff

# Run individual source modules (with built-in tests)
cd src
python3 bssn_variables.py     # Test BSSN variable initialization
python3 bssn_derivatives.py   # Test 4th order FD accuracy
python3 bssn_rhs.py           # Test simplified RHS (flat spacetime)
python3 bssn_rhs_full.py      # Test full RHS
python3 bssn_integrator.py    # Test RK4 integrator
python3 bssn_initial_data.py  # Test Brill-Lindquist data
python3 bssn_boundary.py      # Test Sommerfeld BCs
python3 bssn_constraints.py   # Test constraint monitor
python3 poisson_solver.py     # Test Poisson solver
```

## File Structure

```
NR/
├── README.md                     # This file
├── STATE.md                      # Development milestone state
├── WRAPUP_STATE.md               # Wrapup validation state
├── instructions_wrapup.md        # Wrapup instructions (read-only)
├── m1_tasks.md - m5_tasks.md     # Milestone task definitions
│
├── src/
│   ├── bssn_variables.py         # 24 BSSN field definitions (BSSNFields class)
│   ├── bssn_derivatives.py       # 4th order FD + KO dissipation kernels
│   ├── bssn_rhs.py               # Simplified BSSN RHS (subset of equations)
│   ├── bssn_rhs_full.py          # Full BSSN RHS with gauge evolution
│   ├── bssn_integrator.py        # RK4 time integration
│   ├── bssn_initial_data.py      # Brill-Lindquist puncture data
│   ├── bssn_boundary.py          # Sommerfeld boundary conditions
│   ├── bssn_constraints.py       # Hamiltonian constraint monitor
│   └── poisson_solver.py         # FEM Poisson solver (warp.fem)
│
├── tests/
│   ├── test_warp_basic.py        # Basic Warp kernel test
│   ├── test_poisson_analytical.py # Poisson convergence test
│   ├── test_bssn_evolution.py    # Flat spacetime + autodiff tests
│   ├── test_puncture_evolution.py # Single puncture evolution
│   └── test_long_evolution.py    # 100+ step evolution + autodiff
│
├── refs/
│   ├── bssn_equations.md         # BSSN equation reference
│   ├── bssn_grid_bc.md           # Grid and BC documentation
│   ├── warp_autodiff.py          # Warp autodiff examples
│   ├── warp_fem_apis.py          # Warp FEM API reference
│   └── warp_refinement_apis.py   # Warp mesh refinement reference
│
└── notes/
    └── gpu_analysis.md           # GPU porting analysis
```

## Implementation Details

### BSSN Variables (24 fields)

| Variable | Count | Description |
|----------|-------|-------------|
| χ (chi) | 1 | Conformal factor: χ = e^(-4φ) = ψ^(-4) |
| γ̃ᵢⱼ | 6 | Conformal 3-metric (symmetric) |
| K | 1 | Trace of extrinsic curvature |
| Ãᵢⱼ | 6 | Traceless conformal extrinsic curvature |
| Γ̃ⁱ | 3 | Contracted Christoffel symbols |
| α | 1 | Lapse function |
| βⁱ | 3 | Shift vector |
| Bⁱ | 3 | Gamma-driver auxiliary variable |

### Numerical Methods

- **Spatial derivatives**: 4th order centered finite differences
  - 1st derivative: `(-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / 12h`
  - 2nd derivative: `(-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2}) / 12h²`
  
- **Time integration**: Classical RK4
  - y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
  
- **Dissipation**: 6th order Kreiss-Oliger
  - σ * (-f_{i+3} + 6f_{i+2} - 15f_{i+1} + 20f_i - 15f_{i-1} + 6f_{i-2} - f_{i-3}) / 64

- **Ghost zones**: 3 points per boundary (required for KO stencil)

- **Gauge conditions**:
  - 1+log slicing: ∂ₜα = -2αK + βⁱ∂ᵢα
  - Gamma-driver: ∂ₜβⁱ = ¾Bⁱ, ∂ₜBⁱ = ∂ₜΓ̃ⁱ - ηBⁱ

- **Boundary conditions**: Sommerfeld (radiative)
  - f → f₀ + (f_interior - f₀)(r_int/r)^n

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓ | Stable 200+ steps, deviation < 10⁻¹⁰ |
| Constraint preservation | ✓ | H_L2, H_Linf < 10⁻¹⁰ for flat space |
| Puncture evolution | ✓ | Single BH stable 50+ steps |
| Long evolution | ✓ | 100+ steps with constraint monitoring |
| Autodiff | ✓ | Gradients through full BSSN RHS |

### Autodiff Usage

```python
import warp as wp
wp.set_module_options({"enable_backward": True})

# Create fields with requires_grad=True
chi = wp.zeros((n, n, n), dtype=float, requires_grad=True)
loss = wp.zeros(1, dtype=float, requires_grad=True)

# Record operations
tape = wp.Tape()
with tape:
    wp.launch(compute_bssn_rhs, ...)
    wp.launch(compute_loss, ...)

# Backward pass
tape.backward(loss)

# Access gradients
chi_grad = tape.gradients[chi]
```

## Known Issues / TODOs

- [ ] **GPU Support**: Currently CPU-only; see `notes/gpu_analysis.md` for porting requirements
- [ ] **Full Ricci tensor**: Constraint monitor uses simplified form (no curvature terms)
- [ ] **Binary black holes**: Only single puncture tested; two-puncture setup needs work
- [ ] **Mesh refinement**: Not implemented; would need Warp's AMR features
- [ ] **Wave extraction**: No gravitational wave extraction (Ψ₄, etc.)
- [ ] **Test warnings**: Some tests return values instead of using assertions only

## Physics Notes

The BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation is a 3+1 decomposition of Einstein's equations that is well-suited for numerical evolution. Key features:

1. **Conformal decomposition**: Physical metric decomposed as γᵢⱼ = χ⁻¹γ̃ᵢⱼ
2. **Trace separation**: Extrinsic curvature split into trace K and traceless Ãᵢⱼ
3. **First-order variables**: Γ̃ⁱ promoted to evolved variable for stability
4. **Gauge freedom**: Lapse α and shift βⁱ freely specifiable

The formulation admits constraint-damping when implemented correctly, making it robust for long-term evolution of spacetimes containing black holes.
