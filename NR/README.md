# Differentiable Numerical Relativity with Warp

## Progress Summary
- **Milestone reached**: M5 (ALL MILESTONES COMPLETE)
- **Key deliverables**:
  - Full BSSN formulation of Einstein's equations in NVIDIA Warp
  - 4th order finite difference spatial discretization
  - RK4 time integration with Kreiss-Oliger dissipation
  - Schwarzschild and Brill-Lindquist puncture initial data
  - 1+log slicing and Gamma-driver shift gauge conditions
  - Sommerfeld radiative boundary conditions
  - Hamiltonian and momentum constraint monitoring
  - End-to-end differentiable pipeline via Warp's autodiff
  - Gradient-based optimization of initial conditions

## What Works
- [x] **BSSN Variables**: Full 3+1 decomposition with 21 evolved fields
- [x] **4th Order FD Derivatives**: Verified O(h⁴) convergence rate
- [x] **Kreiss-Oliger Dissipation**: 6th derivative dissipation for stability
- [x] **RK4 Time Integration**: 4th order Runge-Kutta stepping
- [x] **Schwarzschild Initial Data**: Puncture representation of single BH
- [x] **Brill-Lindquist Data**: Binary black hole puncture data
- [x] **1+log Slicing**: Singularity-avoiding gauge condition
- [x] **Gamma-Driver Shift**: Dynamical shift evolution
- [x] **Sommerfeld Boundaries**: Radiative outgoing-wave conditions
- [x] **Constraint Monitoring**: Hamiltonian and momentum constraints
- [x] **Autodiff Through RHS**: Gradients through BSSN RHS computation
- [x] **Autodiff Through Evolution**: Gradients through full time stepping
- [x] **Differentiable Losses**: Constraint, stability, waveform losses
- [x] **ML Pipeline**: End-to-end differentiable for optimization/ML
- [ ] FEM-based Poisson solver: Needs warp FEM utils (non-critical)

## Requirements

```bash
pip install warp-lang numpy scipy
```

**Note**: Currently runs on CPU. GPU support requires CUDA-enabled warp installation.

## Quick Start

```bash
cd NR/src

# Test BSSN variable initialization
python3 bssn_vars.py

# Test 4th order derivatives (verify O(h⁴) convergence)
python3 bssn_derivs.py

# Test autodiff through BSSN RHS
python3 bssn_autodiff_test.py

# Run single black hole evolution (100 steps)
python3 bssn_evolution_test.py

# Test gradients through time evolution
python3 bssn_autodiff_evolution_test.py

# Test end-to-end ML pipeline
python3 bssn_ml_pipeline.py

# Test gradient-based optimization
python3 bssn_optimization.py
```

## File Structure

```
NR/
├── README.md                      # This file
├── STATE.md                       # Milestone tracking
├── WRAPUP_STATE.md               # Wrapup progress
├── src/
│   ├── bssn_vars.py              # BSSN variable definitions + BSSNGrid class
│   ├── bssn_derivs.py            # 4th order FD derivatives + KO dissipation
│   ├── bssn_rhs.py               # Simplified BSSN RHS (flat spacetime)
│   ├── bssn_rhs_full.py          # Complete BSSN RHS with Christoffels
│   ├── bssn_integrator.py        # RK4 time integration
│   ├── bssn_initial_data.py      # Schwarzschild + Brill-Lindquist punctures
│   ├── bssn_boundary.py          # Sommerfeld radiative boundaries
│   ├── bssn_constraints.py       # Hamiltonian/momentum constraint monitoring
│   ├── bssn_losses.py            # Differentiable loss functions
│   ├── bssn_waveform.py          # Gravitational waveform extraction
│   ├── bssn_optimization.py      # Gradient-based optimization
│   ├── bssn_ml_pipeline.py       # End-to-end differentiable pipeline
│   ├── bssn_autodiff_test.py     # Autodiff verification (RHS)
│   ├── bssn_autodiff_evolution_test.py  # Autodiff through evolution
│   ├── bssn_evolution_test.py    # Single BH evolution test
│   ├── poisson_solver.py         # FEM Poisson solver (WIP)
│   └── test_autodiff_diffusion.py # FEM autodiff demo (WIP)
└── refs/
    ├── bssn_equations.md         # BSSN equations from McLachlan
    ├── autodiff_mechanism.py     # Warp tape autodiff reference
    ├── mesh_field_apis.py        # Warp FEM geometry/field APIs
    ├── adaptive_grid_apis.py     # Adaptive grid APIs (CUDA-only)
    ├── ml_integration_api.py     # ML integration API documentation
    └── schwarzschild_comparison.md # Comparison with known behavior
```

## Implementation Details

### BSSN Variables (21 evolved fields)

| Variable | Description | Components |
|----------|-------------|------------|
| φ | Conformal factor: W = e^{-2φ} | 1 |
| γ̃ᵢⱼ | Conformal metric (det = 1) | 6 |
| K | Trace of extrinsic curvature | 1 |
| Ãᵢⱼ | Traceless conformal extrinsic curvature | 6 |
| Γ̃ⁱ | Conformal connection functions | 3 |
| α | Lapse function | 1 |
| βⁱ | Shift vector | 3 |

### Numerical Methods

- **Spatial derivatives**: 4th order central finite differences
  - Stencil: (-1, 8, 0, -8, 1) / 12h for first derivatives
  - Stencil: (-1, 16, -30, 16, -1) / 12h² for second derivatives
  - Ghost zones: 3 points each boundary (for 6th derivative dissipation)

- **Time integration**: Classical RK4
  - k₁ = f(t, y)
  - k₂ = f(t + dt/2, y + dt/2 · k₁)
  - k₃ = f(t + dt/2, y + dt/2 · k₂)
  - k₄ = f(t + dt, y + dt · k₃)
  - y(t+dt) = y(t) + dt/6 · (k₁ + 2k₂ + 2k₃ + k₄)

- **Dissipation**: Kreiss-Oliger 6th derivative
  - D = -ε · Δx · ∂⁶/∂x⁶ / 64
  - 7-point stencil: [1, -6, 15, -20, 15, -6, 1]
  - Typical ε = 0.1-0.3

### Gauge Conditions

- **1+log slicing**: ∂ₜα = -2αK (singularity-avoiding)
- **Gamma-driver shift**: ∂ₜβⁱ = (3/4)Bⁱ, ∂ₜBⁱ = ∂ₜΓ̃ⁱ - ηBⁱ

### Boundary Conditions

- **Sommerfeld radiative**: ∂ₜu + (u - u₀)/r + ∂ᵣu = 0
  - Implements outgoing wave condition
  - Applied at all 6 faces of computational domain

### Initial Data

- **Schwarzschild puncture**: Single BH with mass M at origin
  - Conformal factor: ψ = 1 + M/(2r)
  - φ = ln(ψ)
  
- **Brill-Lindquist**: Binary BH with masses at specified locations
  - ψ = 1 + Σᵢ Mᵢ/(2|r - rᵢ|)

### Loss Functions

| Loss | Purpose | Formula |
|------|---------|---------|
| Constraint | Physics violation | L₂(H², M₁², M₂², M₃²) |
| Stability | Prevent blowup | Penalize α<0.1, large φ, K |
| Asymptotic | Flatness at infinity | L₂(α-1, γᵢⱼ-δᵢⱼ) at boundary |
| Waveform | Match target GW | L₂(Ψ₄ - Ψ₄_target) |

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓ | Variables remain constant |
| 4th order convergence | ✓ | Rate = 4.02 |
| Constraint preservation | ✓ | H_L2 grows slowly |
| Single BH evolution | ✓ | 100 steps stable, α evolves correctly |
| Autodiff through RHS | ✓ | Non-zero gradients computed |
| Autodiff through evolution | ✓ | Gradients through 5+ timesteps |
| Gradient descent | ✓ | Loss decreases with optimization |
| End-to-end pipeline | ✓ | Full evolution + loss + gradients |

### Example Evolution Output

```
Single Schwarzschild Black Hole Evolution Test
Grid: 48x48x48, Domain: [-8M, +8M]³
Resolution: dx = 0.333M, dt = 0.033M (CFL = 0.1)

Step  |  Time  |   α_min  |   α_max  |  H_L2    |  H_max
    0 |  0.00M | 0.1340   | 0.9302   | 1.30e-02 | 1.38e+00
   50 |  1.67M | 0.1508   | 0.9310   | 2.35e-02 | 1.36e+00
  100 |  3.33M | 0.2111   | 0.9337   | 4.64e-02 | 1.45e+00

✓ Single black hole evolution stable!
```

### Example Autodiff Output

```
Gradient Through Evolution Test
Grid: 16x16x16, dx = 1.0, dt = 0.05
Evolving for 5 steps...

Final loss: 4.999e-05
∂L/∂α statistics:
  max:     1.581e-04
  mean:    9.945e-05
  nonzero: 4094/4096 points

✓ Gradients successfully computed through evolution!
```

## Known Issues / TODOs

- **FEM solvers**: `poisson_solver.py` and `test_autodiff_diffusion.py` need warp FEM example utilities (non-critical for BSSN work)
- **GPU support**: Currently CPU-only; needs device parameter changes for CUDA
- **Constraint damping**: Could add Z4c-style constraint damping for better long-term stability
- **AMR**: Adaptive mesh refinement would improve efficiency near punctures
- **Waveform extraction**: Basic implementation; could use more sophisticated Ψ₄ extraction

## Architecture Notes

The code uses NVIDIA Warp's kernel-based programming model:

```python
@wp.kernel
def compute_bssn_rhs_full_kernel(
    phi: wp.array(dtype=wp.float32),
    # ... 40+ array parameters ...
):
    tid = wp.tid()
    # Convert to 3D indices
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Compute derivatives, Christoffels, Ricci, RHS...
```

Autodiff is enabled via `wp.Tape()`:

```python
tape = wp.Tape()
with tape:
    for step in range(n_steps):
        rk4_step(grid, dt, compute_rhs)
    loss = compute_loss(grid)
tape.backward(loss=loss)
# Gradients now available in grid.alpha.grad, etc.
```

## References

- Baumgarte & Shapiro, "Numerical Relativity" (Cambridge, 2010)
- McLachlan thorn for Einstein Toolkit
- NVIDIA Warp documentation: https://nvidia.github.io/warp/
