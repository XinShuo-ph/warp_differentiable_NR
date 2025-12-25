# BSSN Evolution Equations (McLachlan Implementation)

## Overview
The BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation is a reformulation of the Einstein equations for numerical relativity. McLachlan implements BSSN using the Kranc code generation system.

**References:**
- PRD 62 044034 (2000) - Original BSSN paper
- PRD 67 084023 (2003) - Improvements
- Phys. Rept. 376 (2003) 41-131 - Baumgarte & Shapiro review
- gr-qc:1106.2254 (2011) - CCZ4 formulation

## Evolved Variables

### Conformal metric
- `phi` or `W = exp(-2*phi)`: conformal factor
- `gt[i,j]`: conformal 3-metric (det(gt) = 1)

### Extrinsic curvature
- `At[i,j]`: traceless conformal extrinsic curvature
- `trK`: trace of extrinsic curvature

### Gauge variables
- `alpha`: lapse function
- `beta[i]`: shift vector
- `A`: time derivative of lapse (optional)
- `B[i]`: auxiliary variable for shift (optional)

### Connection functions
- `Xt[i]` or `Gamma[i]`: contracted Christoffel symbols

### CCZ4 additions (optional)
- `Theta`: CCZ4 auxiliary variable

## Evolution Equations

### Conformal Factor
```
∂_t φ = -1/6 * α * K + 1/6 * ∂_i β^i + β^i ∂_i φ + dissipation
```
(with W formulation: ∂_t φ = +1/3 φ * (α * K - ∂_i β^i))

### Conformal Metric
```
∂_t g̃_ij = -2 α Ã_ij + g̃_ik ∂_j β^k + g̃_jk ∂_i β^k - 2/3 g̃_ij ∂_k β^k
          + β^k ∂_k g̃_ij + dissipation
```

### Conformal Traceless Extrinsic Curvature
```
∂_t Ã_ij = e^{-4φ} [Ãs_ij - 1/3 g_ij tr(Ãs)]
         + α (K Ã_ij - 2 Ã_ik Ã^k_j)
         + Ã_ik ∂_j β^k + Ã_jk ∂_i β^k - 2/3 Ã_ij ∂_k β^k
         + β^k ∂_k Ã_ij + dissipation
         - α * 8π * e^{-4φ} [T_ij - 1/3 g_ij tr(T)]  (matter)

where:
  Ãs_ij = -D_i D_j α + 2 ∂_i α ∂_j φ + 2 ∂_j α ∂_i φ + α R_ij
```

### Trace of Extrinsic Curvature
```
∂_t K = -e^{-4φ} [g̃^ij (∂_i ∂_j α + 2 ∂_i φ ∂_j α) - Γ^i ∂_i α]
      + α (Ã_ij Ã^ij + 1/3 K^2)
      + β^i ∂_i K + dissipation
      + 4π α (ρ + tr(S))  (matter)
```

### Contracted Christoffel Symbols
```
∂_t Γ̃^i = -2 Ã^ij ∂_j α
         + 2 α (Γ̃^i_jk Ã^jk - 2/3 g̃^ij ∂_j K + 6 Ã^ij ∂_j φ)
         + g̃^jk ∂_j ∂_k β^i + 1/3 g̃^ij ∂_j ∂_k β^k
         - Γ̃^j ∂_j β^i + 2/3 Γ̃^i ∂_j β^j
         + β^j ∂_j Γ̃^i + dissipation
         - 16π α g̃^ij S_j  (matter)
```

### Lapse Function
Harmonic slicing:
```
∂_t α = -f α^n (K + α_driver (α - 1)) + β^i ∂_i α + dissipation

where:
  f = harmonic factor (typically 1 or 2)
  n = harmonic exponent (typically 1 or 2)
  α_driver = damping coefficient
```

### Shift Vector
Gamma driver:
```
∂_t β^i = B^i  (or: shiftGammaCoeff * α^{shiftAlphaPower} * (Γ̃^i - β_driver β^i))

∂_t B^i = ∂_t Γ̃^i - β_driver B^i
```

Harmonic:
```
∂_t β^i = -g̃^ij α (2 α ∂_j φ + ∂_j α + 1/2 α ∂_j ln(det g) - α g̃^kl ∂_l g̃_jk)
```

## Spatial Derivatives

McLachlan uses finite differences with configurable order (typically 4th or 8th order).

### Standard derivatives
```
∂_i f = (f[i+1] - f[i-1]) / (2 dx)  (2nd order centered)
```

### Upwind derivatives (for advection)
```
β^i ∂_i f = upwind(β^i, f, i)
```

### Dissipation (Kreiss-Oliger)
```
dissipation[f] = ε_diss * (-1)^{n/2} (Δx)^{n-1} ∂^n f / ∂x^n

where:
  n = dissipation order (typically 5)
  ε_diss = dissipation strength parameter
```

## Auxiliary Quantities

### Conformal Christoffel Symbols
```
Γ̃^i_jk = 1/2 g̃^il (∂_j g̃_lk + ∂_k g̃_lj - ∂_l g̃_jk)
```

### Contracted Christoffel (from metric)
```
Γ̃^i = g̃^jk Γ̃^i_jk
```

### Covariant derivatives
```
D_i f = ∂_i f
D_i V^j = ∂_i V^j + Γ̃^j_ik V^k
D_i V_j = ∂_i V_j - Γ̃^k_ij V_k
```

### Ricci Tensor
```
R_ij = R̃_ij + D_i D_j φ + g̃_ij g̃^kl D_k D_l φ
     - 2 g̃^kl D_k φ D_l φ + 2 g̃_ij g̃^kl ∂_k φ ∂_l φ
     + higher order terms involving Ã_ij and K
```

where R̃_ij is the conformal Ricci tensor.

## Constraints

### Hamiltonian Constraint
```
H = R + K^2 - A_ij A^ij - 16π ρ = 0
```

### Momentum Constraint
```
M_i = D_j (A^j_i - g^j_i K) - 8π S_i = 0
```

These are not evolved but monitored to check solution quality.

## Matter Terms

When coupling to matter (hydrodynamics):
- `ρ`: energy density
- `S_i`: momentum density  
- `S_ij`: stress tensor
- `trS = g^ij S_ij`: trace of stress

Matter terms appear in K, Ã_ij, and Γ̃^i evolution.

## Numerical Implementation Details

### Time Integration
Typically Method of Lines (MoL) with RK4 or similar:
1. Compute RHS of all evolution equations
2. Update variables using RK substeps
3. Apply boundary conditions
4. Enforce constraints if needed

### Boundary Conditions
- Inner boundaries: excision or regularization
- Outer boundaries: radiative (Sommerfeld), flat, or constraint-preserving
- Symmetry boundaries: reflection symmetry often used

### Grid Structure
- Adaptive Mesh Refinement (AMR) using Carpet
- Multiple refinement levels near compact objects
- Coarser grid at large distances
- Typical resolutions: 0.1M to 2M (M = total mass)

## Conformal Method Choices

### phi method (cmPhi=0)
- Evolve φ directly
- det(g̃) = 1 enforced

### W method (cmW=1)  
- Evolve W = exp(-2φ)
- Numerically more stable
- det(g̃) = 1 enforced

## Formulation Variants

### BSSN (fBSSN=0)
- Standard BSSN formulation
- Most commonly used

### CCZ4 (fCCZ4=1)
- Conformal and covariant Z4 system
- Better constraint preservation
- Adds Θ variable and Z terms
- Damping parameters: dampk1, dampk2
