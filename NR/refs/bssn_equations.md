# BSSN Evolution Equations

Reference: McLachlan thorn (Einstein Toolkit)
Source: https://bitbucket.org/einsteintoolkit/mclachlan

## BSSN Variables

The BSSN formulation uses the following evolved variables:

1. **φ (phi)** - Conformal factor: exp(-4φ) = det(γ̃ᵢⱼ)^(-1/3) or W = exp(-2φ)
2. **γ̃ᵢⱼ (gt)** - Conformal metric: γ̃ᵢⱼ = exp(-4φ) γᵢⱼ with det(γ̃) = 1
3. **K (trK)** - Trace of extrinsic curvature: K = γⁱʲKᵢⱼ
4. **Ãᵢⱼ (At)** - Traceless conformal extrinsic curvature: Ãᵢⱼ = exp(-4φ)(Kᵢⱼ - γᵢⱼK/3)
5. **Γ̃ⁱ (Xt)** - Conformal connection functions: Γ̃ⁱ = γ̃ʲᵏΓ̃ⁱⱼₖ
6. **α (alpha)** - Lapse function
7. **βⁱ (beta)** - Shift vector

## Evolution Equations

### Conformal Factor (φ or W)
```
∂ₜφ = -1/6 α K + 1/6 βⁱ∂ᵢφ + β ⁱ∂ᵢφ

For W formulation:
∂ₜW = 1/3 W (α K - ∂ᵢβⁱ) + βⁱ∂ᵢW
```

### Conformal Metric (γ̃ᵢⱼ)
```
∂ₜγ̃ᵢⱼ = -2α Ãᵢⱼ + γ̃ᵢₖ∂ⱼβᵏ + γ̃ⱼₖ∂ᵢβᵏ - 2/3 γ̃ᵢⱼ∂ₖβᵏ + βᵏ∂ₖγ̃ᵢⱼ
```

### Trace of Extrinsic Curvature (K)
```
∂ₜK = -e⁻⁴ᶠ[γ̃ⁱʲ(∂ᵢ∂ⱼα + 2∂ᵢφ∂ⱼα) - Γ̃ⁱ∂ᵢα] 
      + α(ÃⁱⱼÃʲⁱ + K²/3) 
      + 4πα(ρ + S) + βⁱ∂ᵢK
```

### Traceless Extrinsic Curvature (Ãᵢⱼ)
```
∂ₜÃᵢⱼ = e⁻⁴ᶠ[(-∇ᵢ∇ⱼα + 2∂ᵢα∂ⱼφ + 2∂ⱼα∂ᵢφ + αRᵢⱼ)ᵀᶠ]
        + α(KÃᵢⱼ - 2ÃᵢₖÃᵏⱼ)
        + Ãᵢₖ∂ⱼβᵏ + Ãⱼₖ∂ᵢβᵏ - 2/3 Ãᵢⱼ∂ₖβᵏ
        + βᵏ∂ₖÃᵢⱼ
        - 8πα e⁻⁴ᶠ(Tᵢⱼ - γᵢⱼS/3)
```
where TF denotes tracefree part.

### Conformal Connection (Γ̃ⁱ)
```
∂ₜΓ̃ⁱ = -2Ãⁱʲ∂ⱼα
       + 2α[Γ̃ⁱⱼₖÃʲᵏ - 2/3 γ̃ⁱʲ∂ⱼK + 6Ãⁱʲ∂ⱼφ]
       + γ̃ʲᵏ∂ⱼ∂ₖβⁱ + 1/3 γ̃ⁱʲ∂ⱼ∂ₖβᵏ
       - Γ̃ⁿⱼ∂ⱼβⁱ + 2/3 Γ̃ⁱ∂ⱼβʲ
       + βʲ∂ⱼΓ̃ⁱ
       - 16πα γ̃ⁱʲSⱼ
```

### Gauge Conditions

**1+log slicing (lapse):**
```
∂ₜα = -2α K + βⁱ∂ᵢα
```
or with evolved A:
```
∂ₜα = -f α² A
∂ₜA = ∂ₜK + driver terms
```

**Gamma-driver shift:**
```
∂ₜβⁱ = C Bⁱ
∂ₜBⁱ = ∂ₜΓ̃ⁱ - η Bⁱ
```
or:
```
∂ₜβⁱ = C Γ̃ⁱ - η βⁱ
```

## Ricci Tensor Decomposition

The conformal Ricci tensor R̃ᵢⱼ is computed as:
```
R̃ᵢⱼ = -1/2 γ̃ᵏˡ∂ₖ∂ˡγ̃ᵢⱼ + γ̃ₖ₍ᵢ∂ⱼ₎Γ̃ᵏ + Γ̃ᵏΓ̃₍ᵢⱼ₎ₖ + γ̃ᵏˡ(2Γ̃ᵐₖ₍ᵢΓ̃ⱼ₎ₘˡ + Γ̃ᵐᵢₖΓ̃ₘⱼˡ)
```

Physical Ricci tensor includes φ terms:
```
Rᵢⱼ = R̃ᵢⱼ + Rᶠᵢⱼ
Rᶠᵢⱼ = -2∇̃ᵢ∇̃ⱼφ - 2γ̃ᵢⱼγ̃ᵏˡ∇̃ₖ∇̃ˡφ + 4∂ᵢφ∂ⱼφ - 4γ̃ᵢⱼγ̃ᵏˡ∂ₖφ∂ˡφ
```

## Constraint Equations

**Hamiltonian constraint:**
```
H = R - KᵢⱼKⁱʲ + K² - 16πρ = 0
```

**Momentum constraint:**
```
Mᵢ = ∇ⱼKⁱⱼ - ∇ᵢK - 8πSᵢ = 0
```

## Numerical Details (from McLachlan)

### Finite Differencing
- Default order: 4th order centered differences
- Upwind derivatives for advection terms
- Kreiss-Oliger dissipation of order fdOrder+1

### Dissipation
```
Dissipation[var] = ε_diss * D^(fdOrder+1) * var / 2^(fdOrder+2)
```
where D is the centered difference operator.

### Time Integration
Uses Method of Lines (MoL) with:
- RK4 (4th order Runge-Kutta)
- Iterated Crank-Nicolson
- Other integrators available via MoL thorn

## Boundary Conditions

### Radiative (Sommerfeld) Boundary Conditions
The "NewRad" boundary condition implements outgoing wave:
```
∂ₜu + v₀/r ∂ᵣ(r(u - u₀)) = 0
```
where:
- u₀ = asymptotic value (e.g., 1 for lapse, 0 for K)
- v₀ = wave speed (typically 1 or sqrt(harmonicF) for lapse)
- radpower = falloff exponent (typically 2)

Applied to each BSSN variable with appropriate asymptotic values:
| Variable | Asymptotic Value | Wave Speed |
|----------|------------------|------------|
| φ (W=1 or φ=0) | conformalMethod ? 1 : 0 | √(harmonicF) |
| γ̃ᵢⱼ diagonal | 1 | 1 |
| γ̃ᵢⱼ off-diag | 0 | 1 |
| Γ̃ⁱ | 0 | 1 |
| K | 0 | √(harmonicF) |
| Ãᵢⱼ | 0 | 1 |
| α | 1 | √(harmonicF) |
| βⁱ | 0 | 1 |

### Sommerfeld Condition Implementation
```c
NewRad_Apply(cctkGH, var, rhs, var0, v0, radpower);
```
Modifies the RHS at boundary points to implement the radiative condition.

## Grid Structure

### Carpet AMR (Adaptive Mesh Refinement)
Einstein Toolkit typically uses Carpet driver for:
- Multiple refinement levels with 2:1 refinement ratio
- Vertex-centered grids
- Prolongation (interpolation to finer grids)
- Restriction (averaging to coarser grids)
- Buffer zones between refinement levels

### Typical Grid Parameters
- Ghost zones: 3-4 points (for 4th order stencils)
- Refinement ratio: 2
- CFL factor: ~0.25 for RK4
- Dissipation: eps_diss ~ 0.1-0.3
