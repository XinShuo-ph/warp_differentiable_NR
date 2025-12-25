# BSSN Evolution Equations
Extracted from McLachlan/Einstein Toolkit

## BSSN Variables

| Variable | Symbol | Tensor Type | Description |
|----------|--------|-------------|-------------|
| `phi` | φ | Scalar | Conformal factor: W = e^{-2φ} or φ = ln(χ^{-1/6}) |
| `gt[i,j]` | γ̃ᵢⱼ | Symmetric 3x3 | Conformal metric: det(γ̃) = 1 |
| `At[i,j]` | Ãᵢⱼ | Symmetric 3x3 | Traceless conformal extrinsic curvature |
| `Xt[i]` | Γ̃ⁱ | Vector | Conformal connection: Γ̃ⁱ = γ̃ʲᵏΓ̃ⁱⱼₖ |
| `trK` | K | Scalar | Trace of extrinsic curvature |
| `alpha` | α | Scalar | Lapse function |
| `beta[i]` | βⁱ | Vector | Shift vector |
| `A` | A | Scalar | Time derivative of lapse (optional) |
| `B[i]` | Bⁱ | Vector | Time derivative of shift (optional) |

## Evolution Equations

### Conformal factor: ∂ₜφ
```
∂ₜφ = (1/3)φ[W] or (-1/6)[phi] × (αK - ∂ᵢβⁱ) + βⁱ∂ᵢφ + dissipation
```

### Conformal metric: ∂ₜγ̃ᵢⱼ
```
∂ₜγ̃ᵢⱼ = -2α(Ãᵢⱼ - (1/3)γ̃ᵢⱼtr(Ã)[CCZ4])
        + γ̃ᵢₖ∂ⱼβᵏ + γ̃ⱼₖ∂ᵢβᵏ - (2/3)γ̃ᵢⱼ∂ₖβᵏ
        + βᵏ∂ₖγ̃ᵢⱼ + dissipation
```

### Conformal connection: ∂ₜΓ̃ⁱ
```
∂ₜΓ̃ⁱ = -2Ãⁱʲ∂ⱼα
       + 2α(Γ̃ⁱⱼₖÃʲᵏ - (2/3)γ̃ⁱʲ∂ⱼK + 6Ãⁱʲ∂ⱼφ)
       + γ̃ʲᵏ∂ⱼ∂ₖβⁱ + (1/3)γ̃ⁱʲ∂ⱼ∂ₖβᵏ
       - Γ̃ⁿʲ∂ⱼβⁱ + (2/3)Γ̃ⁿⁱ∂ⱼβʲ
       + βʲ∂ⱼΓ̃ⁱ + dissipation
       + matter terms: -16παγ̃ⁱʲSⱼ
```

### Trace of extrinsic curvature: ∂ₜK
```
∂ₜK = -e⁻⁴ᵠ(γ̃ⁱʲ(∂ᵢ∂ⱼα + 2∂ᵢφ∂ⱼα) - Γ̃ⁿⁱ∂ᵢα)
     + α(ÃⁱⱼÃʲⁱ + (1/3)K²)
     + βⁱ∂ᵢK + dissipation
     + matter terms: 4πα(ρ + S)
```

### Traceless extrinsic curvature: ∂ₜÃᵢⱼ
```
∂ₜÃᵢⱼ = e⁻⁴ᵠ[TF](-D̃ᵢD̃ⱼα + 2∂ᵢα∂ⱼφ + 2∂ⱼα∂ᵢφ + αRᵢⱼ)
       + α(KÃᵢⱼ - 2ÃᵢₖÃᵏⱼ)
       + Ãᵢₖ∂ⱼβᵏ + Ãⱼₖ∂ᵢβᵏ - (2/3)Ãᵢⱼ∂ₖβᵏ
       + βᵏ∂ₖÃᵢⱼ + dissipation
       + matter terms: -e⁻⁴ᵠα8π[TF](Tᵢⱼ)
```
where [TF] denotes trace-free projection.

### Lapse: ∂ₜα (1+log slicing)
```
∂ₜα = -αf(α)(K + αDriver(α-1))
     + βⁱ∂ᵢα + dissipation
```
Typical: f(α) = 2/α (Bona-Masso)

### Shift: ∂ₜβⁱ (Gamma-driver)
```
∂ₜβⁱ = μₛα^nBⁱ  (with B evolution)
   or = μₛα^n(Γ̃ⁱ - ηβⁱ)  (without B)
     + βʲ∂ⱼβⁱ + dissipation
```

## Auxiliary Quantities

### Christoffel symbols
```
Γ̃ⁱⱼₖ = (1/2)γ̃ⁱˡ(∂ⱼγ̃ₖˡ + ∂ₖγ̃ⱼˡ - ∂ˡγ̃ⱼₖ)
```

### Ricci tensor
```
R̃ᵢⱼ = -(1/2)γ̃ᵏˡ∂ₖ∂ˡγ̃ᵢⱼ + (1/2)γ̃ₖᵢ∂ⱼΓ̃ᵏ + (1/2)γ̃ₖⱼ∂ᵢΓ̃ᵏ
      + (1/2)Γ̃ⁿᵏΓ̃ᵢⱼₖ + (1/2)Γ̃ⁿᵏΓ̃ⱼᵢₖ
      + Γ̃ᵏᵢₗΓ̃ˡⱼₖ + ...

Rᵢⱼ = R̃ᵢⱼ + Rᵠᵢⱼ
```
where Rᵠ is the conformal contribution.

## Constraints

### Hamiltonian constraint
```
H = R + K² - KᵢⱼKⁱʲ - 16πρ = 0
```

### Momentum constraint
```
Mⁱ = ∂ⱼKⁱʲ - ∂ⁱK - 8πSⁱ = 0
```

## Numerical Implementation

### Derivative order
- Default: 4th order finite differences
- Higher order: 6th or 8th order available

### Dissipation (Kreiss-Oliger)
```
Dissipation[u] = ε × (-1)^(n+1) × (Δx)^(2n-1) × ∂^(2n)/∂x^(2n) u
```
Typical: n=2 or n=3 for 4th/6th order

### Time integration
- Method of Lines (MoL) with RK4
- Courant factor typically 0.25-0.5

## References
- PRD 62, 044034 (2000) - BSSN formulation
- PRD 67, 084023 (2003) - Gamma driver shift
- Baumgarte & Shapiro, Phys. Rept. 376 (2003) 41-131
