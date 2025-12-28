# BSSN Evolution Equations

Reference: Baumgarte & Shapiro, Phys. Rev. D 59, 024007 (1998)

## Variables

The BSSN formulation evolves the following conformal variables:

- `φ` (phi): Conformal factor, defined by `γ̃ᵢⱼ = e^{-4φ} γᵢⱼ` where `det(γ̃) = 1`
- `γ̃ᵢⱼ` (gtilde): Conformal metric (traceless part)
- `K`: Trace of extrinsic curvature
- `Ãᵢⱼ` (Atilde): Traceless conformal extrinsic curvature, `Ãᵢⱼ = e^{-4φ}(Kᵢⱼ - γᵢⱼK/3)`
- `Γ̃ⁱ` (Gtilde): Conformal connection functions, `Γ̃ⁱ = γ̃ʲᵏ Γ̃ⁱⱼₖ`

## Gauge Variables

- `α` (alpha): Lapse function
- `βⁱ` (beta): Shift vector

## Evolution Equations

### Conformal factor
```
∂ₜφ = -α K/6 + βⁱ∂ᵢφ + ∂ᵢβⁱ/6
```

### Conformal metric
```
∂ₜγ̃ᵢⱼ = -2α Ãᵢⱼ + βᵏ∂ₖγ̃ᵢⱼ + γ̃ᵢₖ∂ⱼβᵏ + γ̃ⱼₖ∂ᵢβᵏ - (2/3)γ̃ᵢⱼ∂ₖβᵏ
```

### Trace of extrinsic curvature
```
∂ₜK = -γⁱʲDᵢDⱼα + α(ÃᵢⱼÃⁱʲ + K²/3) + 4πα(ρ + S) + βⁱ∂ᵢK
```

### Traceless extrinsic curvature
```
∂ₜÃᵢⱼ = e^{-4φ}[-DᵢDⱼα + αRᵢⱼ]^TF + α(KÃᵢⱼ - 2ÃᵢₖÃᵏⱼ) 
        + βᵏ∂ₖÃᵢⱼ + Ãᵢₖ∂ⱼβᵏ + Ãⱼₖ∂ᵢβᵏ - (2/3)Ãᵢⱼ∂ₖβᵏ
        - 8παe^{-4φ}Sᵢⱼ^TF
```
where `[...]^TF` denotes the trace-free part.

### Conformal connection functions
```
∂ₜΓ̃ⁱ = -2Ãⁱʲ∂ⱼα + 2α(Γ̃ⁱⱼₖÃᵏʲ - (2/3)γ̃ⁱʲ∂ⱼK + 6Ãⁱʲ∂ⱼφ - 8πγ̃ⁱʲSⱼ)
        + βʲ∂ⱼΓ̃ⁱ - Γ̃ʲ∂ⱼβⁱ + (2/3)Γ̃ⁱ∂ⱼβʲ + (1/3)γ̃ⁱʲ∂ⱼ∂ₖβᵏ + γ̃ᵏʲ∂ⱼ∂ₖβⁱ
```

## Gauge Conditions

### 1+log slicing (for lapse)
```
∂ₜα = -2αK + βⁱ∂ᵢα
```

### Gamma-driver shift condition
```
∂ₜβⁱ = (3/4)Bⁱ + βʲ∂ⱼβⁱ
∂ₜBⁱ = ∂ₜΓ̃ⁱ - ηBⁱ + βʲ∂ⱼBⁱ
```
where `η` is a damping parameter (typically `η ~ 2/M` for mass `M`).

## Constraint Equations

### Hamiltonian constraint
```
H = R + K² - KᵢⱼKⁱʲ - 16πρ = 0
```

### Momentum constraints
```
Mⁱ = DⱼKⁱʲ - DⁱK - 8πSⁱ = 0
```

## Spatial Derivatives

Use 4th-order centered finite differences:
```
∂ᵢf ≈ (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12h)

∂ᵢ∂ⱼf ≈ (4th order cross derivatives)
```

## Time Integration

Use 4th-order Runge-Kutta (RK4):
```
k₁ = f(t, y)
k₂ = f(t + dt/2, y + dt*k₁/2)
k₃ = f(t + dt/2, y + dt*k₂/2)
k₄ = f(t + dt, y + dt*k₃)
y_{n+1} = y_n + dt*(k₁ + 2k₂ + 2k₃ + k₄)/6
```

## Kreiss-Oliger Dissipation

Add artificial dissipation to control high-frequency noise:
```
∂ₜu → ∂ₜu - σ(-1)^{(p+1)/2} h^p D₊^{(p+1)/2} D₋^{(p+1)/2} u
```
For 4th-order FD, use 5th-order dissipation (p=5).

## Boundary Conditions

Typical choices:
1. Sommerfeld (radiative) outgoing wave condition
2. Extrapolation boundary conditions
3. Periodic (for testing)

## Flat Spacetime Initial Data

For testing, use Minkowski spacetime:
```
φ = 0
γ̃ᵢⱼ = δᵢⱼ
K = 0
Ãᵢⱼ = 0
Γ̃ⁱ = 0
α = 1
βⁱ = 0
```
