# BSSN Evolution Equations

The BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation is a conformal traceless reformulation of the ADM equations.

## Variables

The BSSN variables are:
- `φ` or `χ = e^(-4φ)`: conformal factor
- `γ̃ᵢⱼ`: conformal 3-metric (det(γ̃) = 1)
- `K`: trace of extrinsic curvature
- `Ãᵢⱼ`: traceless part of conformal extrinsic curvature
- `Γ̃ⁱ`: conformal connection functions
- `α`: lapse function
- `βⁱ`: shift vector

## Evolution Equations

### Conformal factor (using χ = e^(-4φ)):
```
∂ₜχ = 2/3 χ (α K - ∂ᵢβⁱ) + βⁱ∂ᵢχ
```

### Conformal metric:
```
∂ₜγ̃ᵢⱼ = -2α Ãᵢⱼ + βᵏ∂ₖγ̃ᵢⱼ + γ̃ᵢₖ∂ⱼβᵏ + γ̃ₖⱼ∂ᵢβᵏ - 2/3 γ̃ᵢⱼ∂ₖβᵏ
```

### Trace of extrinsic curvature:
```
∂ₜK = -γⁱⱼDᵢDⱼα + α(ÃᵢⱼÃⁱⱼ + 1/3 K²) + βⁱ∂ᵢK
```

where `DᵢDⱼα` is the Laplacian term that needs special treatment.

### Traceless conformal extrinsic curvature:
```
∂ₜÃᵢⱼ = χ [−DᵢDⱼα + α(Rᵢⱼ − 8πSᵢⱼ)]^TF 
        + α(K Ãᵢⱼ − 2ÃᵢₖÃʲₖ)
        + βᵏ∂ₖÃᵢⱼ + Ãᵢₖ∂ⱼβᵏ + Ãₖⱼ∂ᵢβᵏ - 2/3 Ãᵢⱼ∂ₖβᵏ
```

where `TF` denotes trace-free part, `Rᵢⱼ` is the conformal Ricci tensor, and `Sᵢⱼ` is the stress tensor.

### Conformal connection functions:
```
∂ₜΓ̃ⁱ = −2Ãⁱⱼ∂ⱼα + 2α(Γ̃ⁱⱼₖÃʲᵏ − 2/3 γ̃ⁱⱼ∂ⱼK − 8πγ̃ⁱⱼSⱼ)
        + βⱼ∂ⱼΓ̃ⁱ − Γ̃ⱼ∂ⱼβⁱ + 2/3 Γ̃ⁱ∂ⱼβⱼ + γ̃ʲᵏ∂ⱼ∂ₖβⁱ + 1/3 γ̃ⁱᵏ∂ⱼ∂ₖβⱼ
```

## Constraint Equations

The Hamiltonian constraint:
```
H = R + K² - ÃᵢⱼÃⁱⱼ - 16πρ = 0
```

The momentum constraints:
```
Mᵢ = Dⱼ(Ãⁱⱼ) − 2/3 ∂ⁱK − 8πSⁱ = 0
```

## Gauge Conditions

Common choices:

### 1-plus-log slicing for lapse:
```
∂ₜα = −2α K + βⁱ∂ᵢα
```

### Gamma-driver shift condition:
```
∂ₜβⁱ = B̃ⁱ
∂ₜB̃ⁱ = 3/4 ∂ₜΓ̃ⁱ − η B̃ⁱ
```

where η is a damping parameter (typically ~1-2).

## Spatial Derivatives

The conformal Ricci tensor requires:
```
Rᵢⱼ = R̃ᵢⱼ + Rᵢⱼ^φ
```

where:
- `R̃ᵢⱼ`: Ricci tensor of conformal metric (computed from Γ̃ⁱ)
- `Rᵢⱼ^φ`: terms from conformal factor

## Discretization Requirements

- 4th order centered finite differences for spatial derivatives
- Kreiss-Oliger dissipation for stability
- Sommerfeld boundary conditions for outgoing waves
- RK4 or RK3 for time integration
