# Grid Structure and Boundary Conditions for BSSN

## Grid Structure

### Cartesian Grid
- 3D uniform Cartesian grid: (nx, ny, nz) points
- Domain: typically centered at origin with symmetric bounds
- Spacing: dx = dy = dz (usually same in all directions)
- Typical resolution: 64³ to 512³ for production runs

### Ghost Zones
- Width: typically 3-4 points for 4th order stencils
- Used for: boundary conditions and inter-process communication
- Updated: after each RHS evaluation, before next step

### Mesh Refinement (optional)
- Box-in-box refinement centered on sources
- Refinement factor: typically 2:1
- Interpolation/restriction at boundaries
- Common packages: Carpet (Cactus), AMReX

## Coordinate System

Standard Cartesian coordinates (x, y, z) with:
- Origin at center of mass (for BBH)
- z-axis: initial orbital angular momentum direction
- Grid extends from -L to +L in each direction
- Outer boundary: far enough to minimize reflection

## Boundary Conditions

### 1. Radiative (Sommerfeld) Boundary Conditions

For outgoing gravitational waves:
```
∂ₜu + v·∇u + u/r = 0
```

where:
- u: any field variable
- v: wave speed (typically speed of light = 1)
- r: radial distance from origin

Discretization:
```
u^(n+1) ≈ u^n - vΔt/Δx (u^n - u^n_neighbor) - Δt·u^n/r
```

### 2. Flat Space Fall-off

For fields at large radius:
```
φ → 0         (conformal factor → 0)
γᵢⱼ → δᵢⱼ     (metric → Minkowski)
Aᵢⱼ → 0       (extrinsic curvature → 0)
K → 0         (trace of K → 0)
```

### 3. Reflection Symmetry (when applicable)

For BBH with symmetric initial data:
- Reflection across xy-plane: z → -z
- Even parity: γᵢⱼ, K, φ
- Odd parity: βᶻ, Γ̃ᶻ, Aᶻⱼ

### 4. Constraint Damping

Modified equations to suppress constraint violations:
```
∂ₜΓ̃ⁱ = ... + κ₁(2Γ̃ⁱ + Γ̃ⁱ_analytic)
∂ₜα = ... + κ₂(α - α_analytic)
```

where κ₁, κ₂ are damping parameters.

## Grid Hierarchies

For adaptive mesh refinement:

```
Level 0: Coarsest, outer boundary
  ├─ Level 1: 2x refinement
     ├─ Level 2: 4x refinement
        └─ Level 3: 8x refinement (around BHs)
```

- Prolongation: coarse to fine (interpolation)
- Restriction: fine to coarse (averaging)
- Buffer zones: between refinement levels

## Typical Parameters

```python
# Grid parameters
nx = ny = nz = 128          # Points per direction
dx = dy = dz = 0.5          # Grid spacing (M = mass units)
outer_boundary = 64.0       # Distance to outer boundary

# Boundary parameters  
ghost_width = 3             # Ghost zone width
wave_speed = 1.0            # Speed of light
fall_off_power = 1.0        # 1/r fall-off

# For BBH
BH_separation = 10.0        # Initial separation
refinement_levels = 4       # Number of AMR levels
BH_radius = 2.0             # Horizon radius (approx)
```

## Excision Regions

For black hole interiors (optional):
- Excise region inside apparent horizon
- No evolution needed inside BH
- Boundary at ~0.5-0.8 × horizon radius
- Extrapolation from exterior points
