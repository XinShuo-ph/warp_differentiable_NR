# Grid Structure and Boundary Conditions (McLachlan/Einstein Toolkit)

## Grid Structure

### Coordinate System
- Cartesian coordinates (x, y, z)
- Domain specified via CoordBase:
  - xmin, xmax, ymin, ymax, zmin, zmax
  - Grid spacing: dx, dy, dz

### Adaptive Mesh Refinement (AMR)
Uses Carpet infrastructure:
- Multiple refinement levels
- Box-in-box mesh refinement
- Adaptive grid following moving sources (e.g., black holes)
- Typical BBH setup: 7-9 refinement levels

### Ghost Zones
- Driver::ghost_size = 3 (for 4th order FD)
- Boundary zones: typically 3 points at each boundary
- Used for:
  - Finite differencing near boundaries
  - Inter-level communication in AMR
  - Boundary condition application

### Symmetries
Common symmetries to reduce computational domain:
- ReflectionSymmetry (across z=0 plane for BBH)
- RotatingSymmetry180 (π rotation around z-axis)
- Bitant symmetry: use only x≥0, z≥0 quadrant

### Example BBH Grid Setup
```
Domain: [0,120] x [-120,120] x [0,120]
Resolution: dx = dy = dz = 2.0
Ghost zones: 3
Boundary zones: 3
Symmetries: reflection_z, rotating180
```

## Boundary Conditions

### Outer Boundary Conditions

#### 1. NewRad (Radiative Boundary Conditions)
Most common for evolved variables:
- Based on characteristic decomposition
- Allows outgoing gravitational waves to exit cleanly
- Prevents spurious reflections
- Implementation: `ML_BSSN_Helper/src/NewRad.c`

Formula:
```
∂_t u + v₀ ∂_r u = 0  (at boundary)
```
where:
- v₀ is the characteristic speed (typically ±1)
- r is the radial coordinate

Applied to:
- Metric components (gt_ij)
- Extrinsic curvature (At_ij, trK)
- Lapse (alpha)
- Shift (beta^i)

#### 2. Sommerfeld Boundary Conditions
Alternative radiative BC:
```
∂_t u + v₀/r ∂_r u + v₀ (u - u₀)/r = 0
```

#### 3. Frozen Boundary
For some gauge quantities:
- Simple extrapolation from interior
- No evolution at boundary

### Boundary Treatment for Different Variables

#### Metric Variables (gt_ij, phi/W)
- NewRad with v₀ = 1.0
- Ensures det(gamma_tilde) = 1 at boundary

#### Extrinsic Curvature (At_ij, trK)
- NewRad with v₀ = 0.0 (flat background)
- Radiative for perturbations

#### Gauge Variables (alpha, beta^i)
- NewRad with appropriate characteristic speeds
- Lapse: typically frozen or NewRad
- Shift: NewRad or driver-based evolution

#### Conformal Connection (Xt^i)
- Computed from Gamma_tilde^i at boundary
- Not directly evolved at boundary

### Symmetry Boundaries

#### Reflection Symmetry (z = 0)
For variables that are even/odd under reflection:
- Even: gt_ij, At_ij (i,j ≠ z), phi, trK, alpha
- Odd: gt_iz, At_iz, beta^z, Xt^z

#### Rotation Symmetry (180° around z-axis)
Variables must be invariant or change sign appropriately

### Constraint Damping at Boundaries
Some formulations (e.g., CCZ4) include constraint damping:
- Adds terms proportional to constraints
- Helps maintain constraint satisfaction
- Particularly important near boundaries

## Time Integration

### Method of Lines (MoL)
- Spatial discretization separate from time integration
- RK3 or RK4 typically used
- Courant factor: CFL ~ 0.25-0.5

### Timestepping
```
Δt = CFL * min(dx, dy, dz)
```

### Subcycling in AMR
- Finer levels take smaller timesteps
- Typical ratio: Δt_fine = Δt_coarse / 2

## Dissipation

### Kreiss-Oliger Dissipation
Applied to all evolved variables:
```
Q[u] = -epsDiss * h^n * (-1)^(n/2) * D^n u
```
where:
- n = 4 for 4th order scheme (or n = 8 for 8th order)
- epsDiss ~ 0.1-0.3 (parameter)
- D^n is centered difference operator

Purpose:
- Suppress high-frequency noise
- Maintain numerical stability
- Does not affect physical modes significantly

## Initial Data

### Puncture Method (TwoPunctures)
- Conformal thin sandwich approach
- Solves constraint equations
- Creates initial data for BBH
- Parameters: masses, spins, separation, momenta

### Gauge Conditions at t=0
- Lapse: alpha = psi^(-2) (isotropic)
- Shift: beta^i = 0 initially
- Evolved according to chosen gauge

## References
- Carpet: mesh refinement infrastructure
- NewRad.c: radiative boundary conditions implementation
- MoL: Method of Lines time integrator
- TwoPunctures: BBH initial data generator
