# Grid Structure and Boundary Conditions

Source: SpacetimeX/CarpetX examples

## Grid Structure

### Domain Setup (BBH example: qc0.par)

```
Domain: [-16, +16]^3 M
Resolution: 64^3 base grid
Ghost zones: 1 layer
```

### Mesh Refinement (AMR)
- CarpetX: Berger-Oliger style adaptive mesh refinement
- `max_num_levels`: typically 5-7 levels
- `regrid_every`: refinement frequency (0 = no regrid, N = every N steps)
- `regrid_error_threshold`: error threshold for refinement
- Error estimator: uses "cube" region shape, scales by resolution

### Grid Properties
- **Non-periodic**: `periodic_x/y/z = no`
- **Optional symmetries**: reflection symmetry can be enabled
  - `reflection_x/y/z = yes`: lower boundary reflections
  - `reflection_upper_x/y/z = yes`: upper boundary reflections
  - BBH simulations often use octant symmetry (3 reflection planes)

### Coordinate System
- Cartesian coordinates (x, y, z)
- Centered at (0, 0, 0)
- Grid spacing: uniform on each refinement level
- Cell-centered finite differences

## Boundary Conditions

### NewRad (Radiative Boundary Conditions)

Based on Sommerfeld radiation condition for outgoing waves.

#### Basic Form
Assumes outgoing radial wave:
```
u(x,t) = u_∞ + f(r - v0*t) / r
```

This gives the boundary condition:
```
∂_t u = -v^i ∂_i u - v0 (u - u_∞) / r
```

where:
- `v^i = v0 x^i / r`: radial velocity components
- `v0`: propagation speed (typically 1 for metric variables)
- `u_∞`: asymptotic value (var0 parameter)
- `r = sqrt(x^2 + y^2 + z^2)`: coordinate radius

#### Enhanced Form (with radpower)
For Coulomb-like fall-off components:
```
u(x,t) = u_∞ + f(r - v0*t) / r + h(t) / r^n
```

Boundary RHS becomes:
```
∂_t u = -v^i ∂_i u - v0 (u - u_∞) / r + (∂_t h) / r^n
```

The `h(t)/r^n` term is extrapolated from interior points near boundary.

#### Finite Differences at Boundary
- **Tangential derivatives** (parallel to boundary): 2nd order centered
  - `∂_x u = (u(i+1) - u(i-1)) / (2*dx)`
- **Normal derivatives** (perpendicular to boundary): 2nd order asymmetric
  - Upper boundary: `∂_x u = (3*u(i) - 4*u(i-1) + u(i-2)) / (2*dx)`
  - Lower boundary: `∂_x u = -(3*u(i) - 4*u(i+1) + u(i+2)) / (2*dx)`

#### Typical Parameters
- Metric variables (gamt_ij, alpha): `var0 = flat spacetime value`, `v0 = 1.0`
- Extrinsic curvature (exKh, exAt_ij): `var0 = 0`, `v0 = 1.0`
- Shift (beta^i): `var0 = 0`, `v0 = 1.0`
- Radpower: typically 0 or 1 depending on variable

### Symmetry Boundaries

When enabled, CarpetX applies reflection symmetry:
- Even parity: `u(-x) = +u(x)` (scalars, diagonal tensors)
- Odd parity: `u(-x) = -u(x)` (vectors normal to plane, off-diagonal)

Symmetry boundaries override radiative conditions on those faces.

## Initial Data

### Puncture Initial Data
- Bowen-York extrinsic curvature
- Conformal factor from elliptic solve
- Punctures at specified positions with given masses and momenta
- Example (head-on collision):
  ```
  npunctures = 2
  mass[0] = 0.5, posz[0] = +0.5
  mass[1] = 0.5, posz[1] = -0.5
  ```
- Example (quasi-circular orbit qc0):
  ```
  npunctures = 2
  mass[0] = 0.453, posx[0] = +1.168, momy[0] = +0.333
  mass[1] = 0.453, posx[1] = -1.168, momy[1] = -0.333
  ```

### Gauge Initial Data
- Lapse: `alpha = 1` or from puncture solver
- Shift: `beta^i = 0` or from puncture solver
- Time derivatives: `dt_alpha = 0`, `dt_beta^i = 0` initially

## Output Files (Typical)
- Silo format: `out_silo_vars = "all"`
- HDF5 format: checkpoint/restart files
- ASCII format: time series at specific points
- Output frequency: `out_every = N` iterations
