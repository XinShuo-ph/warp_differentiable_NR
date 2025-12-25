# BSSN Grid Structure and Boundary Conditions

## Grid Structure

### Vertex-Centered vs Cell-Centered
Most NR codes use vertex-centered grids where variables are stored at grid points.

### Typical Grid Layout
```
Grid dimensions: (Nx, Ny, Nz)
Grid spacing: dx = dy = dz = h
Physical domain: [-L, L]^3 or [0, L]^3
```

### Ghost Zones
```
Typical: 3-4 ghost zones per boundary for 4th-order stencils
Grid with ghosts: (Nx + 2*ng, Ny + 2*ng, Nz + 2*ng)
```

### Mesh Refinement (Carpet/AMR)
- Box-in-box refinement with 2:1 ratio
- Finer grids near the sources (black holes)
- Typical structure: 8-10 refinement levels
- Buffer zones between refinement levels

## Boundary Conditions

### Sommerfeld (Radiative) Boundary Conditions
Most common for outer boundary. Assumes outgoing waves:
```
(partial_t + partial_r + u/r) * f = 0

where:
  f = any evolved variable
  r = radial distance from origin
  u = f - f_0 (deviation from background value)
```

Implementation:
```
f(t+dt, r_max) = f(t, r_max) - dt * (partial_r(f) + f/r)
```

### Robin Boundary Conditions
For specific falloff behavior:
```
partial_r(f) + n/r * f = 0

where n is the falloff power (e.g., n=1 for 1/r falloff)
```

### Reflection Symmetry (Equatorial/Bitant/Octant)
For symmetric configurations:
```
Bitant symmetry (z -> -z):
  phi, K, A_xx, A_yy, A_zz, gamma_xx, gamma_yy, gamma_zz: even
  A_xz, A_yz, beta^z, Gamma^z: odd

Octant symmetry (full 8-fold):
  Apply at all three reflection planes
```

### Periodicity (Testing only)
For convergence tests with periodic solutions:
```
f(x + L) = f(x)
```

## Initial Data

### Black Hole Initial Data

#### Brill-Lindquist (time-symmetric)
```
psi = 1 + sum_i (m_i / (2*|r - r_i|))   # conformal factor
gamma_ij = psi^4 * delta_ij              # metric
K_ij = 0                                  # time-symmetric
```

#### Bowen-York (spinning/boosted)
```
A_ij^BY = (3/(2*r^2)) * [P_i*n_j + P_j*n_i - (delta_ij - n_i*n_j)*P^k*n_k]
        + (3/r^3) * [epsilon_ijk * S^k * n_j + epsilon_ijk * S^k * n_i]

Solve Hamiltonian constraint for conformal factor
```

#### Puncture Data (standard BBH)
```
psi = 1 + sum_i (m_i / (2*|r - r_i|)) + u   # u from solving constraint

chi = 1/psi^4  (for chi formulation)
W = 1/psi^2    (for W formulation)
```

## Puncture Tracking

### Moving Punctures
Evolution variables regularized at punctures:
- Use chi or W instead of phi near punctures
- chi, W -> 0 at punctures (regular behavior)

### Shift Condition for Punctures
Gamma-driver with eta ~ 2/M for stable puncture motion

## Typical Parameters (BBH)

```
Domain: [-400M, 400M]^3 (M = total mass)
Base resolution: h ~ 2M
Finest resolution: h ~ M/64 near punctures
Outer boundary: r ~ 300-500M
Refinement levels: 8-10
```
