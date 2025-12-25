# Time Integration Scheme

Source: Einstein Toolkit with Method of Lines (MoL)

## Method of Lines (MoL)

The Einstein Toolkit uses **Method of Lines** to separate spatial and temporal discretization:
1. Spatial derivatives computed using finite differences
2. Resulting ODEs integrated in time using Runge-Kutta methods

## Standard Configuration

### RK4 (4th order Runge-Kutta)
```
MoL::ODE_Method = "RK4"
MoL::MoL_Intermediate_Steps = 4
MoL::MoL_Num_Scratch_Levels = 1
```

Classical 4-stage Runge-Kutta method:
```
k1 = f(t_n, u_n)
k2 = f(t_n + dt/2, u_n + dt/2 * k1)
k3 = f(t_n + dt/2, u_n + dt/2 * k2)
k4 = f(t_n + dt, u_n + dt * k3)

u_{n+1} = u_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

### Time Storage
- **Scratch levels**: temporary storage for RK substeps
- **Time levels**: typically 3 levels stored (n, n-1, n-2) for:
  - Current state
  - Previous state (for boundaries)
  - Earlier state (for 2nd order prolongation in time)

## AMR Time Integration

### Subcycling
With Carpet/CarpetX adaptive mesh refinement:
- Coarse grids take large timesteps
- Fine grids take smaller timesteps (factor of 2 per level)
- **Prolongation in time**: 2nd order interpolation between coarse timesteps
- **Prolongation in space**: 5th order interpolation for grid boundaries
  ```
  Carpet::prolongation_order_space = 5
  Carpet::prolongation_order_time = 2
  ```

### Buffer Zones
- Ghost zones: 3 cells (for 4th order FD + Kreiss-Oliger dissipation)
- Buffer zones: additional cells to handle subcycling
  ```
  driver::ghost_size = 3
  Carpet::use_buffer_zones = yes
  ```

## Timestep Selection

### Courant-Friedrichs-Lewy (CFL) Condition
For stability:
```
dt ≤ CFL * min(dx, dy, dz)
```

Typical values:
- BSSN: CFL ~ 0.25 - 0.5
- With Kreiss-Oliger dissipation: CFL ~ 0.25

### Per-Level Timesteps
On refinement level `l` with base spacing `dx_0`:
```
dx_l = dx_0 / 2^l
dt_l = dt_0 / 2^l
```

## Dissipation

### Kreiss-Oliger Dissipation
Added for numerical stability:
```
dt u += -eps * (-1)^(p/2+1) * h^(p-1) * D_+^p u
```

where:
- `eps`: dissipation strength (typically 0.1 - 0.5)
- `p`: order (typically 5 for 4th order FD)
- `D_+^p`: centered finite difference of order p
- Applied to all evolved variables

Standard form (4th order dissipation):
```
dt u += eps/16 * (u(i-2) - 4*u(i-1) + 6*u(i) - 4*u(i+1) + u(i+2))
```

## Alternative Methods

The MoL thorn supports multiple ODE integrators:

### Explicit Methods
- `RK2`: 2nd order Runge-Kutta (2 stages)
- `RK3`: 3rd order Runge-Kutta (3 stages)
- `RK4`: 4th order Runge-Kutta (4 stages) **[standard]**
- `RK87`: 8th order Runge-Kutta (13 stages)

### Special Methods
- `ICN`: Iterative Crank-Nicholson (implicit)
- `RK45`: Adaptive 4th/5th order with error control
- `RK65`: Adaptive 6th/5th order with error control

## Typical BBH Simulation Parameters

### Courant Factor
```
Time::dtfac = 0.25
```

### Initial Timestep Fill
```
Carpet::init_fill_timelevels = yes
```
Fills past timelevels by backwards evolution during initial setup.

### Regridding
```
CarpetRegrid2::regrid_every = 128
```
Rebuild AMR hierarchy every N iterations to track moving features.

## Numerical Stencils

### Spatial Derivatives (Interior)
- 4th order centered finite differences (SBP operators)
- Summation-by-parts for energy conservation
  ```
  SummationByParts::order = 4
  ```

### Spatial Derivatives (Boundary)
- 2nd-3rd order asymmetric stencils near boundaries
- Compatible with SBP property

### Time Derivative Application
1. Compute spatial derivatives at each point
2. Evaluate RHS of evolution equations
3. Apply boundary conditions to RHS
4. RK time integration using computed RHS
5. Apply constraints (if using constraint damping)
