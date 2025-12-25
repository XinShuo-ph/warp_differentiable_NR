# Time Integration in Einstein Toolkit / McLachlan

## Method of Lines (MoL)

Einstein Toolkit uses the **Method of Lines** approach:
1. Spatial discretization → system of ODEs
2. Time integration with ODE solver

### MoL Thorn
The `MoL` (Method of Lines) thorn provides time integrators.

## Time Integration Schemes

### RK4 (4th-order Runge-Kutta)
Most common for BSSN evolution:

```
u^(1) = u^n + (dt/2) * RHS(u^n)
u^(2) = u^n + (dt/2) * RHS(u^(1))
u^(3) = u^n + dt * RHS(u^(2))
u^{n+1} = u^n + (dt/6) * [RHS(u^n) + 2*RHS(u^(1)) + 2*RHS(u^(2)) + RHS(u^(3))]
```

**Properties:**
- 4 RHS evaluations per step
- 4th order accurate in time
- Stable for small enough dt
- No memory of previous steps

### RK3 (3rd-order Runge-Kutta)
Alternative, cheaper option:

```
u^(1) = u^n + dt * RHS(u^n)
u^(2) = u^n + (dt/4) * [RHS(u^n) + RHS(u^(1))]
u^{n+1} = u^n + (dt/6) * [RHS(u^n) + RHS(u^(1)) + 4*RHS(u^(2))]
```

**Properties:**
- 3 RHS evaluations per step
- 3rd order accurate
- Slightly less stable than RK4

### RK2 (2nd-order Runge-Kutta / Midpoint)
Rarely used for production:

```
u^(1) = u^n + (dt/2) * RHS(u^n)
u^{n+1} = u^n + dt * RHS(u^(1))
```

### ICN (Iterative Crank-Nicholson)
Implicit method, rarely used for BSSN:

```
u^{n+1,0} = u^n + dt * RHS(u^n)
u^{n+1,k+1} = u^n + (dt/2) * [RHS(u^n) + RHS(u^{n+1,k})]
```
Iterate until convergence.

## Time Step Selection

### CFL Condition
```
dt ≤ CFL * min(dx, dy, dz) / v_max
```

where:
- CFL = Courant factor (typically 0.25 - 0.5)
- v_max = maximum characteristic speed (~1 for light)

### AMR Time Stepping
With mesh refinement:

**Subcycling:**
Each level evolves with its own time step:
```
dt_level(l) = dt_level(0) / 2^l
```

**Time refinement factor:**
Typically 2 (refinement in both space and time).

**Synchronization:**
- Coarse level takes 1 step
- Next finer level takes 2 steps
- Finest level takes many steps
- All levels synchronized at coarse time steps

**Prolongation in time:**
When coarse level advances, prolongate to fine level initial data.

## MoL Registration

McLachlan registers variables with MoL:

### Evolved Variables
```c
MoLRegisterEvolved(phi_gf, phirhs_gf);
MoLRegisterEvolved(gt11_gf, gt11rhs_gf);
MoLRegisterEvolved(At11_gf, At11rhs_gf);
// ... etc for all evolved variables
```

Each evolved variable paired with its RHS.

### Constrained Variables
```c
MoLRegisterConstrained(alpha_gf);
MoLRegisterConstrained(beta1_gf);
```

Not evolved by MoL (set by gauge conditions).

### Save and Restore
```c
MoLRegisterSaveAndRestore(temp_var);
```

For temporary storage needed across RK substeps.

## Evolution Schedule

### Typical Evolution Step

1. **CCTK_PRESTEP**
   - Prepare for step
   - Set up temporary storage

2. **MoL_Evolution** (managed by MoL thorn)
   - **MoL_StartLoop**
     - Save initial data
   
   - For each RK substep:
     - **MoL_PreStep**
       - Update time level
     
     - **MoL_CalcRHS** ← McLachlan computes RHS here
       - `ML_BSSN_EvolutionInterior`
       - `ML_BSSN_EvolutionBoundary`
       - Apply dissipation
     
     - **MoL_PostStep**
       - RK update: u^{new} = u^{old} + k * RHS
       - Apply boundary conditions
       - Enforce constraints if needed
   
   - **MoL_PostRHS**
     - Finalize RK step
     - Copy result to main variables

3. **CCTK_POSTSTEP**
   - Analysis
   - Output
   - Constraint calculation

### MoL Parameters
```
MoL::ODE_Method = "RK4"
MoL::MoL_Intermediate_Steps = 4
MoL::MoL_Num_Scratch_Levels = 1
```

## Dissipation

### Kreiss-Oliger Dissipation
Added to RHS to damp high-frequency noise:

```
dissipation[f] = ε * (-1)^{n/2} (Δx)^{n-1} ∂^n f / ∂x^n
```

Typically:
- n = 5 (5th derivative)
- ε = 0.1 - 0.5 (strength parameter)

Applied at each RK substep for stability.

### Implementation
```
+ Dissipation[phi]
+ Dissipation[gt[la,lb]]
+ Dissipation[At[la,lb]]
// ... etc
```

In code:
```c
phirhs += epsdis * kDissFactor * PDdiss[phi];
```

where PDdiss is the discrete dissipation operator.

## Convergence Order

### Expected Convergence
With 4th order space + 4th order time:
```
error ∝ (Δx)^4 + (Δt)^4
```

In practice:
- Constraint violations: ~4th order
- Waveforms: ~4th order in phase, ~3rd in amplitude
- Kreiss-Oliger dissipation reduces to ~3rd order

### Convergence Testing
Run at multiple resolutions:
- Coarse: Δx
- Medium: Δx/2
- Fine: Δx/4

Expected error ratio: 16:4:1 (4th order) or 8:2:1 (3rd order)

## Time Step Calculation

### Formula
```c
dt = dtfac * min_dx
```

where:
- `dtfac` = time step factor (parameter, ~0.25)
- `min_dx` = minimum grid spacing (finest level)

### With AMR
```c
dt_0 = dtfac * dx_0  // coarse level
dt_1 = dt_0 / 2       // first refinement
dt_2 = dt_1 / 2       // second refinement
// etc.
```

### Adaptive time stepping (optional)
Adjust `dtfac` based on:
- Constraint violations
- Lapse collapse (α → 0 near singularity)
- NaN detection

## Summary

**Standard configuration:**
- Method: RK4
- Time order: 4th
- Space order: 4th (or 8th)
- Dissipation: 5th order Kreiss-Oliger
- CFL factor: 0.25 - 0.5
- AMR: 2× time refinement with spatial refinement

**Key files in McLachlan:**
- `schedule.ccl`: Defines when routines run
- `RegisterMoL.cc`: Registers variables with MoL
- `ML_BSSN_EvolutionInterior.cc`: Computes RHS
- Boundary routines: Apply BCs between substeps

**Typical timestep for BBH:**
- Finest grid: Δx ~ 0.03M → Δt ~ 0.01M
- Total evolution: 1000-5000 M
- ~100,000 - 500,000 time steps on finest level
