# Time Integration Scheme (Einstein Toolkit / McLachlan)

## Method of Lines (MoL) Framework

### Overview
The Einstein Toolkit uses the Method of Lines (MoL) approach:
1. **Spatial Discretization**: Finite differences for derivatives
2. **Time Integration**: ODE solver for resulting semi-discrete system

### MoL Time Integrator

#### Common Schemes

**1. RK3 (Runge-Kutta 3rd order)**
```
k₁ = f(tⁿ, uⁿ)
k₂ = f(tⁿ + Δt/2, uⁿ + Δt/2 k₁)
k₃ = f(tⁿ + Δt, uⁿ - Δt k₁ + 2Δt k₂)
uⁿ⁺¹ = uⁿ + Δt/6 (k₁ + 4k₂ + k₃)
```
- 3 intermediate steps
- 3rd order accurate in time
- Memory efficient

**2. RK4 (Runge-Kutta 4th order)** - Most common for BBH
```
k₁ = f(tⁿ, uⁿ)
k₂ = f(tⁿ + Δt/2, uⁿ + Δt/2 k₁)
k₃ = f(tⁿ + Δt/2, uⁿ + Δt/2 k₂)
k₄ = f(tⁿ + Δt, uⁿ + Δt k₃)
uⁿ⁺¹ = uⁿ + Δt/6 (k₁ + 2k₂ + 2k₃ + k₄)
```
- 4 intermediate steps
- 4th order accurate in time
- Standard choice for production runs

**3. ICN (Iterative Crank-Nicholson)**
- 2nd order implicit method
- More stable but requires iteration
- Less commonly used

### MoL Configuration Parameters
```
MoL::ODE_Method             = "RK4"
MoL::MoL_Intermediate_Steps = 4
MoL::MoL_Num_Scratch_Levels = 1
```

## Timestep Selection

### CFL Condition
For stability, the Courant-Friedrichs-Lewy condition must be satisfied:
```
Δt ≤ CFL × min(Δx, Δy, Δz)
```

Typical values:
- CFL = 0.25-0.5 for BSSN with RK4
- CFL = 0.125 for 8th order finite differences
- More restrictive for higher-order methods

### Example Calculation
```
Grid spacing: Δx = 2.0
CFL factor: 0.25
⟹ Δt = 0.5
```

## Adaptive Mesh Refinement (AMR) and Time

### Subcycling
In AMR simulations:
- Each refinement level has its own timestep
- Typical ratio: Δt_fine = Δt_coarse / refinement_factor
- Usually refinement_factor = 2

Example hierarchy:
```
Level 0 (coarsest):  Δt₀ = 0.5
Level 1:             Δt₁ = 0.25
Level 2:             Δt₂ = 0.125
Level 3 (finest):    Δt₃ = 0.0625
```

### Time Interpolation
- Coarse levels provide boundary data for fine levels
- Time interpolation needed at fine level boundaries
- Usually 2nd order polynomial interpolation in time

## Complete Time Evolution Algorithm

### Single Level Evolution (Simplified)
```
For each timestep n → n+1:
  1. Compute RHS at current time (RK stage 1)
     - Evaluate spatial derivatives
     - Apply BSSN evolution equations
     - Apply dissipation
  
  2. Intermediate RK stages (2, 3, 4)
     - Update variables to intermediate time
     - Apply boundary conditions
     - Re-evaluate RHS
  
  3. Final update
     - Combine RK stages with weights
     - Update all evolved variables
  
  4. Post-processing
     - Apply constraint damping (if CCZ4)
     - Enforce gauge conditions
     - Apply boundary conditions
```

### Multi-Level AMR Evolution
```
Function Evolve(level, time, dt):
  if level < max_level:
    # Evolve fine level with smaller timestep
    for substep in [0, refinement_factor-1]:
      Evolve(level+1, time + substep*dt/ref, dt/ref)
      Sync_from_fine_to_coarse()
  
  # Evolve this level
  for RK_stage in [1, 2, 3, 4]:
    Compute_RHS(level)
    if level > 0:
      Interpolate_boundary_from_coarse()
    Apply_boundary_conditions()
    Update_variables(RK_weights[stage])
  
  Sync_from_coarse_to_fine()
```

## Typical Evolution Schedule

### Full BBH Simulation
```
t = 0:           Initial data (TwoPunctures)
t = 0 to t_junk: "Junk radiation" phase (~50-100M)
                 System settles from initial data
t_junk to t_m:   Inspiral phase
t_m to t_m+20M:  Merger phase
t_m+20M onward:  Ringdown phase
```
where M is the total mass of the system.

### Typical Runtime Parameters
```
Initial time:    cctk_initial_time = 0
Final time:      cctk_final_time = 5000
Timestep:        dtfac = 0.25
Iterations:      ~20,000 iterations for full inspiral-merger-ringdown
```

## Constraint Preservation

### Constraint Violation Growth
In standard BSSN:
- Constraints not enforced during evolution
- Violation grows approximately linearly with time
- Monitored via Hamiltonian and momentum constraints

### Constraint Damping (CCZ4)
CCZ4 formulation adds damping terms:
```
∂_t Θ = ... - dampk1 * (2 + dampk2) * α * Θ
```
Parameters:
- dampk1 ~ 0.02-0.1
- dampk2 ~ 0-1
- Helps maintain constraint satisfaction

## Numerical Stability

### Sources of Instability
1. **CFL violation**: Too large timestep
2. **Gauge pathologies**: Lapse collapse, shift divergence
3. **Boundary reflections**: Spurious incoming radiation
4. **AMR artifacts**: Poor interpolation at level boundaries
5. **High-frequency noise**: Requires dissipation

### Stability Measures
1. **Dissipation**: Kreiss-Oliger added to all variables
2. **Gauge conditions**: Carefully tuned parameters
3. **Minimum lapse**: Prevent lapse from going to zero
4. **Constraint monitoring**: Stop simulation if constraints blow up

## Performance Considerations

### Computational Cost per Timestep
- Most expensive: RHS evaluation (spatial derivatives)
- 4 RHS evaluations per RK4 step
- Constraint evaluation: typically every 8-32 timesteps
- Wave extraction: typically every 32-128 timesteps

### Scaling
- Strong scaling: efficiency drops with more CPUs (communication overhead)
- Weak scaling: better, but AMR introduces load imbalance
- GPU acceleration: significant speedup for RHS evaluation

## References
- MoL thorn: Method of Lines infrastructure
- Carpet: AMR and time integration orchestration
- Alcubierre, "Introduction to 3+1 Numerical Relativity" (2008), Chapter 3
- Baumgarte & Shapiro, "Numerical Relativity" (2010), Chapter 5
