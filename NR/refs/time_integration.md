# Time Integration for BSSN

## Overview

BSSN evolution is typically performed using explicit Runge-Kutta methods due to:
- Stiff source terms requiring small timesteps anyway
- Simplicity compared to implicit methods
- Good stability properties with appropriate CFL condition

## Method of Lines (MoL)

The BSSN equations are written as:
```
∂ₜu = RHS(u, ∂ᵢu, ∂ᵢⱼu)
```

where u = (χ, γ̃ᵢⱼ, K, Ãᵢⱼ, Γ̃ⁱ, α, βⁱ) and RHS contains all spatial derivatives.

## RK4 (4th Order Runge-Kutta)

Most common choice for high accuracy:

```python
# RK4 integration
k1 = dt * RHS(u^n)
k2 = dt * RHS(u^n + k1/2)
k3 = dt * RHS(u^n + k2/2)
k4 = dt * RHS(u^n + k3)

u^(n+1) = u^n + (k1 + 2*k2 + 2*k3 + k4)/6
```

Properties:
- 4 RHS evaluations per timestep
- 4th order accurate in time
- Stable for CFL ≤ ~0.45 (Cartesian grid)

## RK3 (3rd Order Runge-Kutta)

Alternative with fewer evaluations:

```python
# Strong Stability Preserving RK3
u₁ = u^n + dt * RHS(u^n)
u₂ = 3/4 * u^n + 1/4 * u₁ + 1/4 * dt * RHS(u₁)
u^(n+1) = 1/3 * u^n + 2/3 * u₂ + 2/3 * dt * RHS(u₂)
```

Properties:
- 3 RHS evaluations per timestep
- 3rd order accurate
- Better stability for some formulations
- CFL ≤ ~0.5

## ICN (Iterative Crank-Nicholson)

Used in some codes for improved stability:

```python
# Predictor
u* = u^n + dt * RHS(u^n)

# Corrector
u^(n+1) = u^n + dt/2 * (RHS(u^n) + RHS(u*))
```

Can iterate corrector step for better accuracy.

## CFL Condition

Timestep constrained by:
```
dt ≤ CFL * min(dx, dy, dz) / c
```

where:
- c = speed of light (1 in geometric units)
- CFL ≈ 0.25-0.45 depending on method and gauge
- Adaptive: dt can change during evolution

## Kreiss-Oliger Dissipation

Added to RHS for stability:
```
RHS_dissipated = RHS + ε * D^(2n) u
```

where:
- D^(2n): centered 2nth order dissipation operator
- n = 1 (4th order dissipation) or n = 2 (6th order)
- ε ≈ 0.01-0.1 (dissipation strength)

4th order dissipation:
```python
def dissipation_4th(u, dx, epsilon):
    # At point i,j,k
    diss = epsilon * (
        -u[i-2] + 4*u[i-1] - 6*u[i] + 4*u[i+1] - u[i+2]
    ) / (16 * dx**3)
    return diss
```

## Implementation Pseudocode

```python
class BSSNEvolver:
    def __init__(self, grid, dt, cfl=0.4):
        self.grid = grid
        self.dt = dt
        self.cfl = cfl
        self.dissipation_epsilon = 0.1
        
    def compute_rhs(self, state):
        """Compute RHS of BSSN equations"""
        # 1. Update ghost zones
        self.apply_boundary_conditions(state)
        
        # 2. Compute spatial derivatives (4th order)
        derivs = self.compute_derivatives(state)
        
        # 3. Compute RHS terms
        rhs = self.bssn_rhs(state, derivs)
        
        # 4. Add dissipation
        rhs += self.kreiss_oliger_dissipation(state)
        
        return rhs
    
    def step_rk4(self, state):
        """Single RK4 timestep"""
        dt = self.dt
        
        k1 = dt * self.compute_rhs(state)
        k2 = dt * self.compute_rhs(state + 0.5*k1)
        k3 = dt * self.compute_rhs(state + 0.5*k2)
        k4 = dt * self.compute_rhs(state + k3)
        
        state_new = state + (k1 + 2*k2 + 2*k3 + k4)/6
        
        return state_new
    
    def evolve(self, state, t_final):
        """Evolve to final time"""
        t = 0
        while t < t_final:
            # Adaptive timestep
            self.dt = self.cfl * self.grid.min_spacing()
            
            state = self.step_rk4(state)
            t += self.dt
            
            # Constraint monitoring
            if self.iteration % 10 == 0:
                self.check_constraints(state)
        
        return state
```

## Timestep Adaptation

Some codes use adaptive timesteps:
```python
# Compute max wave speed
v_max = max(|α|, |βⁱ|) across grid

# Update timestep
dt = CFL * min_spacing / v_max

# Restrict growth
dt = min(dt, 1.1 * dt_previous)
```

## Checkpointing

For long runs:
- Save full state every N timesteps
- Enables restart from checkpoint
- Binary format for exact reproducibility
- Typical: checkpoint every 1000-10000 steps
