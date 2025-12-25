# Grid Structure and Boundary Conditions in Einstein Toolkit

## Grid Infrastructure (Carpet/Cactus)

### Coordinate System
- Cartesian coordinates (x, y, z)
- Domain typically symmetric (using reflection symmetries)
- Rotating symmetries (e.g., 180° rotation) for binary systems

### Domain Size
Example from qc0-mclachlan.par:
```
xmin =    0.00, xmax = +120.00
ymin = -120.00, ymax = +120.00
zmin =    0.00, zmax = +120.00
dx = dy = dz = 2.00
```

Symmetries reduce computational domain:
- Reflection in z=0 plane (bitant symmetry)
- Reflection in x=0 plane (octant symmetry for single BH)
- 180° rotational symmetry for equal-mass binaries

### Adaptive Mesh Refinement (AMR)

**Carpet driver parameters:**
```
max_refinement_levels = 10
ghost_size = 3  # stencil width
use_buffer_zones = yes
```

**Refinement hierarchy:**
- Level 0: Coarsest, covers entire domain
- Level 1, 2, ...: Progressively finer grids
- Refinement factor: typically 2× per level
- Child grids centered on compact objects

**Prolongation and Restriction:**
```
prolongation_order_space = 5  # 5th order spatial interpolation
prolongation_order_time = 2   # 2nd order temporal interpolation
```

**Regridding:**
- Static grids: fixed hierarchy (common for BBH)
- Dynamic grids: adapt based on error estimates

### Grid Hierarchy Example (BBH)
```
Level 0: 240M × 240M × 240M  (dx = 2M)
Level 1: 120M × 120M × 120M  (dx = 1M)
Level 2: 60M × 60M × 60M     (dx = 0.5M)
Level 3: 30M × 30M × 30M     (dx = 0.25M)
...
Level 7: ~2M × 2M × 2M       (dx = 0.03125M)
```
where M is the total mass of the system.

## Boundary Conditions

### Physical Boundaries

#### Outer Boundary (far field)
McLachlan typically uses "flat" or radiative BCs:

**Flat BC:**
```
f(boundary) = f_background
```
Assumes spacetime becomes flat at large distances.

**Radiative BC (Sommerfeld):**
```
(∂_t + v_r ∂_r)(f - f_0) = 0
```
where:
- v_r = radial wave speed (~1 for light)
- f_0 = expected value at infinity

**NewRad BC (improved radiative):**
More sophisticated treatment accounting for:
- Fall-off rates (1/r, 1/r², etc.)
- Multiple characteristic speeds

#### Inner Boundary (black hole interior)
Two main approaches:

**Excision:**
- Remove interior of apparent horizon
- Causal structure ensures no information needed from inside
- Requires horizon tracking

**Puncture (moving puncture):**
- Regularize at singularity
- "1+log" slicing drives α → 0 at singularity
- Gamma-driver shift moves puncture with BH
- No excision needed

### Symmetry Boundaries

**Reflection Symmetry (z=0):**
```
ReflectionSymmetry::reflection_z = yes
ReflectionSymmetry::avoid_origin_z = no
```

Variables transform as:
- Even (scalar, T^xx, etc.): f(x,y,-z) = +f(x,y,z)
- Odd (vector z-component): f_z(x,y,-z) = -f_z(x,y,z)

**Rotational Symmetry (180°):**
```
RotatingSymmetry180::poison_boundaries = yes
```
For equal-mass binaries: (x,y,z) → (-x,-y,z)

### Refinement Boundaries

**Buffer zones:**
```
use_buffer_zones = yes
```
Extra cells at refinement boundaries for:
- Smooth interpolation
- Stability at level transitions

**Boundary width:**
```
boundary_size_x_lower = 3
boundary_size_y_lower = 3
boundary_size_z_lower = 3
boundary_size_x_upper = 3
boundary_size_y_upper = 3
boundary_size_z_upper = 3
```

**Boundary shiftout:**
```
boundary_shiftout_x_lower = 1
boundary_shiftout_z_lower = 1
```
Shift boundary away from symmetry axes to avoid numerical issues.

## Constraint Preservation at Boundaries

### Standard BCs
Simple BCs don't guarantee constraint preservation.
Constraints can grow exponentially at boundaries.

### Constraint-preserving BCs
- Maximize constraints (BCMax)
- Harmonic extraction
- Sommerfeld on constraint variables

McLachlan_BSSN_Helper provides:
```c
SelectBCsADMBase.c  # Select BC type
NewRad.c            # Radiative BC implementation
ExtrapolateGammas.c # Special treatment for Γ̃^i
```

## Finite Differencing Details

### Stencil
4th order centered:
```
∂_x f[i] ≈ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12 dx)
```

8th order (optional):
```
More points: i±1, i±2, i±3, i±4
```

### Ghost Zones
```
ghost_size = 3  # for 4th order
ghost_size = 4  # for 8th order
```

Filled by:
1. Prolongation from coarser levels
2. Boundary conditions at domain edges
3. Symmetry operations at symmetry boundaries

### One-sided Derivatives
At boundaries, switch to one-sided stencils:
- Maintain order of accuracy
- Use more points on interior side

## Grid Indexing (Cactus)

### 3D index
```c
int idx = CCTK_GFINDEX3D(cctkGH, i, j, k);
```

### Stride
```c
ptrdiff_t di = 1;
ptrdiff_t dj = CCTK_GFINDEX3D(cctkGH,0,1,0) - CCTK_GFINDEX3D(cctkGH,0,0,0);
ptrdiff_t dk = CCTK_GFINDEX3D(cctkGH,0,0,1) - CCTK_GFINDEX3D(cctkGH,0,0,0);
```

### Loop over interior
```c
for (int k = imin[2]; k < imax[2]; k++)
  for (int j = imin[1]; j < imax[1]; j++)
    for (int i = imin[0]; i < imax[0]; i++)
```
Excludes ghost zones automatically.

## Summary

**Key features:**
- AMR with Carpet (2× refinement ratio)
- Reflection and rotational symmetries
- Multiple boundary types: flat, radiative, excision
- 4th or 8th order finite differences
- 3 ghost zones minimum
- Prolongation/restriction between levels
- Constraint monitoring (not enforcement)

**Typical setup for BBH:**
- 7-9 refinement levels
- Finest grid: 2-4 points across BH diameter
- Outer boundary: 100-500M from origin
- Time step: CFL limited, finest level determines dt
- Output: gravitational waves extracted at multiple radii
