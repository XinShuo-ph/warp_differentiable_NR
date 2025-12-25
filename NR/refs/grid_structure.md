# Grid Structure & Boundary Conditions

## Grid Structure (Carpet / Box-in-Box AMR)

The Einstein Toolkit typically uses the `Carpet` driver for adaptive mesh refinement (AMR).

### Structure
- **Hierarchy:** Nested rectangular grids (levels).
- **Refinement Factor:** Typically 2:1 between levels.
- **Berger-Oliger:** Recursive time sub-cycling (finer grids take 2 time steps for every 1 coarse step).
- **Components:**
  - `Maps`: Coordinate patches (for multi-patch systems like Llama).
  - `Levels`: Refinement levels $l=0 \dots l_{max}$.
  - `Components`: Disjoint grids at the same level.

### Typical BBH Grid
- **Coarse Grid:** Covers the wave extraction zone (distant observer).
- **Fine Grids:** Track the black holes ("Puncture" locations).
- **Symmetry:** Often reflection symmetry across $z=0$ or $\pi$-symmetry is used to reduce cost.

## Boundary Conditions

### Outer Boundary
- **Sommerfeld / Radiative:** Outgoing wave boundary conditions.
  - $f(r, t) \sim \frac{u(r-t)}{r}$
  - $\partial_t f + \frac{r^i}{r} \partial_i f + \frac{f}{r} = 0$
- Applied to all evolution variables.

### Refinement Boundaries (AMR)
- **Prolongation (Coarse to Fine):**
  - Spatial: 5th order polynomial interpolation.
  - Temporal: 2nd order time interpolation.
- **Restriction (Fine to Coarse):**
  - Average or inject fine grid values to coarse grid.

### Punctures (Internal)
- Moving punctures approach handles singularities without excision.
- Gauge choices ($\alpha$ collapse) handle the singularity naturally.
