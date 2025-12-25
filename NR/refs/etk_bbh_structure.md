# Einstein Toolkit BBH Simulation Structure

## Overview
The standard Binary Black Hole (BBH) simulation in the Einstein Toolkit uses the `McLachlan` thorn for evolution (BSSN or CCZ4 formulation) and `Carpet` for Adaptive Mesh Refinement (AMR). Initial data is typically generated using `TwoPunctures`.

## Thorns & Modules
- **Evolution**: `McLachlan` (Auto-generated C++ code for BSSN equations).
- **Grid/AMR**: `Carpet`, `CarpetLib`, `CarpetRegrid2` (Berger-Oliger AMR).
- **Initial Data**: `TwoPunctures` (Spectral solver for constraint equations).
- **Time Integration**: `MoL` (Method of Lines), usually RK4.
- **Boundary Conditions**: `NewRad` (Sommerfeld radiative BCs).
- **Horizon Finding**: `AHFinderDirect`.
- **Wave Extraction**: `WeylScal4`, `Multipole`.

## Grid Structure
- **Cartesian Grid**: Block-structured AMR.
- **Refinement**: Nested boxes centering on the punctures (black holes).
- **Symmetry**: Often uses bitant symmetry (z>0) or no symmetry for precessing binaries.
- **Outer Boundary**: Placed far away (e.g., > 100M) to minimize reflections.

## Output Files
The `CarpetIOBasic`, `CarpetIOASCII`, and `CarpetIOHDF5` thorns handle output.

### Standard Outputs
- **Scalar (0D)** (`*.asc`, `*.ts`):
  - `norm1`, `norm2`, `max` of grid functions.
  - `H.asc`: Hamiltonian constraint violation (L2 norm).
  - `alp.max.asc`: Maximum lapse.
- **1D/2D/3D** (`*.asc`, `*.h5`):
  - 1D lines along axes.
  - 2D planes (xy, xz).
  - 3D full grid dumps (checkpoints).
- **Gravitational Waves**:
  - `mp_psi4.asc`: Decomposed Weyl scalar $\Psi_4$ into spherical harmonics ($l, m$ modes).

## Time Integration
- **Method**: Method of Lines (MoL).
- **Integrator**: `RK4` (GenericRungeKutta), 4th order.
- **CFL Condition**: Regulated by `Time` thorn, typically $\Delta t \approx 0.4 \Delta x$.

## Boundary Conditions
- **Outer**: Radiative (`NewRad`).
- **Inner**: None (Punctures are handled by the specific BSSN variable choice/regularization, effectively "moving punctures" where the singularity is avoided on the grid).
