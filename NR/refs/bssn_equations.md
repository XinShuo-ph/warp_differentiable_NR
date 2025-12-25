# BSSN Evolution Equations (extracted from Einstein Toolkit / McLachlan)

## Variables
- $\phi$: Conformal factor logarithm ($\phi = \ln \psi$) or $W = \psi^{-2}$.
- $\tilde{\gamma}_{ij}$: Conformal metric (unit determinant).
- $K$: Trace of extrinsic curvature.
- $\tilde{A}_{ij}$: Conformal traceless extrinsic curvature.
- $\tilde{\Gamma}^i$: Conformal connection functions.
- $\alpha$: Lapse function.
- $\beta^i$: Shift vector.
- $B^i$: Auxiliary shift variable.

## Evolution Equations

### 1. Conformal Factor ($\phi$)
$$
\partial_t \phi = - \frac{1}{6} \alpha K + \beta^i \partial_i \phi + \frac{1}{6} \partial_i \beta^i
$$
(Note: ET implements advection $\mathcal{L}_\beta \phi$ terms carefully, often upwinded).

### 2. Conformal Metric ($\tilde{\gamma}_{ij}$)
$$
\partial_t \tilde{\gamma}_{ij} = -2 \alpha \tilde{A}_{ij} + \beta^k \partial_k \tilde{\gamma}_{ij} + \tilde{\gamma}_{ik} \partial_j \beta^k + \tilde{\gamma}_{kj} \partial_i \beta^k - \frac{2}{3} \tilde{\gamma}_{ij} \partial_k \beta^k
$$

### 3. Trace of Extrinsic Curvature ($K$)
$$
\partial_t K = -D^2 \alpha + \alpha (\tilde{A}_{ij} \tilde{A}^{ij} + \frac{1}{3} K^2) + \beta^i \partial_i K
$$
where $D^2 \alpha = \gamma^{ij} D_i D_j \alpha$ (Laplacian).

### 4. Conformal Traceless Extrinsic Curvature ($\tilde{A}_{ij}$)
$$
\partial_t \tilde{A}_{ij} = e^{-4\phi} (-D_i D_j \alpha + \alpha R_{ij})^{TF} + \alpha (K \tilde{A}_{ij} - 2 \tilde{A}_{ik} \tilde{A}^k_j) + \mathcal{L}_\beta \tilde{A}_{ij}
$$
(TF denotes trace-free part).

### 5. Conformal Connection Functions ($\tilde{\Gamma}^i$)
$$
\partial_t \tilde{\Gamma}^i = -2 \tilde{A}^{ij} \partial_j \alpha + 2 \alpha (\tilde{\Gamma}^i_{jk} \tilde{A}^{jk} - \frac{2}{3} \tilde{\gamma}^{ij} \partial_j K + 6 \tilde{A}^{ij} \partial_j \phi) + \mathcal{L}_\beta \tilde{\Gamma}^i - \frac{2}{3} \tilde{\Gamma}^i \partial_j \beta^j + \tilde{\gamma}^{jk} \partial_j \partial_k \beta^i + \frac{1}{3} \tilde{\gamma}^{ij} \partial_j \partial_k \beta^k
$$

## Gauge Conditions

### 1+log Slicing (Lapse)
$$
\partial_t \alpha = -2 \alpha K + \beta^i \partial_i \alpha
$$
(Standard form often used: $\partial_t \alpha - \beta^i \partial_i \alpha = -2 \alpha K$. In McLachlan parfile: `harmonicN=1`, `harmonicF=2.0` implies $\partial_t \alpha = -F \alpha^n K$).

### Gamma-driver Shift
$$
\partial_t \beta^i = \frac{3}{4} B^i + \beta^j \partial_j \beta^i
$$
$$
\partial_t B^i = \partial_t \tilde{\Gamma}^i - \eta B^i + \beta^j \partial_j B^i - \beta^j \partial_j \tilde{\Gamma}^i
$$
(Coefficients like 3/4 and $\eta$ are tunable).

## Boundary Conditions
- **NewRad**: Sommerfeld radiation condition applied to all evolved variables at the outer boundary.
- $f = f_0 + u(r-v t) / r$ asymptotic behavior.

## Time Integration
- **Method of Lines (MoL)**
- **Integrator**: RK4 (Runge-Kutta 4th order).
- **Courant Factor**: Typically 0.25 - 0.5.

## Grid Structure
- **Carpet**: Berger-Oliger AMR (Adaptive Mesh Refinement).
- **Structure**: Nested grids (boxes), typically centered on black holes.
- **Refinement**: 2:1 refinement ratio between levels.
