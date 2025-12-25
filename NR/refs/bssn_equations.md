# BSSN Evolution Equations

The BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation decomposes the metric $g_{ij}$ into conformal quantities to ensure numerical stability.

## Variables

- $\phi$: Conformal factor, related to metric determinant. $g_{ij} = e^{4\phi} \tilde{\gamma}_{ij}$. (Note: Some codes use $\chi = e^{-4\phi}$ or $W = e^{-2\phi}$).
- $\tilde{\gamma}_{ij}$: Conformal 3-metric. $det(\tilde{\gamma}_{ij}) = 1$.
- $K$: Trace of the extrinsic curvature $K_{ij}$.
- $\tilde{A}_{ij}$: Conformal traceless extrinsic curvature. $K_{ij} = e^{4\phi} \tilde{A}_{ij} + \frac{1}{3} g_{ij} K$.
- $\tilde{\Gamma}^i$: Conformal connection functions (evolved independently). $\tilde{\Gamma}^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}$.

## Evolution Equations

Assume vacuum ($S_{\mu\nu} = 0$). $\mathcal{L}_\beta$ is the Lie derivative along shift $\beta^i$.

### 1. Conformal Factor ($\phi$)
$$
\partial_t \phi - \beta^k \partial_k \phi = -\frac{1}{6} (\alpha K - \partial_k \beta^k)
$$

### 2. Conformal Metric ($\tilde{\gamma}_{ij}$)
$$
\partial_t \tilde{\gamma}_{ij} - \beta^k \partial_k \tilde{\gamma}_{ij} = -2 \alpha \tilde{A}_{ij} + \tilde{\gamma}_{ik} \partial_j \beta^k + \tilde{\gamma}_{jk} \partial_i \beta^k - \frac{2}{3} \tilde{\gamma}_{ij} \partial_k \beta^k
$$

### 3. Trace of Extrinsic Curvature ($K$)
$$
\partial_t K - \beta^k \partial_k K = -D^i D_i \alpha + \alpha (\tilde{A}_{ij} \tilde{A}^{ij} + \frac{1}{3} K^2)
$$
where $D^i D_i \alpha = \frac{1}{e^{4\phi}} (\tilde{D}^2 \alpha + 2 \tilde{\gamma}^{ij} \partial_i \phi \partial_j \alpha)$ is the Laplacian in the physical metric.

### 4. Conformal Traceless Extrinsic Curvature ($\tilde{A}_{ij}$)
$$
\partial_t \tilde{A}_{ij} - \beta^k \partial_k \tilde{A}_{ij} = e^{-4\phi} \left[ -D_i D_j \alpha + \alpha R_{ij} \right]^{TF} + \alpha (K \tilde{A}_{ij} - 2 \tilde{A}_{ik} \tilde{A}^k_j) + \tilde{A}_{ik} \partial_j \beta^k + \tilde{A}_{jk} \partial_i \beta^k - \frac{2}{3} \tilde{A}_{ij} \partial_k \beta^k
$$
where $[\cdot]^{TF}$ denotes the trace-free part: $T_{ij}^{TF} = T_{ij} - \frac{1}{3} \tilde{\gamma}_{ij} \tilde{\gamma}^{kl} T_{kl}$.

### 5. Conformal Connection Functions ($\tilde{\Gamma}^i$)
$$
\partial_t \tilde{\Gamma}^i - \beta^k \partial_k \tilde{\Gamma}^i = -2 \tilde{A}^{ij} \partial_j \alpha + 2 \alpha (\tilde{\Gamma}^i_{jk} \tilde{A}^{jk} - \frac{2}{3} \tilde{\gamma}^{ij} \partial_j K + 6 \tilde{A}^{ij} \partial_j \phi) + \tilde{\gamma}^{jk} \partial_j \partial_k \beta^i + \frac{1}{3} \tilde{\gamma}^{ik} \partial_k \partial_j \beta^j + \beta^j \partial_j \tilde{\Gamma}^i - \tilde{\Gamma}^j \partial_j \beta^i + \frac{2}{3} \tilde{\Gamma}^i \partial_j \beta^j
$$

## Gauge Conditions (Moving Puncture)

### 1. 1+log Slicing (Lapse $\alpha$)
$$
\partial_t \alpha - \beta^i \partial_i \alpha = -2 \alpha K
$$

### 2. Gamma-driver Shift (Shift $\beta^i$)
Introduces auxiliary variable $B^i$.
$$
\partial_t \beta^i - \beta^j \partial_j \beta^i = \frac{3}{4} B^i
$$
$$
\partial_t B^i - \beta^j \partial_j B^i = \partial_t \tilde{\Gamma}^i - \beta^j \partial_j \tilde{\Gamma}^i - \eta B^i
$$
Usually simplified to:
$$
\partial_t \beta^i = \frac{3}{4} B^i
$$
$$
\partial_t B^i = \partial_t \tilde{\Gamma}^i - \eta B^i
$$

## Grid Structure

- **Finite Difference**: Standard BSSN implementations (like McLachlan) use high-order finite differencing (typically 4th, 6th, or 8th order) on a structured Cartesian grid.
- **Variables location**: Vertex-centered (nodes are at grid points).
- **Boundaries**: Radiative boundary conditions (Sommerfeld) at the outer boundary.
- **Kreiss-Oliger Dissipation**: Added to evolution equations to damp high-frequency noise.

## Time Integration

- **Method**: Method of Lines (MoL)
- **Integrator**: RK4 (Runge-Kutta 4th order) is standard.
