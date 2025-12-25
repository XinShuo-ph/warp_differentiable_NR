# BSSN Evolution Equations (McLachlan / Standard)

## Variables
- $\phi$: Conformal factor exponent ($\gamma_{ij} = e^{4\phi} \tilde{\gamma}_{ij}$).
- $\tilde{\gamma}_{ij}$: Conformal metric ($\det \tilde{\gamma} = 1$).
- $K$: Trace of extrinsic curvature.
- $\tilde{A}_{ij}$: Conformal traceless extrinsic curvature.
- $\tilde{\Gamma}^i$: Conformal connection functions.
- $\alpha$: Lapse function.
- $\beta^i$: Shift vector.

## Evolution Equations

### 1. Conformal Factor ($\phi$)
$$
\partial_t \phi - \beta^i \partial_i \phi = -\frac{1}{6} \alpha K + \frac{1}{6} \partial_i \beta^i
$$

### 2. Conformal Metric ($\tilde{\gamma}_{ij}$)
$$
\partial_t \tilde{\gamma}_{ij} - \beta^k \partial_k \tilde{\gamma}_{ij} = -2 \alpha \tilde{A}_{ij} + \tilde{\gamma}_{ik} \partial_j \beta^k + \tilde{\gamma}_{jk} \partial_i \beta^k - \frac{2}{3} \tilde{\gamma}_{ij} \partial_k \beta^k
$$

### 3. Trace of Extrinsic Curvature ($K$)
$$
\partial_t K - \beta^i \partial_i K = -D^2 \alpha + \alpha (\tilde{A}_{ij} \tilde{A}^{ij} + \frac{1}{3} K^2)
$$
where $D^2 \alpha = \gamma^{ij} D_i D_j \alpha$ (covariant derivative wrt physical metric).

### 4. Conformal Traceless Extrinsic Curvature ($\tilde{A}_{ij}$)
$$
\partial_t \tilde{A}_{ij} - \beta^k \partial_k \tilde{A}_{ij} = e^{-4\phi} [ -D_i D_j \alpha + \alpha (R_{ij} - 8\pi S_{ij}) ]^{TF} + \alpha (K \tilde{A}_{ij} - 2 \tilde{A}_{ik} \tilde{A}^k_j) + \tilde{A}_{ik} \partial_j \beta^k + \tilde{A}_{jk} \partial_i \beta^k - \frac{2}{3} \tilde{A}_{ij} \partial_k \beta^k
$$
where TF denotes the trace-free part.

### 5. Conformal Connection Functions ($\tilde{\Gamma}^i$)
$$
\partial_t \tilde{\Gamma}^i - \beta^j \partial_j \tilde{\Gamma}^i = -2 \tilde{A}^{ij} \partial_j \alpha + 2 \alpha (\tilde{\Gamma}^i_{jk} \tilde{A}^{jk} - \frac{2}{3} \tilde{\gamma}^{ij} \partial_j K + 6 \tilde{A}^{ij} \partial_j \phi) + \tilde{\gamma}^{jk} \partial_j \partial_k \beta^i + \frac{1}{3} \tilde{\gamma}^{ij} \partial_j \partial_k \beta^k + \beta^j \partial_j \tilde{\Gamma}^i - \tilde{\Gamma}^j \partial_j \beta^i + \frac{2}{3} \tilde{\Gamma}^i \partial_j \beta^j
$$
*(Note: The advection term $\beta^j \partial_j \tilde{\Gamma}^i$ is often treated carefully or substituted)*

## Gauge Conditions (Moving Punctures)

### 1+log Slicing
$$
\partial_t \alpha - \beta^i \partial_i \alpha = -2 \alpha K
$$

### Gamma-driver Shift
$$
\partial_t \beta^i - \beta^j \partial_j \beta^i = \frac{3}{4} B^i
$$
$$
\partial_t B^i - \beta^j \partial_j B^i = \partial_t \tilde{\Gamma}^i - \beta^j \partial_j \tilde{\Gamma}^i - \eta B^i
$$
Usually implemented as:
$$
\partial_t \beta^i = \beta^j \partial_j \beta^i + \frac{3}{4} B^i
$$
$$
\partial_t B^i = \beta^j \partial_j B^i + \partial_t \tilde{\Gamma}^i - \beta^j \partial_j \tilde{\Gamma}^i - \eta B^i
$$
Or simpler "Hyperbolic Gamma-driver":
$$
\partial_t \beta^i = \frac{3}{4} B^i
$$
$$
\partial_t B^i = \partial_t \tilde{\Gamma}^i - \eta B^i
$$
(Advection terms are often omitted in simplified versions or included). Standard Moving Puncture uses the advective form.

## Constraints
- **Hamiltonian**: $R + K^2 - K_{ij} K^{ij} = 16\pi \rho$
- **Momentum**: $D_j (K^{ij} - \gamma^{ij} K) = 8\pi S^i$
