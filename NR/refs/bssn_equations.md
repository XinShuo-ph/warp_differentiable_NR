# BSSN Evolution Equations

Based on the standard BSSN formulation used in the Einstein Toolkit (McLachlan).

## Variables
- Conformal factor: $\phi = \frac{1}{12} \ln(\det(\gamma_{ij}))$ or $\chi = e^{-4\phi}$ or $W = e^{-2\phi}$
  - McLachlan often uses $\phi$ or $W$. We will use $\phi$ for now.
- Conformal metric: $\tilde{\gamma}_{ij} = e^{-4\phi} \gamma_{ij}$
  - Constraint: $\det(\tilde{\gamma}_{ij}) = 1$
- Trace of extrinsic curvature: $K = \gamma^{ij} K_{ij}$
- Conformal traceless extrinsic curvature: $\tilde{A}_{ij} = e^{-4\phi} (K_{ij} - \frac{1}{3} \gamma_{ij} K)$
  - Constraint: $\tilde{\gamma}^{ij} \tilde{A}_{ij} = 0$
- Conformal connection functions: $\tilde{\Gamma}^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}$

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
where $D^2 \alpha = \gamma^{ij} D_i D_j \alpha$.

### 4. Conformal Traceless Extrinsic Curvature ($\tilde{A}_{ij}$)
$$
\partial_t \tilde{A}_{ij} - \beta^k \partial_k \tilde{A}_{ij} = e^{-4\phi} (-D_i D_j \alpha + \alpha R_{ij})^{TF} + \alpha (K \tilde{A}_{ij} - 2 \tilde{A}_{ik} \tilde{A}^k_j) + \tilde{A}_{ik} \partial_j \beta^k + \tilde{A}_{jk} \partial_i \beta^k - \frac{2}{3} \tilde{A}_{ij} \partial_k \beta^k
$$
where TF denotes the trace-free part.

### 5. Conformal Connection Functions ($\tilde{\Gamma}^i$)
$$
\partial_t \tilde{\Gamma}^i - \beta^k \partial_k \tilde{\Gamma}^i = -2 \tilde{A}^{ij} \partial_j \alpha + 2 \alpha (\tilde{\Gamma}^i_{jk} \tilde{A}^{jk} - \frac{2}{3} \tilde{\gamma}^{ij} \partial_j K + 6 \tilde{A}^{ij} \partial_j \phi) + \tilde{\gamma}^{jk} \partial_j \partial_k \beta^i + \frac{1}{3} \tilde{\gamma}^{ij} \partial_j \partial_k \beta^k + \beta^k \partial_k \tilde{\Gamma}^i - \tilde{\Gamma}^k \partial_k \beta^i + \frac{2}{3} \tilde{\Gamma}^i \partial_k \beta^k
$$

## Gauge Conditions (1+log, Gamma-driver)

### Lapse ($\alpha$) - 1+log
$$
\partial_t \alpha - \beta^i \partial_i \alpha = -2 \alpha K
$$

### Shift ($\beta^i$) - Gamma-driver
$$
\partial_t \beta^i - \beta^k \partial_k \beta^i = \frac{3}{4} B^i
$$
$$
\partial_t B^i - \beta^k \partial_k B^i = \partial_t \tilde{\Gamma}^i - \beta^k \partial_k \tilde{\Gamma}^i - \eta B^i
$$
(This is the "integrated" version often used).

## Constraints
- Hamiltonian: $R + K^2 - K_{ij} K^{ij} = 0$ (Vacuum)
- Momentum: $D_j (K^{ij} - \gamma^{ij} K) = 0$
