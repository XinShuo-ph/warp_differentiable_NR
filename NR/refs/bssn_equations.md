# BSSN Formulation for Numerical Relativity

## Overview
BSSN (Baumgarte-Shapiro-Shibata-Nakamura) is a 3+1 decomposition of Einstein's equations
suitable for numerical evolution. References:
- Shibata & Nakamura, Phys. Rev. D 52, 5428 (1995)
- Baumgarte & Shapiro, Phys. Rev. D 59, 024007 (1999)

## Variables

### Conformal Factor
```
phi = (1/12) * ln(det(gamma_ij))      # conformal factor
chi = exp(-4*phi)                      # alternative: chi formulation
W = exp(-2*phi)                        # alternative: W formulation
```

### Conformal Metric
```
gamma_tilde_ij = exp(-4*phi) * gamma_ij    # conformal 3-metric
det(gamma_tilde_ij) = 1                     # unit determinant constraint
```

### Extrinsic Curvature
```
K = gamma^ij * K_ij                    # trace of extrinsic curvature
A_tilde_ij = exp(-4*phi) * (K_ij - (1/3)*gamma_ij*K)   # traceless conformal
```

### Connection Functions
```
Gamma_tilde^i = gamma_tilde^jk * Gamma_tilde^i_jk   # contracted Christoffel
              = -partial_j(gamma_tilde^ij)          # equivalent form
```

### Gauge Variables
```
alpha = lapse function
beta^i = shift vector
B^i = time derivative of shift (Gamma-driver gauge)
```

## Evolution Equations

### Conformal Factor (phi formulation)
```
partial_t(phi) = beta^i * partial_i(phi) - (1/6) * alpha * K
               + (1/6) * partial_i(beta^i)
```

### Conformal Metric
```
partial_t(gamma_tilde_ij) = beta^k * partial_k(gamma_tilde_ij)
                          + gamma_tilde_ik * partial_j(beta^k)
                          + gamma_tilde_jk * partial_i(beta^k)
                          - (2/3) * gamma_tilde_ij * partial_k(beta^k)
                          - 2 * alpha * A_tilde_ij
```

### Trace of Extrinsic Curvature
```
partial_t(K) = beta^i * partial_i(K)
             - D^i D_i(alpha)
             + alpha * (A_tilde_ij * A_tilde^ij + (1/3) * K^2)
             + 4*pi*alpha * (rho + S)
```

### Traceless Extrinsic Curvature
```
partial_t(A_tilde_ij) = beta^k * partial_k(A_tilde_ij)
                      + A_tilde_ik * partial_j(beta^k)
                      + A_tilde_jk * partial_i(beta^k)
                      - (2/3) * A_tilde_ij * partial_k(beta^k)
                      + exp(-4*phi) * [-D_i D_j(alpha) + alpha*R_ij]^TF
                      + alpha * (K * A_tilde_ij - 2 * A_tilde_ik * A_tilde^k_j)
                      - 8*pi*alpha * exp(-4*phi) * S_ij^TF
```

### Connection Functions (Gamma_tilde^i)
```
partial_t(Gamma_tilde^i) = beta^j * partial_j(Gamma_tilde^i)
                         - Gamma_tilde^j * partial_j(beta^i)
                         + (2/3) * Gamma_tilde^i * partial_j(beta^j)
                         + gamma_tilde^jk * partial_j partial_k(beta^i)
                         + (1/3) * gamma_tilde^ij * partial_j partial_k(beta^k)
                         - 2 * A_tilde^ij * partial_j(alpha)
                         + 2 * alpha * (Gamma_tilde^i_jk * A_tilde^jk
                                       - (2/3) * gamma_tilde^ij * partial_j(K)
                                       + 6 * A_tilde^ij * partial_j(phi)
                                       - 8*pi * gamma_tilde^ij * S_j)
```

## Gauge Conditions

### 1+log Slicing
```
partial_t(alpha) = beta^i * partial_i(alpha) - 2 * alpha * K
```

### Gamma-Driver Shift
```
partial_t(beta^i) = (3/4) * B^i + beta^j * partial_j(beta^i)
partial_t(B^i) = partial_t(Gamma_tilde^i) - eta * B^i + beta^j * partial_j(B^i)
```

## Constraints

### Hamiltonian Constraint
```
H = R + K^2 - K_ij * K^ij - 16*pi*rho = 0
```

### Momentum Constraint
```
M^i = D_j(K^ij - gamma^ij * K) - 8*pi*S^i = 0
```

### Algebraic Constraints
```
det(gamma_tilde_ij) = 1           # unit determinant
gamma_tilde^ij * A_tilde_ij = 0   # traceless A_tilde
Gamma_tilde^i = gamma_tilde^jk * Gamma_tilde^i_jk   # connection definition
```

## Numerical Implementation Notes

### Kreiss-Oliger Dissipation
Add 6th-order dissipation: -sigma * h^5 * D_+^3 D_-^3 (u)
Typical sigma = 0.1 to 0.3

### Finite Differencing
4th-order centered derivatives recommended:
D_0(u)_i = (-u_{i+2} + 8*u_{i+1} - 8*u_{i-1} + u_{i-2}) / (12*h)

### Time Integration
RK4 (4th order Runge-Kutta) is standard
