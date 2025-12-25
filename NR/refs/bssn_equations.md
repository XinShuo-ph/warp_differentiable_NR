# BSSN Evolution Equations (from McLachlan)

## BSSN Variables

The BSSN formulation uses the following evolved variables:

1. **phi** (or W): Conformal factor
   - phi: conformal factor where gamma_ij = e^(4phi) gamma_tilde_ij
   - W: alternative conformal factor where W = e^(-2phi)

2. **gamma_tilde_ij** (gt): Conformal 3-metric
   - Traceless part of the physical metric
   - det(gamma_tilde) = 1

3. **A_tilde_ij** (At): Traceless conformal extrinsic curvature
   - Defined as: A_tilde_ij = e^(-4phi) (K_ij - 1/3 gamma_ij K)

4. **K** (trK): Trace of extrinsic curvature

5. **Gamma_tilde^i** (Xt): Conformal connection functions
   - Gamma_tilde^i = gamma_tilde^jk Gamma_tilde^i_jk

6. **alpha**: Lapse function

7. **beta^i**: Shift vector

## BSSN Evolution Equations (from McLachlan)

### Conformal Factor (PRD 62 044034 (2000), eqn. (10))
```
dot[phi] = (1/3 phi or -1/6) * (alpha * trK - D_i beta^i)
           + beta^i D_i phi
           + Dissipation[phi]
```

### Conformal Metric (PRD 62 044034 (2000), eqn. (9))
```
dot[gt_ij] = -2 alpha At_ij
             + gt_ik D_j beta^k
             + gt_jk D_i beta^k
             - 2/3 gt_ij D_k beta^k
             + beta^k D_k gt_ij
             + Dissipation[gt_ij]
```

### Conformal Connection (PRD 62 044034 (2000), eqn. (20))
```
dot[Xt^i] = -2 At^ij D_j alpha
            + 2 alpha (Gamma^i_jk At^jk - 2/3 g^ij D_j trK + 6 At^ij D_j phi)
            + g^jk D_jk beta^i
            + 1/3 g^ij D_jk beta^k
            - Xt^j D_j beta^i
            + 2/3 Xt^i D_j beta^j
            + beta^j D_j Xt^i
            + Dissipation[Xt^i]
```

### Trace of Extrinsic Curvature (PRD 62 044034 (2000), eqn. (11))
```
dot[trK] = -e^(-4phi) (g^ij D_ij alpha + 2 D_i phi D^i alpha - Xt^i D_i alpha)
           + alpha (At^ij At_ij + 1/3 trK^2)
           + beta^i D_i trK
           + Dissipation[trK]
```

### Traceless Extrinsic Curvature (PRD 62 044034 (2000), eqn. (12))
```
Ats_ij = -D_ij alpha + 2 (D_i alpha D_j phi + D_j alpha D_i phi) + alpha R_ij

dot[At_ij] = e^(-4phi) (Ats_ij - 1/3 g_ij g^kl Ats_kl)
             + alpha (trK At_ij - 2 At_ik A^k_j)
             + At_ik D_j beta^k
             + At_jk D_i beta^k
             - 2/3 At_ij D_k beta^k
             + beta^k D_k At_ij
             + Dissipation[At_ij]
```

where R_ij is the Ricci tensor of the conformal metric.

### Lapse Function (Harmonic gauge)
```
dot[alpha] = -harmonicF * alpha^harmonicN * (trK + alphaDriver * (alpha - 1))
             + beta^i D_i alpha  (if advectLapse)
             + Dissipation[alpha]
```

### Shift Vector (Gamma driver)
```
dot[beta^i] = shiftGammaCoeff * alpha^shiftAlphaPower * (Xt^i - betaDriver * beta^i)
              + beta^j D_j beta^i  (if advectShift)
              + Dissipation[beta^i]
```

## Auxiliary Quantities

### Ricci Tensor Components
The Ricci tensor R_ij of the conformal metric is computed from:
- Christoffel symbols Gamma^i_jk
- Their derivatives
- The conformal metric gt_ij and its derivatives

### Finite Differencing
- 4th order accurate centered differences for spatial derivatives
- Upwind differencing for advection terms (beta^i D_i X)

### Kreiss-Oliger Dissipation
Added to each evolved variable to suppress high-frequency noise:
```
Dissipation[X] = epsDiss * (-1)^(order/2) * h^order * D^order X
```

## Constraints

### Hamiltonian Constraint
```
H = R - A^ij A_ij + 2/3 K^2 = 0
```

### Momentum Constraint
```
M_i = D_j (A^j_i - 2/3 gamma^j_i K) = 0
```

### Gauge Constraint
```
Gamma_tilde^i - Xt^i = 0
```

## Reference
- Pretorius, Phys. Rev. D 62, 044034 (2000)
- Brown et al., Phys. Rev. D 67, 084023 (2003)
- Alic et al., arXiv:1106.2254 (2011)
- Baumgarte & Shapiro, Phys. Rept. 376 (2003) 41-131
