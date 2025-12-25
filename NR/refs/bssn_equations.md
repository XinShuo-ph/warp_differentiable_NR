# BSSN/Z4c Evolution Equations

Source: SpacetimeX/Z4cowGPU (modern implementation)
Reference: arXiv:1212.2901 [gr-qc]

## Evolution Variables

### Primary Variables (evolved)
- `W`: conformal factor (W = exp(phi), where phi = ln(sqrt(det(gamma))))
- `gamt_ij`: conformal 3-metric (det(gamt) = 1)
- `exKh`: trace of extrinsic curvature
- `exAt_ij`: tracefree part of conformal extrinsic curvature
- `trGt^i`: conformal connection functions
- `Theta`: Z4c constraint damping variable
- `alpha`: lapse function
- `beta^i`: shift vector

### Derived Quantities
- Physical metric: `gam_ij = W^(-2) gamt_ij`
- Inverse conformal metric: `invgamt^ij` (with det(gamt) = 1 enforced)

## Evolution Equations

### (1) Conformal Factor
```
dt W = beta^k d_k W + (1/3) W (alpha (exKh + 2*Theta) - d_i beta^i)
```

### (2) Conformal Metric
```
dt gamt_ij = beta^k d_k gamt_ij - 2 alpha exAt_ij 
             + gamt_ki d_j beta^k + gamt_kj d_i beta^k 
             - (2/3) gamt_ij d_k beta^k
```

### (3) Trace of Extrinsic Curvature
```
dt exKh = beta^k d_k exKh - invgam^kl DD_kl alpha 
          + alpha (exAt_kl exAtUU^kl + (1/3)(exKh + 2*Theta)^2) 
          + 4*pi*alpha*(trSs + rho) 
          + alpha kappa1 (1 - kappa2) Theta
```

### (4) Tracefree Extrinsic Curvature
```
dt exAt_ij = beta^k d_k exAt_ij 
             + W^2 ((-DD_ij alpha + alpha (R_ij - 8*pi*Ss_ij)) 
                    - (1/3) gam_ij invgam^kl (-DD_kl alpha + alpha (R_kl - 8*pi*Ss_kl)))
             + alpha ((exKh + 2*Theta) exAt_ij - 2 invgamt^kl exAt_ki exAt_lj)
             + exAt_ki d_j beta^k + exAt_kj d_i beta^k 
             - (2/3) exAt_ij d_k beta^k
```

### (5) Conformal Connection Functions
```
dt trGt^i = beta^k d_k trGt^i - 2 exAtUU^ij d_j alpha 
            + 2 alpha (Gt^i_jk exAtUU^jk - 3 exAtUU^ij d_j ln W 
                       - (1/3) invgamt^ij (2 d_j exKh + d_j Theta) 
                       - 8*pi invgamt^ij Sm_j)
            + invgamt^jk dd_jk beta^i + (1/3) invgamt^ij dd_jk beta^k 
            - trGtd_j d^j beta^i + (2/3) trGtd^i d_j beta^j 
            - 2 alpha kappa1 (trGt^i - trGtd^i)
```

### (6) Z4c Constraint Damping
```
dt Theta = beta^k d_k Theta 
           + (1/2) alpha (trR - exAt_kl exAtUU^kl + (2/3)(exKh + 2*Theta)^2) 
           - alpha (8*pi*rho + kappa1 (2 + kappa2) Theta)
```

## Gauge Conditions

### (11) 1+log Slicing (Lapse)
```
dt alpha = beta^k d_k alpha - alpha muL exKh
```

### (12) Gamma-driver Shift
```
dt beta^i = beta^k d_k beta^i + muS trGt^i - eta beta^i
```

## Ricci Tensor Components

### (8) Conformal factor contribution
```
RtW_ij = (1/W) tDtD_ij W + (1/W) gamt_ij invgamt^kl tDtD_kl W 
         - 2 gamt_ij invgamt^kl d_k ln W d_l ln W
```
where `tDtD_ij W = dd_ij W - Gt^k_ij d_k W` (conformal covariant derivative)

### (9) Metric contribution
```
Rt_ij = -(1/2) invgamt^lm dd_lm gamt_ij 
        + (1/2) (gamt_ki d_j trGt^k + gamt_kj d_i trGt^k)
        + (1/2) trGtd^k (GtDDD_ijk + GtDDD_jik)
        + (Gt^k_li GtDDU_j^kl + Gt^k_lj GtDDU_i^kl + Gt^k_im GtDDU_k^jm)
```

### (10) Total Ricci
```
R_ij = RtW_ij + Rt_ij
trR = invgam^kl R_kl
```

## Constraints

### (13) Gauge constraint
```
ZtC^i = (trGt^i - trGtd^i) / 2
```

### (14) Hamiltonian constraint
```
HC = trR - exAt_kl exAtUU^kl + (2/3)(exKh + 2*Theta)^2 - 16*pi*rho
```

### (15) Momentum constraint
```
MtC^i = trd exAtUU^i + Gt^i_jk exAtUU^jk - (2/3) invgamt^ij (d_j exKh + 2 d_j Theta)
        - 3 exAtUU^ij d_j ln W - 8*pi invgamt^ij Sm_j
```

## Matter Terms

```
rho = alpha^(-2) (eTtt - 2 beta^j eTt_j + beta^i beta^j eT_ij)
Sm_i = -alpha^(-1) (eTt_i - beta^k eT_ki)
Ss_ij = eT_ij
trSs = invgam^kl Ss_kl
```

## Parameters

- `kappa1`, `kappa2`: Z4c constraint damping parameters
- `muL`: lapse evolution parameter (1+log slicing)
- `muS`: shift driver parameter
- `eta`: shift damping parameter
