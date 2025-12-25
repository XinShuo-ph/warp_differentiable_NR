# Time Integration Scheme

The Einstein Toolkit standardly uses the **Method of Lines (MoL)**.

## Scheme
- **Explicit Runge-Kutta 4 (RK4)** is the workhorse.
- **Courant Factor (CFL):** Typically $\Delta t / \Delta x \le 0.5$ (often 0.25 or 0.4 for stability).

### Algorithm (RK4)
Given state $U^n$ at $t_n$:

1. $k_1 = \mathcal{L}(U^n)$
2. $k_2 = \mathcal{L}(U^n + \frac{\Delta t}{2} k_1)$
3. $k_3 = \mathcal{L}(U^n + \frac{\Delta t}{2} k_2)$
4. $k_4 = \mathcal{L}(U^n + \Delta t k_3)$
5. $U^{n+1} = U^n + \frac{\Delta t}{6} (k_1 + 2k_2 + 2k_3 + k_4)$

where $\mathcal{L}(U)$ is the right-hand side of the evolution equations (BSSN + Gauge).

## Dissipation
- **Kreiss-Oliger Dissipation:** Added to the RHS to dampen high-frequency noise from finite differencing.
- Usually 5th or 7th order dissipation added to variables.
- Form: $\partial_t U = \dots + \frac{\sigma}{2^k} (\Delta x)^{2k-1} (D_+)^k (D_-)^k U$
  - For RK4 (4th order), typically order 6 or higher dissipation is needed? Actually usually matches order of error.
  - ET often uses `dissipation_order = 5` or `7`.
