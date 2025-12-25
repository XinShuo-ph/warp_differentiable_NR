Source: `NR/warp/warp/examples/fem/example_adaptive_grid.py`

```python
# Adaptive grid construction from a scalar refinement field
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(sim_vol)), func=refinement_field, values={"volume": collider.id}
)
self._geo = fem.adaptive_nanogrid_from_field(sim_vol, level_count, refinement_field=refinement, grading="face")
```

```python
# Handling resolution-boundary discontinuities via Sides + jump/average
p_side_test = fem.make_test(p_space, domain=fem.Sides(self._geo))
u_side_trial = fem.make_trial(u_space, domain=fem.Sides(self._geo))
divergence_matrix += fem.integrate(
    side_divergence_form,
    fields={"u": u_side_trial, "psi": p_side_test},
    output_dtype=float,
    assembly="generic",
)
```

