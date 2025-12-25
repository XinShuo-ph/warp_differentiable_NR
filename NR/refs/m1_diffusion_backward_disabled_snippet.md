Source: `NR/warp/warp/examples/fem/example_diffusion.py`

```python
if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ...

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            degree=args.degree,
            resolution=args.resolution,
            mesh=args.mesh,
            serendipity=args.serendipity,
            viscosity=args.viscosity,
            boundary_value=args.boundary_value,
            boundary_compliance=args.boundary_compliance,
        )
        example.step()
        example.render()
```

