import numpy as np

import warp as wp


def main() -> None:
    from warp.examples.fem.example_adaptive_grid import Example

    wp.init()
    example = Example(
        quiet=True,
        degree=1,
        div_conforming=False,
        base_resolution=6,
        level_count=2,
        headless=True,
    )
    example.step()

    u = example.velocity_field.dof_values.numpy()
    p = example.pressure_field.dof_values.numpy()

    device = wp.get_device()
    print(f"device={device.alias} is_cuda={device.is_cuda}")
    print(f"u_dof_count={u.shape[0]} p_dof_count={p.shape[0]}")
    print(f"checksum_sum_u={float(np.sum(u)):.9f}")
    print(f"checksum_sum_p={float(np.sum(p)):.9f}")


if __name__ == "__main__":
    main()

