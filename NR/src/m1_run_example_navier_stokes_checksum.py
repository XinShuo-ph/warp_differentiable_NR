import numpy as np

import warp as wp


def main() -> None:
    from warp.examples.fem.example_navier_stokes import Example

    wp.init()

    example = Example(quiet=True, degree=2, resolution=12, Re=200.0, top_velocity=1.0, mesh="grid")
    example.step()

    u = example._u_field.dof_values.numpy()
    p = example._p_field.dof_values.numpy()

    device = wp.get_device()
    print(f"device={device.alias} is_cuda={device.is_cuda}")
    print(f"u_dof_count={u.shape[0]} p_dof_count={p.shape[0]}")
    print(f"checksum_sum_u={float(np.sum(u)):.9f}")
    print(f"checksum_sum_p={float(np.sum(p)):.9f}")


if __name__ == "__main__":
    main()

