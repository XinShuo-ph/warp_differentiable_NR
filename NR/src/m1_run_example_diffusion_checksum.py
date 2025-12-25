import numpy as np

import warp as wp


def main() -> None:
    from warp.examples.fem.example_diffusion import Example

    wp.init()
    example = Example(quiet=True, resolution=30, degree=2, mesh="grid")
    example.step()

    dofs = example._scalar_field.dof_values.numpy()
    checksum = float(np.sum(dofs))

    device = wp.get_device()
    print(f"device={device.alias} is_cuda={device.is_cuda}")
    print(f"dof_count={dofs.size}")
    print(f"checksum_sum_dofs={checksum:.9f}")


if __name__ == "__main__":
    main()

