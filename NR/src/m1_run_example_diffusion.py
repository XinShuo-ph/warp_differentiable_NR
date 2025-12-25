import numpy as np

import warp as wp
from warp.examples.fem.example_diffusion import Example


def main() -> None:
    wp.set_module_options({"enable_backward": False})

    with wp.ScopedDevice("cpu"):
        ex = Example(quiet=True, resolution=10)
        ex.step()

        # stable numeric fingerprint for "2 consistent runs" validation
        u = ex._scalar_field.dof_values.numpy()
        checksum = float(np.sum(u))
        l2 = float(np.linalg.norm(u))
        print(f"checksum={checksum:.17e} l2={l2:.17e}")


if __name__ == "__main__":
    main()

