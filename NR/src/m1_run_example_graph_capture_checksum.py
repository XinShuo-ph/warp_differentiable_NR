import numpy as np

import warp as wp


def main() -> None:
    # Import from the installed Warp package (not the repo checkout).
    from warp.examples.core.example_graph_capture import Example

    wp.init()
    example = Example()
    example.step()

    pixels = example.pixel_values.numpy()
    checksum = float(np.sum(pixels))

    # Stable, comparable output across runs
    device = wp.get_device()
    print(f"device={device.alias} is_cuda={device.is_cuda}")
    print(f"checksum_sum_pixels={checksum:.9f}")


if __name__ == "__main__":
    main()

