import argparse

import warp as wp


@wp.kernel
def square_kernel(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
    y[0] = x[0] * x[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--x", type=float, default=3.0)
    args = parser.parse_args()

    wp.init()

    with wp.ScopedDevice(args.device):
        x = wp.array([args.x], dtype=wp.float32, requires_grad=True)
        y = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(square_kernel, dim=1, inputs=[x], outputs=[y])

        tape.backward(loss=y)

        print("y =", float(y.numpy()[0]))
        print("dy/dx =", float(x.grad.numpy()[0]))


if __name__ == "__main__":
    main()

