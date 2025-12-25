import warp as wp
import numpy as np

wp.init()

@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def test_add():
    n = 10
    a = wp.array(np.ones(n, dtype=np.float32), dtype=float)
    b = wp.array(np.ones(n, dtype=np.float32) * 2, dtype=float)
    c = wp.zeros(n, dtype=float)

    wp.launch(add_kernel, dim=n, inputs=[a, b, c])

    result = c.numpy()
    expected = np.ones(n) * 3
    assert np.allclose(result, expected), f"Got {result}, expected {expected}"
    print("Test passed! Result:", result)

if __name__ == "__main__":
    test_add()
