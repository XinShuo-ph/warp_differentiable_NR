import warp as wp
import math
from derivatives import D_1_4th, D_2_4th

wp.init()

@wp.kernel
def test_deriv_kernel(
    f: wp.array(dtype=float, ndim=3),
    df_dx: wp.array(dtype=float, ndim=3),
    d2f_dx2: wp.array(dtype=float, ndim=3),
    dx: float
):
    i, j, k = wp.tid()
    df_dx[i, j, k] = D_1_4th(f, i, j, k, 0, dx)
    d2f_dx2[i, j, k] = D_2_4th(f, i, j, k, 0, dx)

def test_derivatives():
    res = 64
    L = 1.0
    dx = L / res
    
    shape = (res, res, res)
    f = wp.zeros(shape, dtype=float)
    df_dx = wp.zeros(shape, dtype=float)
    d2f_dx2 = wp.zeros(shape, dtype=float)
    
    # Initialize f = sin(2*pi*x)
    data_np = f.numpy()
    for i in range(res):
        x = i * dx
        val = math.sin(2.0 * math.pi * x)
        data_np[i, :, :] = val
    f = wp.array(data_np, dtype=float)
    
    wp.launch(
        kernel=test_deriv_kernel,
        dim=shape,
        inputs=[f, df_dx, d2f_dx2, dx]
    )
    
    # Verify
    # f'(x) = 2*pi * cos(2*pi*x)
    # f''(x) = -(2*pi)^2 * sin(2*pi*x)
    
    # Check center point to avoid boundary
    i = res // 2
    x = i * dx
    
    exact_df = 2.0 * math.pi * math.cos(2.0 * math.pi * x)
    exact_d2f = - (2.0 * math.pi)**2 * math.sin(2.0 * math.pi * x)
    
    calc_df = df_dx.numpy()[i, 0, 0]
    calc_d2f = d2f_dx2.numpy()[i, 0, 0]
    
    print(f"x = {x}")
    print(f"df/dx: exact={exact_df:.5f}, calc={calc_df:.5f}, err={abs(exact_df-calc_df):.5e}")
    print(f"d2f/dx2: exact={exact_d2f:.5f}, calc={calc_d2f:.5f}, err={abs(exact_d2f-calc_d2f):.5e}")
    
    # 4th order should be very accurate
    assert abs(exact_df - calc_df) < 1e-4
    assert abs(exact_d2f - calc_d2f) < 1e-3
    print("Derivative test passed.")

if __name__ == "__main__":
    test_derivatives()
