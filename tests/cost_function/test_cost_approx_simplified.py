""" Test ot check that you can simplify the last term of the cost function """
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../cost_function')))
import cost_function_approx as ca


def test_cost_function_approximations():
    # Define a range of inputs
    lambd_values = [1.0, 10.0, 20.0]
    kappa_values = [0.1, 1.0, 10.0]
    N_values = [5, 10, 15, 50]

    for lambd in lambd_values:
        for kappa in kappa_values:
            for N in N_values:
                # Generate random input arrays
                a_n = np.random.rand(N)
                b_n = np.random.rand(N)

                # Calculate the cost using both functions
                cost_approx = ca.cost_fn_a_approx(a_n, b_n, lambd, kappa)
                cost_approx_simplified = ca.cost_fn_a_approx_simplified(a_n, b_n, lambd, kappa)

                # Assert that the costs are almost equal
                assert np.isclose(cost_approx, cost_approx_simplified, atol=1e-6), \
                    f"Failed for a_n={a_n}, b_n={b_n}, lambd={lambd}, kappa={kappa}, N={N}"
