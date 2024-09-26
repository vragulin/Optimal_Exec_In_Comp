""" Test ot check that you can simplify the last term of the cost function """
import pytest as pt
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../cost_function')))
import cost_function_approx as ca


# def test_cost_fn_a_approx_v_old():
#     # Define a range of inputs
#     lambd_values = [1.0, 10.0, 20.0]
#     kappa_values = [0.1, 1.0, 10.0]
#     N_values = [5, 10, 15, 50]
#
#     for lambd in lambd_values:
#         for kappa in kappa_values:
#             for N in N_values:
#                 # Generate random input arrays
#                 a_n = np.random.rand(N)
#                 b_n = np.random.rand(N)
#
#                 # Calculate the cost using both functions
#                 new_val = ca.cost_fn_a_approx(a_n, b_n, lambd, kappa)
#                 old_val = ca.cost_fn_a_approx_old(a_n, b_n, lambd, kappa)
#
#                 # Assert that the costs are almost equal
#                 assert np.isclose(new_val, old_val, atol=1e-6), \
#                     f"Failed for a_n={a_n}, b_n={b_n}, lambd={lambd}, kappa={kappa}, N={N}"


@pt.mark.parametrize("lambd", [1.0, 10.0, 20.0])
@pt.mark.parametrize("kappa", [0.1, 1.0, 10.0])
@pt.mark.parametrize("N", [5, 10, 15, 50])
def test_cost_fn_a_approx_v_old(lambd, kappa, N):
    # Generate random input arrays
    a_n = np.random.rand(N)
    b_n = np.random.rand(N)

    # Calculate the cost using both functions
    new_val = ca.cost_fn_a_approx(a_n, b_n, lambd, kappa)
    old_val = ca.cost_fn_a_approx_old(a_n, b_n, lambd, kappa)

    # Assert that the costs are almost equal
    assert np.isclose(new_val, old_val, atol=1e-6), \
        f"Failed for a_n={a_n}, b_n={b_n}, lambd={lambd}, kappa={kappa}, N={N}"


@pt.mark.parametrize("lambd", [1.0, 10.0, 20.0])
@pt.mark.parametrize("kappa", [0.1, 1.0, 10.0])
@pt.mark.parametrize("N", [5, 10, 15, 50])
def test_cost_fn_b_approx_v_old(lambd, kappa, N):
    # Generate random input arrays
    a_n = np.random.rand(N)
    b_n = np.random.rand(N)

    # Calculate the cost using both functions
    new_val = ca.cost_fn_b_approx(a_n, b_n, lambd, kappa)
    old_val = ca.cost_fn_b_approx_old(a_n, b_n, lambd, kappa)

    # Assert that the costs are almost equal
    assert np.isclose(new_val, old_val, atol=1e-6), \
        f"Failed for a_n={a_n}, b_n={b_n}, lambd={lambd}, kappa={kappa}, N={N}"
