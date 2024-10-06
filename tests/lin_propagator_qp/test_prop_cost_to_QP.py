""" Test the resresentation of the cost function as a quadratic program
    V. Ragulin - 10/05/2024
"""
import os
import sys
import pytest as pt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'matrix_utils')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'lin_propagator_qp')))

from propagator import cost_fn_prop_a_matrix
from prop_cost_to_QP import cost_to_QP_numerical, cost_QP_params


DEBUG = True


def test_cost_to_QP_numerical():
    a_n = np.array([1, 2, 3, 4])
    b_n = np.array([2, 3, 4, 5])
    lambd = 1
    rho = 10

    H, f, C = cost_to_QP_numerical(a_n, b_n, lambd, rho)
    print(H)
    print(f)
    print(C)

    # Test that the cost function is correctly represented as a quadratic program
    x = a_n
    cost_from_func = cost_fn_prop_a_matrix(x, b_n, lambd, rho)
    cost_from_QP = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_QP), "Cost function is not correctly represented as a QP"


def test_QP_params():
    a_n = np.array([1, 2, 3, 4])
    b_n = np.array([2, 3, 4, 5])
    lambd = 1
    rho = 10

    # Caclulate the matrices numerically
    H0, f0, C0 = cost_to_QP_numerical(a_n, b_n, lambd, rho)
    if DEBUG:
        print("Numerical parameters")
        print(H0)
        print(f0)
        # print(C0)

    # Calculate the matrices analytically
    H, f = cost_QP_params(b_n, lambd, rho)
    if DEBUG:
        print("Analytical parameters")
        print(H)
        print(f)
        # print(C)

    # Test that the cost function is correctly represented as a quadratic program
    for p0, p1 in zip([H0, f0], [H, f]):
        assert np.allclose(p0, p1), "Parameters are not equal"


def test_QP_params_rand():
    np.random.seed(0)
    N = 20
    a_n = np.random.randn(N)
    b_n = np.random.randn(N)
    lambd = 1
    rho = 10

    # Caclulate the matrices numerically
    H0, f0, C0 = cost_to_QP_numerical(a_n, b_n, lambd, rho)
    if DEBUG:
        print("Numerical parameters")
        print(H0)
        print(f0)
        # print(C0)

    # Calculate the matrices analytically
    H, f = cost_QP_params(b_n, lambd, rho)
    if DEBUG:
        print("Analytical parameters")
        print(H)
        print(f)
        # print(C)

    # Test that the cost function is correctly represented as a quadratic program
    for p0, p1 in zip([H0, f0], [H, f]):
        assert np.allclose(p0, p1), "Parameters are not equal"