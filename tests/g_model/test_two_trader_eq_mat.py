""" Test two trader equilibrium
    V. Ragulin - 11/15/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

import g_multi_trader as gm
import g_one_trader as go

DEBUG = True


def test_tte_N2():
    N = 2

    def g(x):
        return 0.5 + 0.5 * np.exp(-10 * x)

    # Define the input arrays
    t_n = np.linspace(0, 1, N)
    D = gm.decay_matrix_lt(t_n, g)
    sizes = [1, 10]
    a_opt, b_opt = gm.two_trader_equilibrium_mat(sizes, D)

    if DEBUG:
        print(f'a_opt: {a_opt}')
        print(f'b_opt: {b_opt}')

    expected_a = np.array([0.5, 0.5])
    expected_b = np.array([5, 5])
    np.testing.assert_allclose(a_opt, expected_a, atol=1e-6)
    np.testing.assert_allclose(b_opt, expected_b, atol=1e-6)


def test_tte_N3():
    N = 3
    rho = 1

    def g(x):
        return 0.5 + 0.5 * np.exp(-rho * x)

    # Define the input arrays
    t_n = np.array([0, 0.4, 1])
    D = gm.decay_matrix_lt(t_n, g)
    sizes = [1, 10]
    a_opt, b_opt = gm.two_trader_equilibrium_mat(sizes, D)

    if DEBUG:
        print(f'a_opt: {a_opt}')
        print(f'b_opt: {b_opt}')

    expected_a = np.array([0.402157932,
                           0.164133772,
                           0.433708296])
    expected_b = np.array([4.021579318,
                           1.641337724,
                           4.337082958])
    np.testing.assert_allclose(a_opt, expected_a, atol=1e-6)
    np.testing.assert_allclose(b_opt, expected_b, atol=1e-6)
