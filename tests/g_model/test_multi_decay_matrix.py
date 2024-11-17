""" Test the multi-trader decay matrix
    V. Ragulin - 11/16/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

import g_one_trader as go
import g_multi_trader as gm

@pt.fixture
def g_exp():
    def resilience_function(t: float) -> float:
        return np.exp(-t + 0.4)
    return resilience_function

@pt.fixture
def g_zero():
    def resilience_function(t: float) -> float:
        return 1
    return resilience_function

def test_no_decay(g_zero):
    t_n = np.array([0.2, 0.3, 0.4])
    N = len(t_n)

    res = gm.decay_matrix_lt(t_n, g_zero)
    for i in range(N):
        for j in range(N):
            if i > j:
                assert res[i,j] == g_zero(t_n[i] - t_n[j])
            elif i == j:
                assert res[i,j] == 0.5 * g_zero(0)
            else:
                assert res[i,j] == 0

def test_exp_decay_5(g_exp):
    # Define the input arrays
    t_n = np.linspace(0, 1, 5)
    N = len(t_n)

    res = gm.decay_matrix_lt(t_n, g_exp)
    for i in range(N):
        for j in range(N):
            if i > j:
                assert res[i,j] == g_exp(t_n[i] - t_n[j])
            elif i == j:
                assert res[i,j] == 0.5 * g_exp(0)
            else:
                assert res[i,j] == 0


def test_exp_decay(g_exp):
    # Define the input arrays
    t_n = np.array([0.2, 0.7])

    # Expected result (this should be calculated based on the expected behavior of price)
    expected = np.array([
        [0.5*np.exp(0.4), 0],
        [np.exp(-0.1), 0.5*np.exp(0.4)],
    ])

    # Call the function
    result = gm.decay_matrix_lt(t_n, g_exp)

    # Assert that the result is gas expected
    np.testing.assert_allclose(result, expected, atol=1e-6)
