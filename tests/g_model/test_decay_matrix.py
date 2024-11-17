""" Test the decay matrix
    V. Ragulin - 11/15/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

from g_one_trader import decay_matrix

@pt.fixture
def g_exp():
    def resilience_function(t: float) -> float:
        return np.exp(-t)
    return resilience_function

@pt.fixture
def g_zero():
    def resilience_function(t: float) -> float:
        return 1
    return resilience_function

def test_no_decay(g_zero):
    # Define the input arrays
    t_n = np.array([0.2, 0.3, 0.4])

    # Expected result (this should be calculated based on the expected behavior of price)
    expected = np.ones([3,3])

    # Call the function
    result = decay_matrix(t_n, g_zero)

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected, atol=1e-6)

def test_exp_decay_5(g_exp):
    # Define the input arrays
    t_n = np.linspace(0, 1, 5)

    # Expected result (this should be calculated based on the expected behavior of price)
    expected = np.zeros([5,5])
    for i in range(5):
        for j in range(5):
            expected[i,j] = g_exp(np.abs(t_n[i] - t_n[j]))

    # Call the function
    result = decay_matrix(t_n, g_exp)

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected, atol=1e-6)

def test_exp_decay(g_exp):
    # Define the input arrays
    t_n = np.array([0.2, 0.7])

    # Expected result (this should be calculated based on the expected behavior of price)
    expected = np.array([
        [1, np.exp(-0.5)],
        [np.exp(-0.5), 1],
    ])

    # Call the function
    result = decay_matrix(t_n, g_exp)

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected, atol=1e-6)