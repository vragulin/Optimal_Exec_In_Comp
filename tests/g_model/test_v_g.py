""" Test the vectorized g function
    V. Ragulin - 11/15/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))
from g_one_trader import v_g

DEBUG = True

# Define the resilience function g
@pt.fixture
def g():
    def resilience_function(t: float) -> float:
        return np.exp(-t)
    return resilience_function

def test_v_g(g):
    # Define the input arrays
    t = np.array([0.1, 0.2, 0.3])

    # Expected result (this should be calculated based on the expected behavior of v_g)
    expected = np.zeros(len(t))
    for i in range(len(t)): # Replace with actual expected values
        expected[i] = g(t[i])

    # Call the vectorized function
    result = v_g(t, g)

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected, atol=1e-6)
