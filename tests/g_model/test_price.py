""" Test the price function
    V. Ragulin - 11/15/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

from g_optimize import price

DEBUG = True

def test_no_decay():
    # Define the input arrays
    t = np.array([0.1, 0.25, 0.35, 0.45])
    t_n = np.array([0.2, 0.3, 0.4])
    x_n = np.array([1.0, 2.0, 3.0])

    # Expected result (this should be calculated based on the expected behavior of price)
    expected = np.array([
        0, # t=0.1
        x_n[0], # t=0.25
        x_n[0] + x_n[1], # t=0.35
        np.sum(x_n), # t=0.45
    ])

    # Call the function
    result = np.array(
        [price(tau, t_n, x_n, lambda x: 1) for tau in t]
    )

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected, atol=1e-6)

def test_half_decay():
    # Define the input arrays
    t = np.array([0.1, 0.25, 0.35, 0.45])
    t_n = np.array([0.2, 0.3, 0.4])
    x_n = np.array([1.0, 2.0, 3.0])

    # Expected result (this should be calculated based on the expected behavior of price)
    expected = np.array([
        0, # t=0.1
        x_n[0]/2, # t=0.25
        (x_n[0] + x_n[1])/2, # t=0.35
        (np.sum(x_n))/2, # t=0.45
    ])

    def g(t):
        if t==0:
            return 1
        else:
            return 0.5

    # Call the function
    result = np.array(
        [price(tau, t_n, x_n, g) for tau in t]
    )

    # Assert that the result is as expected
    np.testing.assert_allclose(result, expected, atol=1e-6)
