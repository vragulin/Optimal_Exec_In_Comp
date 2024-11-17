""" Test the cost function
    V. Ragulin - 11/15/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

from g_one_trader import price, cost_trader, decay_matrix, cost_trader_matrix

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

@pt.mark.parametrize("g_idx, expected", [
    (0, 0.5),
    (1, 0.349896556219511),
])
def test_imp_cost(g_idx, expected, g_zero, g_exp):
    # Test the cost function
    def g(t):
        if g_idx == 0:
            return g_zero(t)
        else:
            return g_exp(t)
    N = 5
    t = np.linspace(0, 1, N)
    x = np.ones(N) * 1/N
    actual = cost_trader(t, x, g)
    assert pt.approx(actual, abs=1e-6) == expected

@pt.mark.parametrize("g_idx, expected", [
    (0, 0.5),
    (1, 0.349896556219511),
])
def test_imp_cost_mat(g_idx, expected, g_zero, g_exp):
    # Test the cost function
    def g(t):
        if g_idx == 0:
            return g_zero(t)
        else:
            return g_exp(t)

    N = 5
    t = np.linspace(0, 1, N)
    x = np.ones(N) * 1 / N
    g_mat = decay_matrix(t, g)
    actual = cost_trader_matrix(x, g_mat)
    assert pt.approx(actual, abs=1e-6) == expected