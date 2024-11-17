""" Test the cost function for a multi-trader market
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
        return np.exp(-t)
    return resilience_function

@pt.fixture
def g_zero():
    def resilience_function(t: float) -> float:
        return 1
    return resilience_function

@pt.mark.parametrize("g_idx", [0, 1])
def test_zero_market1(g_idx, g_zero, g_exp):
    # Test the cost function for a zero market
    def g(t):
        if g_idx == 0:
            return g_zero(t)
        else:
            return g_exp(t)
    N = 5
    t = np.linspace(0, 1, N)
    x = np.ones(N) * 1/N
    m = np.zeros(N)
    actual = gm.cost_trader(t, x, m, g, trd_in_mkt=False)
    expected = go.cost_trader(t, x, g)
    assert pt.approx(actual, abs=1e-6) == expected

@pt.mark.parametrize("g_idx", [0, 1])
def test_zero_market2(g_idx, g_zero, g_exp):
    # Test the cost function for a zero market
    def g(t):
        if g_idx == 0:
            return g_zero(t)
        else:
            return g_exp(t)

    N = 5
    t = np.linspace(0, 1, N)
    x = np.ones(N) * 1 / N
    actual = gm.cost_trader(t, x, x, g)
    expected = go.cost_trader(t, x, g)
    assert pt.approx(actual, abs=1e-6) == expected

@pt.mark.parametrize("g_idx", [0, 1])
@pt.mark.parametrize("mkt_mult", [0.5, 1, 2])
def test_mkt_mult_of_trader1(g_idx, mkt_mult, g_zero, g_exp):
    # Test the cost function when the mkt is proportional to trader
    def g(t):
        if g_idx == 0:
            return g_zero(t)
        else:
            return g_exp(t)

    N = 5
    t = np.linspace(0, 1, N)
    x = np.ones(N) * 1 / N
    mkt = x * mkt_mult
    actual = gm.cost_trader(t, x, mkt, g)
    expected = go.cost_trader(t, x, g) * mkt_mult
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("g_idx", [0, 1])
@pt.mark.parametrize("mkt_mult", [0.5, 1, 2])
def test_mkt_mult_of_trader2(g_idx, mkt_mult, g_zero, g_exp):
    # Test the cost function when the mkt is proportional to trader
    def g(t):
        if g_idx == 0:
            return g_zero(t)
        else:
            return g_exp(t)

    N = 7
    t = np.linspace(0, 1, N)
    x = np.ones(N) * 1 / N
    mkt = x * mkt_mult
    actual = gm.cost_trader(t, x, mkt, g, trd_in_mkt=False)
    expected = go.cost_trader(t, x, g) * (mkt_mult+1)
    assert pt.approx(actual, abs=1e-6) == expected

@pt.mark.parametrize("g_idx, expected", [
    (0, 1.125),
    (1, 0.825729907)])
def test_manual_inp1(g_idx, expected, g_zero, g_exp):
    """ Test using manual input from the spreadsheet
    """
    def g(t):
        if g_idx == 0:
            return g_zero(t)
        else:
            return g_exp(t)

    t = np.array([0, 0.2, 0.7])
    x = np.array([0.2, 0, 0.3])
    m = np.array([1, 1, 2])

    res = gm.cost_trader(t, x, m, g, trd_in_mkt=False)
    assert pt.approx(res, abs=1e-6) == expected