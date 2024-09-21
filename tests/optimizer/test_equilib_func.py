""" Test for b investor trading functions b(t)
"""
import pytest as pt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../optimizer')))
import trading_funcs as tf


@pt.mark.parametrize("lambd, kappa, trader_a, t, expected", [
    (1, 1, 1, 0, 0),
    (1, 1, 1, 1, 1),
    (1, 1, 0, 0, 0),
    (1, 1, 0, 1, 1),
    (5, 1, 1, 0, 0),  # Lambda = 5 so that a and b trading diff strategies
    (5, 1, 1, 1, 1),
    (5, 1, 0, 0, 0),
    (5, 1, 0, 1, 1),
    (5, 1, 1, 0.5, 0.86963),  # Lambda = 5, and look at mid-points
    (5, 1, 0, 0.5, 0.475959),
    (5, 1, 1, 0.2, 0.424839),
    (5, 1, 0, 0.2, 0.188049),
])
def test_equil_2trader(lambd, kappa, trader_a, t, expected):
    params = {'lambd': lambd, 'kappa': kappa, 'trader_a': trader_a}
    actual = tf.equil_2trader(t, params)
    assert pt.approx(actual, abs=1e-5) == expected


# @pt.mark.skip("Not ready")
@pt.mark.parametrize("lambd, kappa, trader_a, t, expected", [
    (1, 1, 1, 0, 1.17591),
    (1, 1, 1, 1, 0.842575),
    (1, 1, 0, 0, 1.17591),
    (1, 1, 0, 1, 0.842575),
    (5, 1, 1, 0, 2.36377),  # Lambda = 5 so that a and b trading diff strategies
    (5, 1, 1, 1, -0.636227),
    (5, 1, 0, 0, 0.938336),
    (5, 1, 0, 1, 1.13834),
    (5, 1, 1, 0.5, 1.06712),  # Lambda = 5, and look at mid-points
    (5, 1, 0, 0.5, 0.981038),
    (5, 1, 1, 0.2, 1.87856),
    (5, 1, 0, 0.2, 0.944374),
])
def test_equil_2trader_deriv(lambd, kappa, trader_a, t, expected):
    params = {'lambd': lambd, 'kappa': kappa, 'trader_a': trader_a}
    actual = tf.equil_2trader_dot(t, params)
    assert pt.approx(actual, abs=1e-5) == expected
