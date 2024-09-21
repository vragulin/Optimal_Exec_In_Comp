""" Test for b investor trading functions b(t)
"""
import pytest as pt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../optimizer')))
import trading_funcs as tf


@pt.mark.parametrize("sig, t, expected", [
    (3, 0, 0),
    (3, 1, 1),
    (3, 0.5, 0.817574)
])
def test_eager_func(sig, t, expected):
    actual = tf.b_func_eager(t, {'sigma':sig, 'gamma':1})
    assert pt.approx(actual, abs=1e-5) == expected


@pt.mark.parametrize("sig, t, expected", [
    (3, 0, 3.15719),
    (3, 1, 0.157187),
    (3, 0.5, 0.704464)
])
def test_eager_deriv(sig, t, expected):
    actual = tf.b_dot_func_eager(t, {'sigma':sig, 'gamma':1})
    assert pt.approx(actual, abs=1e-5) == expected
