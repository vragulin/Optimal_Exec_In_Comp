
""" Test the price function
    V. Ragulin - 11/15/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

from g_one_trader import position


def test_position():
    t_n = np.array([0.1, 0.2, 0.4, 0.8, 0.9])
    x_n = np.array([10, 20, -5, 15, 10])

    assert pt.approx(position(0, t_n, x_n), abs=1e-6) == 0
    assert pt.approx(position(1, t_n, x_n), abs=1e-6) == np.sum(x_n)
    assert pt.approx(position(0.5, t_n, x_n), abs=1e-6) == np.sum(x_n[:3])
    assert pt.approx(position(0.9, t_n, x_n), abs=1e-6) == np.sum(x_n)


def test_position_trades_0_1():
    t_n = np.array([0, 0.4, 1])
    x_n = np.array([10, 11, 12])

    assert pt.approx(position(0, t_n, x_n), abs=1e-6) == 0
    assert pt.approx(position(1, t_n, x_n), abs=1e-6) == np.sum(x_n)
    assert pt.approx(position(0.5, t_n, x_n), abs=1e-6) == np.sum(x_n[:2])

