"""Calculate the bounds and multipliers for the wavelet functions."""
import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs_fast as hf

DEBUG = True


def test_calc_lmum_level1():
    level = 1
    lmum = hf.calc_lmum(level=level)
    if DEBUG:
        print("lmum:", lmum)
    expected = np.array([[0, 2, 4, 1], [0, 0.5, 1, 1]])
    assert np.allclose(lmum, expected)


def test_calc_lmum_level2():
    level = 2
    lmum = hf.calc_lmum(level=level)
    if DEBUG:
        print("lmum:", lmum)
    expected = np.array([
        [0, 2, 4, 1],
        [0, 0.5, 1, 1],
        [0, 0.25, 0.5, 1.4142135623730951],
        [0.5, 0.75, 1, 1.4142135623730951]
    ])
    assert np.allclose(lmum, expected)


def test_calc_lmum_level3():
    level = 3
    lmum = hf.calc_lmum(level=level)
    if DEBUG:
        print("lmum:", lmum)
    expected = np.array([
        [0, 2, 4, 1],
        [0, 0.5, 1, 1],
        [0, 0.25, 0.5, 1.4142135623730951],
        [0.5, 0.75, 1, 1.4142135623730951],
        [0, 0.125, 0.25, 2],
        [0.25, 0.375, 0.5, 2],
        [0.5, 0.625, 0.75, 2],
        [0.75, 0.875, 1, 2]
    ])
    assert np.allclose(lmum, expected)
