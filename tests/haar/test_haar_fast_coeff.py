import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs_fast as hf

DEBUG = True
TOL = 1e-6


@pt.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_haar_coeff_zero(level):
    def func(t):
        return 0

    h = hf.haar_coeff(func, level)

    if DEBUG:
        print("haar:", h)
    assert h[0] == 0
    assert len(h) == 2 ** level
    for i, coef in enumerate(h[1:]):
        assert pt.approx(coef, abs=TOL) == 0


@pt.mark.parametrize("level", [1, 2, 3, 4, 5])
@pt.mark.parametrize("c", [1, 2, 3, 4, 5])
def test_haar_coeff_const(level, c):
    def func(t):
        return c

    h = hf.haar_coeff(func, level)

    if DEBUG:
        print("haar:", h)
    assert h[0] == c
    assert len(h) == 2 ** level
    for i, coef in enumerate(h[1:]):
        assert pt.approx(coef, abs=TOL) == 0


@pt.mark.parametrize("level", [1, 2, 3])
@pt.mark.parametrize("c", [1, 2, 3])
def test_haar_coeff_x(level, c):
    def func(t):
        return c * t

    h = hf.haar_coeff(func, level)

    if DEBUG:
        print("haar:", h)
    assert len(h) == 2 ** level
    for i, coef in enumerate(h):
        if i == 0:
            assert h[0] == c / 2
        else:
            j, k = hf.i_to_nk(i)
            step = 1 / 2 ** j
            lb = step * k * 2
            mid = step * (k * 2 + 0.5)
            ub = step * (k * 2 + 1)
            mult = 2 ** (j / 2)
            exp_coef = c / 2 * mult * (mid ** 2 * 2 - ub
                                      ** 2 - lb ** 2)
            assert np.isclose(coef, exp_coef, atol=1e-6)


@pt.mark.parametrize("level", [1, 2, 3, 5])
@pt.mark.parametrize("k0", [0, 1])
@pt.mark.parametrize("k2", [1, 2, 3])
def test_haar_coeff_x_sq(level, k0, k2):
    """ Test for quadratic function of the type y=h[0]+c2*t^2 """

    def func(t):
        return k0 + k2 * t ** 2

    def int_func(t):
        return k0 * t + k2 / 3 * t ** 3

    h = hf.haar_coeff(func, level)

    if DEBUG:
        print("haar:", h)
    assert len(h) == 2 ** level
    for i, coef in enumerate(h):
        if i == 0:
            assert pt.approx(h[0], abs=1e-6) == (int_func(1) - int_func(0))
        else:
            j, k = hf.i_to_nk(i)
            step = 1 / 2 ** j
            lb = step * k
            mid = step * (k + 0.5)
            ub = step * (k + 1)
            mult = 2 ** (j / 2)
            exp_coef = mult * (2 * int_func(mid) - int_func(ub) - int_func(lb))
            assert np.isclose(coef, exp_coef, atol=1e-10)


@pt.mark.parametrize("level", [1, 2, 3, 5, 10, 12])
@pt.mark.parametrize("k0", [0, 1])
@pt.mark.parametrize("k2", [1, 2, 3])
def test_haar_coeff_exp(level, k0, k2):
    """ Test for quadratic function of the type y=h[0]+c2*t^2 """

    sigma = 3

    def func(t):
        return np.exp(-sigma * t)

    def int_func(t):
        return -1 / sigma * np.exp(-sigma * t)

    h = hf.haar_coeff(func, level)

    if DEBUG:
        print("haar:", h)
    assert len(h) == 2 ** level
    for i, coef in enumerate(h):
        if i == 0:
            assert pt.approx(h[0], abs=1e-6) == (int_func(1) - int_func(0))
        else:
            j, k = hf.i_to_nk(i)
            step = 1 / 2 ** j
            lb = step * k
            mid = step * (k + 0.5)
            ub = step * (k + 1)
            mult = 2 ** (j / 2)
            exp_coef = mult * (2 * int_func(mid) - int_func(ub) - int_func(lb))
            assert np.isclose(coef, exp_coef, atol=1e-10)


def test_haar_coeff_w_args():
    def func(t, a, b):
        return a * np.exp(-b * t)

    def int_func(t):
        return -a / b * np.exp(-b * t)

    a = 2
    b = 3
    level = 5

    h = hf.haar_coeff(func, level, func_args=(a, b))

    if DEBUG:
        print("haar:", h)
    assert len(h) == 2 ** level
    for i, coef in enumerate(h):
        if i == 0:
            assert pt.approx(h[0], abs=1e-6) == (int_func(1) - int_func(0))
        else:
            j, k = hf.i_to_nk(i)
            step = 1 / 2 ** j
            lb = step * k
            mid = step * (k + 0.5)
            ub = step * (k + 1)
            mult = 2 ** (j / 2)
            exp_coef = mult * (2 * int_func(mid) - int_func(ub) - int_func(lb))
            assert np.isclose(coef, exp_coef, atol=1e-10)
