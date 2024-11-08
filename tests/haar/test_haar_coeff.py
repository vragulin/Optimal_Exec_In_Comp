import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs as hf

DEBUG = True


@pt.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_haar_coeff_zero(level):
    def func(t):
        return 0

    c0, coeffs = hf.haar_coeff(func, (0, 1), level)

    if DEBUG:
        print("c0:", c0)
        print("coeffs:", coeffs)
    assert c0 == 0
    assert len(coeffs) == 2 ** level - 1
    for j, k, djk in coeffs:
        assert pt.approx(djk, abs=1e-6) == 0


@pt.mark.parametrize("level", [1, 2, 3, 4, 5])
@pt.mark.parametrize("c", [1, 2, 3, 4, 5])
def test_haar_coeff_const(level, c):
    def func(t):
        return c

    c0, coeffs = hf.haar_coeff(func, (0, 1), level)

    if DEBUG:
        print("c0:", c0)
        print("coeffs:", coeffs)
    assert c0 == c
    assert len(coeffs) == 2 ** level - 1
    for j, k, djk in coeffs:
        assert pt.approx(djk, abs=1e-6) == 0


@pt.mark.parametrize("level", [1, 2, 3])
@pt.mark.parametrize("c", [1, 2, 3])
def test_haar_coeff_x(level, c):
    def func(t):
        return c * t

    c0, coeffs = hf.haar_coeff(func, (0, 1), level)

    if DEBUG:
        print("c0:", c0)
        print("coeffs:", coeffs)
    assert c0 == c / 2
    assert len(coeffs) == 2 ** level - 1
    for j, k, djk in coeffs:
        step = 1 / 2 ** j
        lb = step * k * 2
        mid = step * (k * 2 + 0.5)
        ub = step * (k * 2 + 1)
        mult = 2 ** (j / 2)
        exp_djk = c / 2 * mult * (mid ** 2 * 2 - ub ** 2 - lb ** 2)
        assert np.isclose(djk, exp_djk, atol=1e-6)


@pt.mark.parametrize("level", [1, 2, 3, 5])
@pt.mark.parametrize("k0", [0, 1])
@pt.mark.parametrize("k2", [1, 2, 3])
def test_haar_coeff_x_sq(level, k0, k2):
    """ Test for quadratic function of the type y=c0+c2*t^2 """

    def func(t):
        return k0 + k2 * t ** 2

    def int_func(t):
        return k0 * t + k2 / 3 * t ** 3

    c0, coeffs = hf.haar_coeff(func, (0, 1), level)

    if DEBUG:
        print("c0:", c0)
        print("coeffs:", coeffs)
    assert pt.approx(c0, abs=1e-6) == (int_func(1) - int_func(0))
    assert len(coeffs) == 2 ** level - 1
    for j, k, djk in coeffs:
        step = 1 / 2 ** j
        lb = step * k
        mid = step * (k + 0.5)
        ub = step * (k + 1)
        mult = 2 ** (j / 2)
        exp_djk = mult * (2 * int_func(mid) - int_func(ub) - int_func(lb))
        assert np.isclose(djk, exp_djk, atol=1e-10)


@pt.mark.parametrize("level", [1, 2, 3, 5, 10, 12])
@pt.mark.parametrize("k0", [0, 1])
@pt.mark.parametrize("k2", [1, 2, 3])
def test_haar_coeff_exp(level, k0, k2):
    """ Test for quadratic function of the type y=c0+c2*t^2 """

    sigma = 3
    
    def func(t):
        return np.exp(-sigma * t)

    def int_func(t):
        return -1 / sigma * np.exp(-sigma * t)

    c0, coeffs = hf.haar_coeff(func, (0, 1), level)

    if DEBUG:
        print("c0:", c0)
        print("coeffs:", coeffs)
    assert pt.approx(c0, abs=1e-6) == (int_func(1) - int_func(0))
    assert len(coeffs) == 2 ** level - 1
    for j, k, djk in coeffs:
        step = 1 / 2 ** j
        lb = step * k
        mid = step * (k + 0.5)
        ub = step * (k + 1)
        mult = 2 ** (j / 2)
        exp_djk = mult * (2 * int_func(mid) - int_func(ub) - int_func(lb))
        assert np.isclose(djk, exp_djk, atol=1e-10)
