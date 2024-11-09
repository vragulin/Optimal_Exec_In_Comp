import pytest as pt
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs as hf

DEBUG = True


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [3, 5])
def test_integrate_haar_quad_const(interval: tuple[float, float], level: int):
    def func(t):
        return 1

    haar_coeffs = hf.haar_coeff(func, level=level)
    integral = hf.integrate_haar_quad(haar_coeffs, *interval)
    exp = interval[1] - interval[0]
    if DEBUG:
        print(f"Integral: {integral}, Expected: {exp}")
    assert pt.approx(integral, abs=1e-8) == interval[1] - interval[0]


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [3, 5])
def test_integrate_haar_quad_linear(interval: tuple[float, float], level: int):
    def func(t):
        return t

    haar_coeffs = hf.haar_coeff(func, level=level)
    integral = hf.integrate_haar_quad(haar_coeffs, *interval)
    exp_integral = 0.5 * (interval[1]**2 - interval[0]**2)
    if DEBUG:
        print(f"Integral: {integral}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=2e-2*2**(-level)) == exp_integral


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [6, 7])
def test_integrate_haar_quad_linear_no_points(interval: tuple[float, float], level: int):
    def func(t):
        return t

    haar_coeffs = hf.haar_coeff(func, level=level)
    integral = hf.integrate_haar_quad(haar_coeffs, *interval, points=False)
    exp_integral = 0.5 * (interval[1]**2 - interval[0]**2)
    if DEBUG:
        print(f"Integral: {integral}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=4e-2*2**(-level)) == exp_integral


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [3, 5])
def test_integrate_haar_quad_sin(interval: tuple[float, float], level: int):
    def func(t):
        return np.sin(np.pi * 2 * t)

    def integral_func(t):
        return -1 / (np.pi * 2) * np.cos(np.pi * 2 * t)

    haar_coeffs = hf.haar_coeff(func, level=level)
    integral = hf.integrate_haar_quad(haar_coeffs, *interval)
    exp_integral = integral_func(interval[1]) - integral_func(interval[0])
    if DEBUG:
        print(f"Integral: {integral}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=5e-2*2**(-level)) == exp_integral
