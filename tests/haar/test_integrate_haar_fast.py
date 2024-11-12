import pytest as pt
import numpy as np
from scipy.integrate import quad
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs_fast as hf

DEBUG = False


# Example Haar coefficients
def test_integrate_input_coeffs():

    haar = np.array([0.5, 0, 0.25, -0.25, 0.125, -0.125, 0, 0])
    a, b = 0.3, 0.8  # integration limits in [0, 1]

    # Compute the integral over [a, b]
    level = len(haar).bit_length() - 1
    lmum = hf.calc_lmum(level)
    result = hf.integrate_haar(haar, a, b, lmum)
    expected = 0.12107864
    assert pt.approx(result, abs=1e-6) == expected, f"Expected {expected}, got {result}"


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [3, 5])
def test_integrate_haar_const(interval: tuple[float, float], level: int):
    def func(t):
        return 1

    haar_coeffs = hf.haar_coeff(func, level=level)
    lmum = hf.calc_lmum(level)
    integral = hf.integrate_haar(haar_coeffs, *interval, lmum=lmum)
    quad_haar_int = quad(lambda t: hf.reconstruct_from_haar(t, haar_coeffs, lmum), *interval)[0]
    exp_integral = quad(func, *interval)[0]
    if DEBUG:
        print(f"Integral: {integral}, Quad_haar: {quad_haar_int}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=1e-8) == quad_haar_int
    assert pt.approx(integral, abs=2e-2*2**(-level)) == exp_integral


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [3, 5])
def test_integrate_haar_linear(interval: tuple[float, float], level: int):
    def func(t):
        return 2 * t + 1

    haar_coeffs = hf.haar_coeff(func, level=level)
    lmum = hf.calc_lmum(level)
    integral = hf.integrate_haar(haar_coeffs, *interval, lmum=lmum)
    quad_haar_int = hf.integrate_haar_quad(haar_coeffs, *interval, lmum=lmum)
    exp_integral = quad(func, *interval)[0]
    if DEBUG:
        print(f"Integral: {integral}, Quad_haar: {quad_haar_int}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=1e-8) == quad_haar_int
    assert pt.approx(integral, abs=5e-2*2**(-level)) == exp_integral


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [3, 5])
def test_integrate_haar_quadratic(interval: tuple[float, float], level: int):
    def func(t):
        return 2 * t**2 + 2 * t + 1

    haar_coeffs = hf.haar_coeff(func, level=level)
    lmum = hf.calc_lmum(level)
    integral = hf.integrate_haar(haar_coeffs, *interval, lmum=lmum)
    quad_haar_int = hf.integrate_haar_quad(haar_coeffs, *interval, lmum=lmum)
    exp_integral = quad(func, *interval)[0]
    if DEBUG:
        print(f"Integral: {integral}, Quad_haar: {quad_haar_int}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=1e-8) == quad_haar_int
    assert pt.approx(integral, abs=5e-2*2**(-level)) == exp_integral


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [3, 5])
def test_integrate_haar_sin(interval: tuple[float, float], level: int):
    def func(t):
        return 2 * np.sin(3*np.pi*t)

    haar_coeffs = hf.haar_coeff(func, level=level)
    lmum = hf.calc_lmum(level)
    integral = hf.integrate_haar(haar_coeffs, *interval, lmum=lmum)
    quad_haar_int = hf.integrate_haar_quad(haar_coeffs, *interval, lmum=lmum)
    exp_integral = quad(func, *interval)[0]
    if DEBUG:
        print(f"Integral: {integral}, Quad_haar: {quad_haar_int}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=1e-8) == quad_haar_int
    assert pt.approx(integral, abs=0.3*2**(-level)) == exp_integral


@pt.mark.parametrize("interval", [
    (0, 1), (0, 0.3), (0.22, 1), (0.22, 0.76)])
@pt.mark.parametrize("level", [6, 7, 8])
def test_integrate_haar_sin_hi_precision(interval: tuple[float, float], level: int):
    def func(t):
        return 2 * np.sin(3*np.pi*t)

    haar_coeffs = hf.haar_coeff(func, level=level)
    lmum = hf.calc_lmum(level)
    integral = hf.integrate_haar(haar_coeffs, *interval, lmum=lmum)
    exp_integral = quad(func, *interval)[0]
    if DEBUG:
        print(f"Integral: {integral}, Expected: {exp_integral}")
    assert pt.approx(integral, abs=3e-2*2**(-level)) == exp_integral
