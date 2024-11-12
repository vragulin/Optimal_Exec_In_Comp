""" Test the price displacement function using haar wavelets.
V. Ragulin, 11/09/2024
"""

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


@pt.mark.parametrize("t", [0, 0.11, 0.20, np.pi / 10, 0.5, 0.71, 1.0])
def test_input_coeffs(t):
    rho = 1.0
    haar = np.array([0.2, 0, 0.5, 0, 0, -0.5, 0.25, -0.25], dtype=float)
    level = len(haar).bit_length() - 1
    lmum = hf.calc_lmum(level)
    result = hf.price_haar(t, haar, rho, lmum=lmum)

    # Now calculate using q
    # uad as a test
    points = np.linspace(0, 1, 2 ** 2 + 1).astype(float)

    def integrand(s):
        m = hf.reconstruct_from_haar(s, haar, lmum=lmum)
        return m * np.exp(-rho * (t - s))

    result_quad = quad(integrand, 0, t, points=points)[0]

    if DEBUG:
        print(f"Integral result over [0, {t}]: haar: {result}, quad: {result_quad}")
    assert pt.approx(result, abs=1e-6) == result_quad


@pt.mark.parametrize("t", [0, 0.11, 0.20, np.pi / 10, 0.5, 0.71, 1.0])
@pt.mark.parametrize("alpha, beta, level", [
    (1, 0, 5),
    (1, -1, 5),
    (1, -20, 6)
])
def test_eager(t, alpha, beta, level):
    def f(t):
        return alpha * np.exp(beta * t)

    haar = hf.haar_coeff(f, level)
    rho = 1.0
    lmum = hf.calc_lmum(level)
    result = hf.price_haar(t, haar, rho, lmum=lmum)

    # Now calculate using quad as a test
    exp_integral = quad(lambda s: f(s) * np.exp(-rho * (t - s)), 0, t)[0]

    if DEBUG:
        print(f"Integral result over [0, {t}]: haar: {result}, quad: {exp_integral}")

    assert pt.approx(result, abs=5e-3 / 2 ** level) == exp_integral
