"""
Tests to ensure that the analytic approximation of the propagator is correct
"""
import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../cost_function')))
import propagator as pp
import fourier as fr


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_integral_risk_averse(rho):
    a_n = np.array([0, 0])
    b_n = np.array([0, 0])
    lambd = 10
    t = 0.5
    actual = pp.prop_price_impact_integral(t, a_n, b_n, lambd, rho)
    expected = (1 - np.exp(-rho * t)) * (1 + lambd) / rho
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_integral_first_term1(rho):
    a_n = [1, 0]
    b_n = np.array([0, 0])
    lambd = 10
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    actual = pp.prop_price_impact_integral(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[0] + b_n[0] * lambd) * pi \
               * (rho * cos(pi * t) + pi * sin(pi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + pi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_integral_first_term2(rho):
    a_n = [0.25, 0]
    b_n = np.array([0.25, 0])
    lambd = 3
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    actual = pp.prop_price_impact_integral(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[0] + b_n[0] * lambd) * pi \
               * (rho * cos(pi * t) + pi * sin(pi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + pi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_integral_second_term1(rho):
    a_n = np.array([0., 0.25])
    b_n = np.array([0., 0.25])
    lambd = 2
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    tpi = 2 * pi
    actual = pp.prop_price_impact_integral(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[1] + b_n[1] * lambd) * tpi \
               * (rho * cos(tpi * t) + tpi * sin(tpi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + tpi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_integral_second_term2(rho):
    a_n = np.array([0., 0.25])
    b_n = np.array([0., 1.0])
    lambd = 1
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    tpi = 2 * pi
    actual = pp.prop_price_impact_integral(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[1] + b_n[1] * lambd) * tpi \
               * (rho * cos(tpi * t) + tpi * sin(tpi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + tpi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_integral_both_terms(rho):
    a_n = np.array([1., 0.25])
    b_n = np.array([-1., 1.0])
    lambd = 1
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    tpi = 2 * pi
    actual = pp.prop_price_impact_integral(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[1] + b_n[1] * lambd) * tpi \
               * (rho * cos(tpi * t) + tpi * sin(tpi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + tpi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


# ****************************************************
# Same tests with the approximate calculation
# ****************************************************
@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_risk_averse(rho):
    a_n = np.array([0, 0])
    b_n = np.array([0, 0])
    lambd = 10
    t = 0.5
    actual = pp.prop_price_impact_approx(t, a_n, b_n, lambd, rho)
    expected = (1 - np.exp(-rho * t)) * (1 + lambd) / rho
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_first_term1(rho):
    a_n = [1, 0]
    b_n = np.array([0, 0])
    lambd = 10
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    actual = pp.prop_price_impact_approx(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[0] + b_n[0] * lambd) * pi \
               * (rho * cos(pi * t) + pi * sin(pi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + pi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_first_term2(rho):
    a_n = [0.25, 0]
    b_n = np.array([0.25, 0])
    lambd = 3
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    actual = pp.prop_price_impact_approx(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[0] + b_n[0] * lambd) * pi \
               * (rho * cos(pi * t) + pi * sin(pi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + pi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_second_term1(rho):
    a_n = np.array([0., 0.25])
    b_n = np.array([0., 0.25])
    lambd = 2
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    tpi = 2 * pi
    actual = pp.prop_price_impact_approx(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[1] + b_n[1] * lambd) * tpi \
               * (rho * cos(tpi * t) + tpi * sin(tpi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + tpi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_second_term2(rho):
    a_n = np.array([0., 0.25])
    b_n = np.array([0., 1.0])
    lambd = 1
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    tpi = 2 * pi
    actual = pp.prop_price_impact_approx(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[1] + b_n[1] * lambd) * tpi \
               * (rho * cos(tpi * t) + tpi * sin(tpi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + tpi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_both_terms(rho):
    a_n = np.array([1., 0.25])
    b_n = np.array([-1., 1.0])
    lambd = 1
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    tpi = 2 * pi
    actual = pp.prop_price_impact_approx(t, a_n, b_n, lambd, rho)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[1] + b_n[1] * lambd) * tpi \
               * (rho * cos(tpi * t) + tpi * sin(tpi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + tpi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_both_terms_precomp(rho):
    a_n = np.array([1., 0.25])
    b_n = np.array([-1., 1.0])
    lambd = 1
    t = 0.5
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    tpi = (2 * pi)
    # Precompute the trigonometric functions
    n = np.arange(1,3)
    sin_n_pi_t = sin(n * pi * t)
    cos_n_pi_t = cos(n * pi * t)
    precomp = {'sin_n_pi_t': sin_n_pi_t, 'cos_n_pi_t': cos_n_pi_t}
    actual = pp.prop_price_impact_approx(t, a_n, b_n, lambd, rho, precomp=precomp)
    expected = (1 - exp(-rho * t)) * (1 + lambd) / rho \
               + (a_n[1] + b_n[1] * lambd) * tpi \
               * (rho * cos(tpi * t) + tpi * sin(tpi * t) - rho * exp(-rho * t)) \
               / (rho ** 2 + tpi ** 2)

    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho", [0.1, 1, 10])  # 0.1, 1.0, 5.0])
def test_approx_analytic_func(rho):
    """ Specify analytic funcs for a and b """
    def a_func(t, kappa, lambd):
        return t ** 2 + np.sin(np.pi * t)

    def b_func(t, kappa, lambd):
        return t + 2 * np.sin(2 * np.pi * t)

    lambd = 1
    kappa = 0
    t = 0.5
    N = 20

    a_n, b_n = fr.find_fourier_coefficients([a_func, b_func], kappa, lambd, N)
    actual = pp.prop_price_impact_approx(t, a_n, b_n, 1, rho)
    expected = pp.prop_price_impact_integral(t, a_n, b_n, 1, rho)
    assert pt.approx(actual, abs=1e-4) == expected
