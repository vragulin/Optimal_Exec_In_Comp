"""Tests the propagator cost function for Trader B
"""
import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../cost_function')))
import propagator as pp
import fourier as fr


@pt.mark.parametrize("rho", [0.1, 1.0, 10])
def test_approx_risk_averse(rho):
    a_n = np.array([0, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_b_approx(a_n, b_n, lambd, rho, verbose=True)
    expected = lambd * (1 + lambd) * (rho - 1 + np.exp(-rho)) / rho ** 2
    assert pt.approx(actual, rel=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    (0.25, 56.3158),
    (1.0, 44.4203),
    (10, 10.186)
])
def test_approx_first_term1(rho, expected):
    a_n = np.array([1, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_b_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, rel=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    (0.25, 2.09194),
    (1.0, 2.77551),
    (10, 1.95236),
])
def test_approx_first_term2(rho, expected):
    a_n = np.array([0, 0])
    b_n = np.array([1, 0])
    lambd = 2
    actual = pp.cost_fn_prop_b_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, rel=1e-5) == expected


@pt.mark.parametrize("rho, expected", [
    (0.5, 12.4725),
    (1.0, 10.5458),
    (5.0, 4.32414),
])
def test_approx_second_term1(rho, expected):
    a_n = np.array([0, 1])
    b_n = np.array([0, 0])
    lambd = 5
    actual = pp.cost_fn_prop_b_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, rel=1e-5) == expected


@pt.mark.parametrize("rho, expected", [
    (0.5, 15.5103),
    (1.0, 17.4502),
    (5.0, 31.8569),
])
def test_approx_second_term2(rho, expected):
    a_n = np.array([0, 0])
    b_n = np.array([0, 1])
    lambd = 5
    actual = pp.cost_fn_prop_b_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, rel=1e-5) == expected


@pt.mark.parametrize("rho, expected", [
    (0.1, 11.384),
    (0.5, 11.9372),
    (1.0, 14.1676),
    (10.0, 32.3481),
])
def test_approx_both_terms(rho, expected):
    a_n = np.array([1, 0.5])
    b_n = np.array([0, 1])
    lambd = 5
    actual = pp.cost_fn_prop_b_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, rel=1e-4) == expected
