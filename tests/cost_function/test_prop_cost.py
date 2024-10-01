"""Tests to ensure that the analytic approximation of the propagator cost is correct
"""
import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../cost_function')))
import propagator as pp
import fourier as fr


@pt.mark.parametrize("rho", [0.1, 1.0, 10])  # 0.1, 1.0, 5.0])
def test_integral_risk_averse(rho):
    a_n = np.array([0, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_a_integral(a_n, b_n, lambd, rho)
    expected = (1 + lambd) * (rho - 1 + np.exp(-rho)) / rho ** 2
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    (0.25, -0.444441),
    (1.0, 0.432887),
    (10, 1.07144)
])
def test_integral_first_term1(rho, expected):
    a_n = np.array([1, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_a_integral(a_n, b_n, lambd, rho)
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    (0.1, 6.59075),
    (0.5, 6.94173),
    (1.0, 7.23051),
    (10.0, 4.95822)
])
def test_integral_both_terms(rho, expected):
    a_n = np.array([1, 0.5])
    b_n = np.array([0, 1])
    lambd = 5
    actual = pp.cost_fn_prop_a_integral(a_n, b_n, lambd, rho)
    assert pt.approx(actual, abs=1e-4) == expected


# -----------------------------------------------
# Test a combinatio of approximation and integration
# -----------------------------------------------
@pt.mark.parametrize("rho", [0.1, 1.0, 10])  # 0.1, 1.0, 5.0])
def test_integ_approx_risk_averse(rho):
    a_n = np.array([0, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_a_integ_approx(a_n, b_n, lambd, rho)
    expected = (1 + lambd) * (rho - 1 + np.exp(-rho)) / rho ** 2
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    (0.25, -0.444441),
    (1.0, 0.432887),
    (10, 1.07144)
])
def test_integ_approx_first_term1(rho, expected):
    a_n = np.array([1, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_a_integ_approx(a_n, b_n, lambd, rho)
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    (0.25, 2.50773),
    (1.0, 1.89434),
    (10, 0.327192),
])
def test_integ_approx_first_term2(rho, expected):
    a_n = np.array([0, 0])
    b_n = np.array([1, 0])
    lambd = 2
    actual = pp.cost_fn_prop_a_integ_approx(a_n, b_n, lambd, rho)
    assert pt.approx(actual, abs=1e-5) == expected


@pt.mark.parametrize("rho, expected", [
    (0.1, 6.59075),
    (0.5, 6.94173),
    (1.0, 7.23051),
    (10.0, 4.95822)
])
def test_integ_approx_both_terms(rho, expected):
    a_n = np.array([1, 0.5])
    b_n = np.array([0, 1])
    lambd = 5
    actual = pp.cost_fn_prop_a_integ_approx(a_n, b_n, lambd, rho)
    assert pt.approx(actual, abs=1e-4) == expected


# -----------------------------------------------
# Test the Fourier calculation of the propagator cost
# -----------------------------------------------
@pt.mark.parametrize("rho", [0.1, 1.0, 10])
def test_approx_risk_averse(rho):
    a_n = np.array([0, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_a_approx(a_n, b_n, lambd, rho, verbose=True)
    expected = (1 + lambd) * (rho - 1 + np.exp(-rho)) / rho ** 2
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    # (0.25, -0.444441),
    (1.0, 0.432887),
    # (10, 1.07144)
])
def test_approx_first_term1(rho, expected):
    a_n = np.array([1, 0])
    b_n = np.array([0, 0])
    lambd = 10
    actual = pp.cost_fn_prop_a_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, abs=1e-6) == expected


@pt.mark.parametrize("rho, expected", [
    (0.25, 2.50773),
    (1.0, 1.89434),
    (10, 0.327192),
])
def test_approx_first_term2(rho, expected):
    a_n = np.array([0, 0])
    b_n = np.array([1, 0])
    lambd = 2
    actual = pp.cost_fn_prop_a_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, abs=1e-5) == expected



@pt.mark.parametrize("rho, expected", [
    (0.1, 6.59075),
    (0.5, 6.94173),
    (1.0, 7.23051),
    (10.0, 4.95822)
])
def test_approx_both_terms(rho, expected):
    a_n = np.array([1, 0.5])
    b_n = np.array([0, 1])
    lambd = 5
    actual = pp.cost_fn_prop_a_approx(a_n, b_n, lambd, rho, verbose=True)
    assert pt.approx(actual, abs=1e-4) == expected
