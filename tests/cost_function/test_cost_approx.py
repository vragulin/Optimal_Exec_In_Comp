"""
Tests to ensure that the analytic approximation is correct
"""
import pytest as pt
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../cost_function')))
import cost_function_approx as ca


@pt.mark.parametrize("lambd, kappa", [
    (1, 1),
    (5, 1),
    (10, 1),
    (50, 1),
    (100, 1),
    (1, 5),
    (1, 10),
    (1, 50),
    (2, 10),
    (10, 10),
    (100, 100),
])
def test_diff_2_sines_lambda_kappa(lambd, kappa):
    a_n = [1.0, 0.5]
    b_n = [-1.0, -5.0]
    assert len(a_n) == len(b_n)
    N = len(a_n)

    # Exact Integral of the Fourier approximations
    I_exact = ca.fourier_integral_cost_fn(a_n, b_n, kappa, lambd)
    I_approx = ca.cost_fn_a_approx(a_n, b_n, kappa, lambd)

    assert np.isclose(I_exact, I_approx, atol=0.001)


@pt.mark.parametrize("lambd, kappa", [
    (1, 1),
    (5, 1),
    (10, 1),
    (50, 1),
    (100, 1),
    (1, 5),
    (1, 10),
    (1, 50),
    (2, 10),
    (10, 10),
    (100, 100),
])
def test_diff_4_sines_lambda_kappa(lambd, kappa):
    a_n = [2.0, 1.0, -0.7, 0.6]
    b_n = [-1.0, -5.0, 1.1, -0.9]
    assert len(a_n) == len(b_n)
    N = len(a_n)

    # Exact Integral of the Fourier approximations
    I_exact = ca.fourier_integral_cost_fn(a_n, b_n, kappa, lambd)
    I_approx = ca.cost_fn_a_approx(a_n, b_n, kappa, lambd)

    assert np.isclose(I_exact, I_approx, atol=0.001)


@pt.mark.parametrize("lambd, kappa, a_exp, b_exp", [
    (1, 1, 1, 1),
    (5, 1, 0.5, 2),
    (10, 1, 0.5, 3),
    (50, 1, 3, 0.5),
    (100, 1, 2, 2),
    (1, 5, 0.2, 4),
    (1, 10, 4, 0.3),
    (1, 50, 0.5, 2),
    (2, 10, 4, 0.2),
    (10, 10, 0.2, 0.3),
    (100, 100, 4, 4),
])
def test_diff_expsin_func_exp_lambda_kappa(lambd, kappa, a_exp, b_exp):
    def a_func(t, kappa, lambd):
        return t ** a_exp + lambd * np.sin(np.pi * t)

    def b_func(t, kappa, lambd):
        return t ** b_exp + kappa * np.sin(2 * np.pi * t)

    def a_func_dot(t, kappa, lamb):
        return a_exp * t ** (a_exp - 1) + lambd * np.pi * np.cos(np.pi * t)

    def b_func_dot(t, kappa, lamb):
        return b_exp * t ** (b_exp - 1) + kappa * 2 * np.pi * np.cos(2 * np.pi * t)

    N = 10

    a_n, b_n = ca.compute_sine_series_for_functions(a_func, b_func, kappa, lambd, N)

    # Exact Integral of the Fourier approximations
    I_exact = ca.fourier_integral_cost_fn(a_n, b_n, kappa, lambd)
    I_approx = ca.cost_fn_a_approx(a_n, b_n, kappa, lambd)

    assert np.isclose(I_exact, I_approx, atol=0.001)


@pt.mark.parametrize("lambd, kappa, a_exp, b_exp", [
    (1, 1, 1, 1),
    (5, 1, 0.5, 2),
    (10, 1, 0.5, 3),
    (50, 1, 3, 0.5),
    (100, 1, 2, 2),
    (1, 5, 0.2, 4),
    (1, 10, 4, 0.3),
    (1, 50, 0.5, 2),
    (2, 10, 4, 0.2),
    (10, 10, 0.2, 0.3),
    (100, 100, 4, 4),
])
def test_diff_old_v_new_approx_func_a(lambd, kappa, a_exp, b_exp):
    def a_func(t, kappa, lambd):
        return t ** a_exp + lambd * np.sin(np.pi * t)

    def b_func(t, kappa, lambd):
        return t ** b_exp + kappa * np.sin(2 * np.pi * t)

    def a_func_dot(t, kappa, lamb):
        return a_exp * t ** (a_exp - 1) + lambd * np.pi * np.cos(np.pi * t)

    def b_func_dot(t, kappa, lamb):
        return b_exp * t ** (b_exp - 1) + kappa * 2 * np.pi * np.cos(2 * np.pi * t)

    N = 10

    a_n, b_n = ca.compute_sine_series_for_functions(a_func, b_func, kappa, lambd, N)

    # Exact Integral of the Fourier approximations
    I_approx = ca.cost_fn_a_approx(a_n, b_n, kappa, lambd)
    I_approx_old = ca.cost_fn_a_approx_old(a_n, b_n, kappa, lambd)

    assert np.isclose(I_approx_old, I_approx, atol=0.001)


@pt.mark.parametrize("lambd, kappa, a_exp", [
    (1, 1, 1),
    (1, 5, 0.2),
    (1, 10, 4),
    (1, 50, 0.5),
])
def test_cost_b_equals_a_when_lambd_1(lambd, kappa, a_exp):
    def a_func(t, kappa, lambd):
        return t ** a_exp + lambd * np.sin(np.pi * t)

    def b_func(t, kappa, lambd):
        return a_func(t, kappa, lambd)

    def a_func_dot(t, kappa, lamb):
        return a_exp * t ** (a_exp - 1) + lambd * np.pi * np.cos(np.pi * t)

    def b_func_dot(t, kappa, lamb):
        return b_func_dot(t, kappa, lamb)

    N = 10

    a_n, b_n = ca.compute_sine_series_for_functions(a_func, b_func, kappa, lambd, N)

    # Exact Integral of the Fourier approximations
    I_approx_a = ca.cost_fn_a_approx(a_n, b_n, kappa, lambd, verbose=True)
    I_approx_b = ca.cost_fn_b_approx(a_n, b_n, kappa, lambd, verbose=True)

    assert np.isclose(I_approx_a, I_approx_b, atol=0.001)


@pt.mark.parametrize("lambd, kappa, a_exp, b_exp", [
    (1, 1, 1, 1),
    (10, 1, 1, 1),
    (5, 1, 0.5, 2),
    (10, 1, 0.5, 3),
    (50, 1, 3, 0.5),
    (100, 1, 2, 2),
    (1, 5, 0.2, 4),
    (1, 10, 4, 0.3),
    (1, 50, 0.5, 2),
    (2, 10, 4, 0.2),
    (10, 10, 0.2, 0.3),
    (100, 100, 4, 4),
])
def test_cost_b_in_terms_of_cost_a(lambd, kappa, a_exp, b_exp):
    def a_func(t, kappa, lambd):
        return t ** a_exp + lambd * np.sin(np.pi * t)

    def b_func(t, kappa, lambd):
        return t ** b_exp + kappa * np.sin(2 * np.pi * t)

    def a_func_dot(t, kappa, lamb):
        return a_exp * t ** (a_exp - 1) + lambd * np.pi * np.cos(np.pi * t)

    def b_func_dot(t, kappa, lamb):
        return b_exp * t ** (b_exp - 1) + kappa * 2 * np.pi * np.cos(2 * np.pi * t)


    N = 10

    a_n, b_n = ca.compute_sine_series_for_functions(a_func, b_func, kappa, lambd, N)

    # Exact Integral of the Fourier approximations
    I_approx_b = ca.cost_fn_b_approx(a_n, b_n, kappa, lambd, verbose=True)
    I_approx_b1 = lambd**2 * ca.cost_fn_a_approx(b_n, a_n, kappa, 1 / lambd, verbose=True)

    assert np.isclose(I_approx_b1, I_approx_b, atol=0.001)
