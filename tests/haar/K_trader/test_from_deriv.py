"""Test file for the from_func method of the SinesBlocks class
V. Ragulin - 10/21/2024
"""
import os
import sys
import pytest as pt
import numpy as np
import matplotlib.pyplot as plt

# Add necessary paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../../..', 'haar'),
])
import haar_funcs_fast as hf
import cost_haar_K as ck
import trading_intensity_funcs as ti

DEBUG = True
PLOT_POINTS = 100


@pt.mark.parametrize("a0, a1", [
    (0.0, 1.0),
    (0.0, 0.0),
    (1.0, 1.0),
    (1.0, 0.0),
    (0.3, 0.8),
    (4.0, -2.0),
])
def test_from_func_line(a0, a1):
    """ a0, a1 are the function values at t=0 (after the block) and t=1 (before the block) """
    level = 5
    lambd = 0.1
    func = lambda t: a0 + (a1 - a0) * t
    strat = ck.HaarStrat.from_deriv(func, level=level, lambd=lambd)

    if DEBUG:
        print(strat)

    assert strat.level == level, f"Level is incorrect, exp={level}, got={strat.level}"
    assert strat.lambd == lambd, f"Lambda is incorrect, exp={lambd}, got={strat.lambd}"
    for t in [0, 0.25, 0.5, 0.75]:
        f = strat.deriv(t)
        exp_f = func(t)
        assert np.isclose(f, exp_f,
                          atol=max(a0, a1) / 2 ** level), f"Price is incorrect for t={t}, exp={exp_f}, got={f}"


@pt.mark.parametrize("level", [5, 6, 7])
def test_from_func_sine(level):
    # Test a sine trading intensity
    """ a0, a1 are the function values at t=0 (after the block) and t=1 (before the block) """
    lambd = 1
    func = lambda t: 0.5 * np.sin(np.pi * t) - 0.3 * np.sin(np.pi * 2 * t)
    strat = ck.HaarStrat.from_deriv(func, level=level, lambd=lambd)

    if DEBUG:
        print(strat)
        x_values = np.linspace(0, 1, PLOT_POINTS + 1)
        f_values = [func(x) for x in x_values]
        strat_values = [strat.deriv(x) for x in x_values]
        plt.plot(x_values[:-1], f_values[:-1], label='Function')
        plt.plot(x_values[:-1], strat_values[:-1], label='HaarStrat')
        plt.legend()
        plt.show()

    for t in [0, 0.25, 0.5, 0.75, 0.97]:
        f = strat.deriv(t)
        exp_f = func(t)
        assert np.isclose(f, exp_f, atol=2 / 2 ** level), f"Price is incorrect for t={t}, exp={exp_f}, got={f}"


@pt.mark.parametrize("level", [5, 6, 7])
def test_from_eager(level):
    # Test an eager trading intensity
    lambd = 1
    a = 3
    func = lambda t: 0.5 * ti.eager(t, a)
    strat = ck.HaarStrat.from_deriv(func, level=level, lambd=lambd)

    if DEBUG:
        print(strat)
        x_values = np.linspace(0, 1, PLOT_POINTS + 1)
        f_values = [func(x) for x in x_values]
        strat_values = [strat.deriv(x) for x in x_values]
        plt.plot(x_values[:-1], f_values[:-1], label='Function')
        plt.plot(x_values[:-1], strat_values[:-1], label='HaarStrat')
        plt.legend()
        plt.show()

    for t in [0, 0.25, 0.5, 0.75, 0.97]:
        f = strat.deriv(t)
        exp_f = func(t)
        assert np.isclose(f, exp_f, atol=2.5 / 2 ** level), f"Price is incorrect for t={t}, exp={exp_f}, got={f}"


@pt.mark.parametrize("level", [5, 6, 7])
def test_from_bucket(level):
    # Test a bucket trading intensity
    lambd = 1
    a = 3
    func = ti.bucket
    func_args = start, end, max_pos = 0.1, 0.9, 5
    strat = ck.HaarStrat.from_deriv(func, level=level, lambd=lambd, func_args=func_args)

    if DEBUG:
        print(strat)
        x_values = np.linspace(0, 1, PLOT_POINTS + 1)
        f_values = [func(x, *func_args) for x in x_values]
        strat_values = [strat.deriv(x) for x in x_values]
        plt.plot(x_values[:-1], f_values[:-1], label='Function')
        plt.plot(x_values[:-1], strat_values[:-1], label='HaarStrat')
        plt.legend()
        plt.show()

    for t in [0, 0.25, 0.5, 0.75, 0.97]:
        f = strat.deriv(t)
        exp_f = func(t, *func_args)
        assert np.isclose(f, exp_f, atol=2.5 / 2 ** level), f"Price is incorrect for t={t}, exp={exp_f}, got={f}"
