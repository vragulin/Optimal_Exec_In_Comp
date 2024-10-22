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
    os.path.join(CURRENT_DIR, '../..', 'lin_prop_blocks'),
    os.path.join(CURRENT_DIR, '../..', 'cost_function'),
    os.path.join(CURRENT_DIR, '../..', 'optimizer_qp')
])

import prop_blocks as pb
import trading_funcs as tf

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
    N = 20
    lambd = 0.1
    func = lambda t: a0 + (a1 - a0) * t
    strat = pb.SinesBlocks.from_func(func, N=N, lambd=lambd)

    if DEBUG:
        print(strat)

    for t in [0, 0.25, 0.5, 0.75, 1]:
        p = strat.calc(t)
        exp_p = func(t) if t not in {0, 1} else t
        assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect for t={t}, exp={exp_p}, got={p}"


def test_from_func_sine_with_blocks():
    # Test a sine function
    # Test a risk-averse function
    """ a0, a1 are the function values at t=0 (after the block) and t=1 (before the block) """
    N = 50
    lambd = 1
    blocks = (0.1, 0.25)
    func = lambda t: blocks[0] + (
            0.5 * np.sin(np.pi * t) - 0.3 *np.sin(np.pi * 2 *t))  * (1 - sum(blocks))
    strat = pb.SinesBlocks.from_func(func, N=N, lambd=lambd)

    if DEBUG:
        print(strat)
        x_values = np.linspace(0, 1, PLOT_POINTS+1)
        f_values = [func(x) for x in x_values]
        strat_values = [strat.calc(x) for x in x_values]
        plt.plot(x_values, f_values, label='Function')
        plt.plot(x_values, strat_values, label='SinesBlocks')
        plt.legend()
        plt.show()

    for t in [0, 0.25, 0.5, 0.75, 1]:
        p = strat.calc(t)
        exp_p = func(t) if t not in {0, 1} else t
        assert np.isclose(p, exp_p, atol=1e-3), f"Price is incorrect for t={t}, exp={exp_p}, got={p}"


def test_from_eager():
    # Test a risk-averse function
    """ a0, a1 are the function values at t=0 (after the block) and t=1 (before the block) """
    N = 50
    lambd = 1
    sigma = 10
    blocks = (0.25, 0.25)
    func = lambda t: blocks[0] + tf.eager(t, sigma=sigma) * (1 - sum(blocks))
    strat = pb.SinesBlocks.from_func(func, N=N, lambd=lambd)

    if DEBUG:
        print(strat)
        x_values = np.linspace(0, 1, PLOT_POINTS+1)
        f_values = [func(x) for x in x_values]
        strat_values = [strat.calc(x) for x in x_values]
        plt.plot(x_values, f_values, label='Function')
        plt.plot(x_values, strat_values, label='SinesBlocks')
        plt.legend()
        plt.show()

    for t in [0, 0.25, 0.5, 0.75, 1]:
        p = strat.calc(t)
        exp_p = func(t) if t not in {0, 1} else t
        assert np.isclose(p, exp_p, atol=1e-3), f"Price is incorrect for t={t}, exp={exp_p}, got={p}"


def test_from_risk_averse():
    # Test a risk-averse function
    """ a0, a1 are the function values at t=0 (after the block) and t=1 (before the block) """
    N = 20
    lambd = 1
    func = tf.risk_averse
    sigma = 0.1
    strat = pb.SinesBlocks.from_func(func, N=N, lambd=lambd, params={'sigma': sigma})

    if DEBUG:
        print(strat)
        x_values = np.linspace(0, 1, PLOT_POINTS+1)
        f_values = [func(x, sigma=sigma) for x in x_values]
        strat_values = [strat.calc(x) for x in x_values]
        plt.plot(x_values, f_values, label='Function')
        plt.plot(x_values, strat_values, label='SinesBlocks')
        plt.legend()
        plt.show()

    for t in [0, 0.25, 0.5, 0.75, 1]:
        p = strat.calc(t)
        exp_p = func(t, sigma=sigma) if t not in {0, 1} else t
        assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect for t={t}, exp={exp_p}, got={p}"


def test_from_risk_averse_with_blocks():
    # Test a risk-averse function
    """ a0, a1 are the function values at t=0 (after the block) and t=1 (before the block) """
    N = 50
    lambd = 1
    sigma = 10
    blocks = (0.25, 0.25)
    func = lambda t: blocks[0] + tf.risk_averse(t, sigma=sigma) * (1 - sum(blocks))
    strat = pb.SinesBlocks.from_func(func, N=N, lambd=lambd)

    if DEBUG:
        print(strat)
        x_values = np.linspace(0, 1, PLOT_POINTS+1)
        f_values = [func(x) for x in x_values]
        strat_values = [strat.calc(x) for x in x_values]
        plt.plot(x_values, f_values, label='Function')
        plt.plot(x_values, strat_values, label='SinesBlocks')
        plt.legend()
        plt.show()

    for t in [0, 0.25, 0.5, 0.75, 1]:
        p = strat.calc(t)
        exp_p = func(t) if t not in {0, 1} else t
        assert np.isclose(p, exp_p, atol=1e-3), f"Price is incorrect for t={t}, exp={exp_p}, got={p}"

