""" Scripts to test the derivatives calculation of the cost function
    V. Ragulin, 10/21/2024
"""
import os
import sys
import pytest as pt
import numpy as np
from numpy.random import randn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'lin_prop_blocks')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'cost_function')))

import prop_blocks as pb
import propagator as pr

DEBUG = True


@pt.mark.parametrize("t", [0.0, 0.2, 0, 0.5, 1.0])
def test_func_2terms(t):
    # Test if we match without blocks
    N = 2
    np.random.seed(42)
    for _ in range(10):
        coeff = np.random.rand(N) / 3
        blocks = np.random.rand(2) / 2 - 0.2
        strat = pb.SinesBlocks(N, blocks=(blocks[0], blocks[1]), coeff=coeff)
        f = strat.calc(t)
        match t:
            case 0.0:
                exp_f = 0
            case 1.0:
                exp_f = 1
            case _:
                exp_f = blocks[0] + strat.mu * t + (coeff[0] * np.sin(np.pi * t) + coeff[1] * np.sin(2 * np.pi * t))

        assert np.isclose(f, exp_f, atol=1e-6), f"f(t) is incorrect, exp={exp_f}, got={f}"


@pt.mark.parametrize("t", [0.0, 0.2, 0, 0.5, 1.0])
def test_deriv_2terms(t):
    # Test if we match without blocks
    N = 2
    np.random.seed(42)
    for _ in range(1):
        coeff = np.random.rand(N) / 3
        # coeff = np.array([0.0, 1])
        blocks = np.random.rand(2) / 2 - 0.2
        strat = pb.SinesBlocks(N, blocks=(blocks[0], blocks[1]), coeff=coeff)
        f = strat.deriv(t)
        cos, pi = np.cos, np.pi
        exp_f = strat.mu + pi * (coeff[0] * cos(pi * t) + coeff[1] * 2 * cos(2 * pi * t))

        assert np.isclose(f, exp_f, atol=1e-6), f"f'(t) is incorrect, exp={exp_f}, got={f}"
