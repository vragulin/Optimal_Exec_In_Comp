"""  Tests for the price functions in the lin_prop_blocks module.
    V. Ragulin, 10/20/2024
"""
import os
import sys
import pytest as pt
import numpy as np
from numpy.random import randn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'matrix_utils')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'lin_prop_blocks')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'lin_prop_blocks/K_trader')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'cost_function'))
])

from prop_blocks import SinesBlocks
from cost_model_K import CostModelK, Group
import propagator as pr

# import ow_equilibrium as ow

# Constants
DEBUG = False
N = 2
RHO = 10
A, B = range(2)


@pt.mark.parametrize("a, b, t, lambd", [
    ((0.0, 0.0), (0.0, 0.0), 0.4, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.0, 0),
    ((0.0, 0.0), (0.0, 0.0), 1.0, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.9, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.4, 1),
    ((0.0, 0.0), (0.0, 0.0), 0.8, 10),
    ((2.0, 0.0), (0.0, 0.0), 0.4, 10),
    ((0.0, 2.0), (0.0, 0.0), 0.4, 10),
    ((0.0, 0.0), (2.0, 0.0), 0.6, 0.1),
    ((0.0, 0.0), (0.0, 2.0), 0.6, 0.1),
    ((1.0, -1.0), (0.0, 2.0), 0.6, 0.1),
    ((0.4, 0.1), (-0.3, 2.0), 0.9, 0.1),
])
def test_no_blocks_2terms(a, b, t, lambd):
    # Test if we match without blocks
    names = ['A', 'B']
    strats = [
        SinesBlocks(N, blocks=(0, 0), coeff=np.array(a)),
        SinesBlocks(N, blocks=(0, 0), coeff=np.array(b), lambd=lambd)
    ]
    ntraders = [1, 2]
    groups = [Group(names[i], strats[i], ntraders[i]) for i in range(len(names))]
    c = CostModelK(groups, RHO)
    p = c.price(t)
    exp_p = pr.prop_price_impact_approx(t, np.array(a), np.array(b), lambd * 2, RHO)
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"


# @pt.mark.skip("Working ok")
def test_no_blocks_Nterms():
    # Test if we match without blocks
    t = 0.4
    n_b = 3
    lambd = 5
    np.random.seed(42)
    names = ['A', 'B']
    strats = [
        SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5),
        SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd),
    ]
    ntraders = [1, n_b]
    groups = [Group(names[i], strats[i], ntraders[i]) for i in range(len(names))]
    c = CostModelK(groups, RHO)
    p = c.price(t)
    exp_p = pr.prop_price_impact_approx(t, strats[A].coeff, strats[B].coeff, lambd * n_b, RHO)
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"


# @pt.mark.skip("Working ok")
@pt.mark.parametrize("a, b, t, lambd", [
    ((0.0, 0.0), (0.0, 0.0), 0.4, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.0, 0),
    ((0.0, 0.0), (0.0, 0.0), 1.0, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.9, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.4, 1),
    ((0.0, 0.0), (0.0, 0.0), 0.8, 10),
    ((2.0, 0.0), (0.0, 0.0), 0.4, 10),
    ((0.0, 1.4), (0.0, 0.0), 0.4, 10),
    ((2.0, 0.3), (-0.6, 1.2), 0.2, 0.5),

])
def test_no_sines_2terms(a, b, t, lambd):
    # Test if we match without blocks
    n_a = 2

    names = ['A', 'B']
    strats = [
        SinesBlocks(N, blocks=(0, 0), coeff=np.zeros(N)),
        SinesBlocks(N, blocks=(0, 0), coeff=np.zeros(N) / 5, lambd=lambd),
    ]
    ntraders = [n_a, 1]
    groups = [Group(names[i], strats[i], ntraders[i]) for i in range(len(names))]
    c = CostModelK(groups, RHO)
    p = c.price(t)

    # Calculate expected value
    exp_p = c.displacement_blocks(t)
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"


# @pt.mark.skip("Working ok")
@pt.mark.parametrize("a_blocks, b_blocks, a_n, b_n, t, lambd", [
    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0.4, 0),
    ((1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0.4, 0),
    ((0.0, -0.5), (0.0, 0.0), (0.0, 0.0), (0.0, -2.0), 0.4, 0),
    ((1.0, -0.5), (0.7, -0.3), (1.1, 0.1), (0.0, -2.0), 0.4, 10),
    ((0.0, -0.5), (0.0, 0.0), (0.0, 0.0), (0.0, -2.0), 0.4, 0.1),
    ((0.2, -0.5), (0.1, 0.3), (-0.2, -0.3), (0.1, -2.0), 0.4, 2),
])
def test_2terms(a_blocks, b_blocks, a_n, b_n, t, lambd):
    # Test if we match without blocks
    n_a = 2
    rho = RHO
    names = ['A', 'B']
    strats = [
        SinesBlocks(N, blocks=a_blocks, coeff=np.array(a_n)),
        SinesBlocks(N, blocks=b_blocks, coeff=np.array(b_n)),
    ]
    ntraders = [n_a, 1]
    groups = [Group(names[i], strats[i], ntraders[i]) for i in range(len(names))]
    c = CostModelK(groups, RHO)
    p = c.price(t)

    # Calculate expected value
    d_blocks = c.displacement_blocks(t)
    d_sines = pr.prop_price_impact_approx(t, strats[B].coeff, strats[A].coeff, 2, rho)
    slope_adj = (1 - np.exp(-rho * t)) / rho * c.mkt.lambd  # adj for double-counting a(t)=t slope
    exp_p = d_blocks + d_sines - slope_adj
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"
