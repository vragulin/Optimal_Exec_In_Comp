"""  Tests for the price functions in the lin_prop_blocks module.
    V. Ragulin, 10/20/2024
"""
import os
import sys
import pytest as pt
import numpy as np
from numpy.random import randn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'matrix_utils')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'lin_prop_blocks')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'cost_function')))
# sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'lin_propagator_qp/equilibrium/ow_exact')))

import prop_blocks as pb
import propagator as pr

# import ow_equilibrium as ow

DEBUG = False


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
    rho = 10
    N = 2
    strat_a = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array(a))
    strat_b = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array(b), lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho)
    p = c.price(t)
    exp_p = pr.prop_price_impact_approx(t, np.array(a), np.array(b), lambd, rho)
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"


# @pt.mark.skip("Working ok")
def test_no_blocks_Nterms():
    # Test if we match without blocks
    rho = 10
    N = 10
    lambd = 0.1
    rho = 10
    t = 0.4
    np.random.seed(42)
    strat_a = pb.SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5)
    strat_b = pb.SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho)
    p = c.price(t)
    exp_p = pr.prop_price_impact_approx(t, strat_a.coeff, strat_b.coeff, lambd, rho)
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"


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
    rho = 10
    N = 2
    strat_a = pb.SinesBlocks(N, blocks=a, coeff=np.zeros(N))
    strat_b = pb.SinesBlocks(N, blocks=b, coeff=np.zeros(N), lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho)
    p = c.price(t)

    # Calculate expected value
    exp_p  = c.displacement_blocks(t)
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"


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
    rho = 10
    N = 2
    strat_a = pb.SinesBlocks(N, blocks=a_blocks, coeff=np.array(a_n))
    strat_b = pb.SinesBlocks(N, blocks=b_blocks, coeff=np.array(b_n), lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho)
    p = c.price(t)

    # Calculate expected value
    d_blocks = c.displacement_blocks(t)
    d_sines = pr.prop_price_impact_approx(t, strat_a.coeff, strat_b.coeff, lambd, rho)
    slope_adj = (1 - np.exp(-rho * t)) / rho * c.mkt.lambd  # adj for double-counting a(t)=t slope
    exp_p = d_blocks + d_sines - slope_adj
    assert np.isclose(p, exp_p, atol=1e-6), f"Price is incorrect, exp={exp_p}, got={p}"
