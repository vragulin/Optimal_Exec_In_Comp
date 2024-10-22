""" Tests for the cost functions in the lin_prop_blocks module.
    V. Ragulin, 10/21/2024
"""

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

DEBUG = True


@pt.mark.parametrize("a, b, lambd", [
    ((0.0, 0.0), (0.0, 0.0), 0),
    ((0.0, 0.0), (0.0, 0.0), 1),
    ((0.0, 0.0), (0.0, 0.0), 10),
    ((2.0, 0.0), (0.0, 0.0), 10),
    ((0.0, 2.0), (0.0, 0.0), 10),
    ((0.0, 0.0), (2.0, 0.0), 0.1),
    ((0.0, 0.0), (0.0, 2.0), 0.1),
    ((1.0, -1.0), (0.0, 2.0), 0.1),
    ((0.4, 0.1), (-0.3, 2.0), 0.1),
])
def test_no_blocks_trader_a_2terms(a, b, lambd):
    # Test if we match without blocks
    rho = 10
    N = 2
    strat_a = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array(a))
    strat_b = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array(b), lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho)
    L = c.cost_trader(verbose=DEBUG)
    exp_L = pr.cost_fn_prop_a_approx(np.array(a), np.array(b), lambd, rho, verbose=DEBUG)
    assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


# @pt.mark.skip("Working ok")
def test_no_blocks_Nterms():
    # Test if we match without blocks for larger N
    rho = 10
    N = 20
    lambd = 0.1
    rho = 10
    np.random.seed(42)
    n_iter = 20

    for _ in range(n_iter):
        strat_a = pb.SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5)
        strat_b = pb.SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd)
        c = pb.CostModel([strat_a, strat_b], rho)
        L = c.cost_trader(verbose=DEBUG)
        exp_L = pr.cost_fn_prop_a_approx(strat_a.coeff, strat_b.coeff, lambd, rho, verbose=DEBUG)
        assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


@pt.mark.parametrize("a, b, lambd", [
    ((0.0, 0.0), (0.0, 0.0), 0),
    ((0.0, 0.0), (0.0, 0.0), 1),
    ((0.0, 0.0), (0.0, 0.0), 10),
    ((2.0, 0.0), (0.0, 0.0), 10),
    ((0.0, 2.0), (0.0, 0.0), 10),
    ((0.0, 0.0), (2.0, 0.0), 0.1),
    ((0.0, 0.0), (0.0, 2.0), 0.1),
    ((1.0, -1.0), (0.0, 2.0), 0.1),
    ((0.4, 0.1), (-0.3, 2.0), 0.1),
])
def test_no_blocks_trader_b_2terms(a, b, lambd):
    # Test if we match without blocks
    rho = 10
    N = 2
    strat_a = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array(a))
    strat_b = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array(b), lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho)
    L = c.cost_trader(trader=c.B, verbose=DEBUG)
    exp_L = pr.cost_fn_prop_b_approx(np.array(a), np.array(b), lambd, rho, verbose=DEBUG) if lambd != 0 else 0
    assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


def test_no_blocks_trader_b_Nterms():
    # Test if we match without blocks for larger N
    rho = 10
    N = 20
    lambd = 0.1
    rho = 10
    np.random.seed(42)
    n_iter = 20

    for _ in range(n_iter):
        strat_a = pb.SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5)
        strat_b = pb.SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd)
        c = pb.CostModel([strat_a, strat_b], rho)
        L = c.cost_trader(verbose=DEBUG)
        exp_L = pr.cost_fn_prop_a_approx(strat_a.coeff, strat_b.coeff, lambd, rho, verbose=DEBUG) if lambd != 0 else 0
        assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


# @pt.mark.skip("Not implemented")
@pt.mark.parametrize("a, b, t, lambd", [
    ((0.0, 0.0), (0.0, 0.0), 0.4, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.0, 0),
    ((0.0, 0.0), (0.0, 0.0), 1.0, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.9, 0),
    ((0.0, 0.0), (0.0, 0.0), 0.4, 1),
    ((0.0, 0.0), (0.0, 0.0), 0.8, 10),
    ((2.0, 0.0), (0.0, 0.0), 0.4, 10),
    ((0.0, 1.4), (0.0, 0.0), 0.4, 100),
    ((2.0, 0.3), (-0.6, 1.2), 0.2, 0.5),
])
def test_no_sines_a(a, b, t, lambd):
    # Test if we match without blocks
    rho = 10
    N = 2
    strat_a = pb.SinesBlocks(N, blocks=a, coeff=np.zeros(N))
    strat_b = pb.SinesBlocks(N, blocks=b, coeff=np.zeros(N), lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho)
    L = c.cost_trader(verbose=DEBUG)

    # Calculate expected value, simple, since we only have 2 trades
    # Formula from the ow_exact/ow_equilibrium.py cost_a()
    trd, mkt, exp = c.strats[c.A], c.mkt, np.exp
    K = (rho + exp(-rho) -1) / rho**2
    J = (1 - exp(-rho)) / rho
    term1 = 0.5 * (mkt.blocks[0] * trd.blocks[0] + mkt.blocks[1] * trd.blocks[1])
    term2 = trd.mu * (mkt.blocks[0] * J + mkt.mu * K)
    term3 = (mkt.blocks[0] * exp(-rho) + mkt.mu * J) * trd.blocks[1]
    L_exp = (term1 + term2 + term3) * trd.lambd * mkt.lambd
    assert np.isclose(L, L_exp, atol=1e-6), f"Price is incorrect, exp={L_exp}, got={L}"


@pt.mark.parametrize("a_n, b_n, a_block, b_block, lambd, rho", [
    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 100, 1),
    ((1.0, 0.0), (-0.4, 0.3), (0.0, 0.0), (0.0, 0.0), 2, 10),
    ((1.0, 0.0), (-0.4, 0.3), (1.0, 0.0), (0.3, -0.3), 2, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 0, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 1, 0.1),
])
def test_cost_v_integral(a_n, b_n, a_block, b_block, lambd, rho):
    # Test if we match without blocks
    N = len(a_n)
    strat_a = pb.SinesBlocks(N, blocks=a_block, coeff=np.array(a_n))
    strat_b = pb.SinesBlocks(N, blocks=b_block, coeff=np.array(b_n), lambd=lambd)
    c = pb.CostModel([strat_a, strat_b], rho=10)
    L = c.cost_trader(verbose=DEBUG)
    L_int  = c.cost_trader_integral(verbose=DEBUG)
    assert np.isclose(L, L_int, atol=0.001), f"Different Results, exact={L}, int={L_int}"