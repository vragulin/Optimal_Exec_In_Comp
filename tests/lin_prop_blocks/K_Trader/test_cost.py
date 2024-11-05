""" Tests for the cost functions in the lin_prop_blocks K-trader module.
    V. Ragulin, 10/21/2024
"""

import os
import sys
import pytest as pt
import numpy as np
from numpy.random import randn
from copy import deepcopy

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

# Constants
DEBUG = False
A, B = range(2)
NAMES = ['A', 'B']
RHO = 10
N = 2


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
    n_b = 2
    strats = [
        SinesBlocks(N, blocks=(0, 0), coeff=np.array(a)),
        SinesBlocks(N, blocks=(0, 0), coeff=np.array(b), lambd=lambd)
    ]
    ntraders = [1, n_b]
    groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
    c = CostModelK(groups, RHO)
    if DEBUG:
        print(c)
    L = c.cost_trader(verbose=DEBUG)
    exp_L = pr.cost_fn_prop_a_approx(np.array(a), np.array(b), lambd * 2, RHO, verbose=DEBUG)
    assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


# @pt.mark.skip("Working ok")
def test_no_blocks_Nterms():
    # Test if we match without blocks for larger N
    N = 20
    lambd = 0.1
    n_b = 4
    ntraders = [1, n_b]

    np.random.seed(42)
    n_iter = 20
    for _ in range(n_iter):
        strats = [
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5),
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd)
        ]
        groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
        c = CostModelK(groups, RHO)
        if DEBUG:
            print(c)
        L = c.cost_trader(verbose=DEBUG)
        exp_L = pr.cost_fn_prop_a_approx(strats[A].coeff, strats[B].coeff, lambd * n_b, RHO, verbose=DEBUG)
        assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


def test_no_blocks_Nterms_several_A():
    # Test if we match without blocks for larger N when there are several A traders
    N = 20
    lambd = 0.1
    np.random.seed(42)
    n_iter = 20
    n_a = 4
    ntraders = [n_a, 1]

    for _ in range(n_iter):
        strats = [
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd),
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5),
        ]
        groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
        c = CostModelK(groups, RHO)
        if DEBUG:
            print(c)
        L = c.cost_trader(verbose=DEBUG)
        exp_L = pr.cost_fn_prop_b_approx(strats[B].coeff, strats[A].coeff, lambd * n_a, RHO, verbose=DEBUG) / n_a
        assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


def test_no_blocks_Nterms_set_v_list():
    # Test if representing N equal traders as an n_traders list works, vs enuerating them
    N = 20
    lambd = 0.1
    rho = 10
    np.random.seed(42)
    n_iter = 20
    n_A, n_B = 4, 3
    ntraders = [n_A, n_B]
    for _ in range(n_iter):
        strats = [
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd),
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5),
        ]
        groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
        c= CostModelK(groups, RHO)
        if DEBUG:
            print(c)
        L = c.cost_trader(verbose=DEBUG)

        # Representing N equal traders as list
        names1 = ['A'] * n_A + ['B'] * n_B
        strats1 = [strats[A]] * ntraders[A] + [strats[B]] * ntraders[B]
        ntraders1 = list(np.ones(n_A + n_B, dtype=int))
        groups1 = [Group(names1[i], strats1[i], ntraders1[i]) for i in range(len(names1))]
        c1 = CostModelK(groups1, RHO)
        if DEBUG:
            print(c1)
        L1 = c1.cost_trader(verbose=DEBUG)

        assert np.isclose(L, L1, atol=1e-6), f"Cost is incorrect, with mult={L}, with list={L1}"


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
    n_b = 2
    ntraders = [1, n_b]
    strats = [
        SinesBlocks(N, blocks=(0, 0), coeff=np.array(a)),
        SinesBlocks(N, blocks=(0, 0), coeff=np.array(b), lambd=lambd)
    ]
    groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
    c = CostModelK(groups, RHO)
    if DEBUG:
        print(c)
    L = c.cost_trader(group=B, verbose=DEBUG)
    exp_L = pr.cost_fn_prop_b_approx(np.array(a), np.array(b), lambd * n_b, RHO,
                                     verbose=DEBUG) / n_b if lambd != 0 else 0
    assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


def test_no_blocks_trader_b_Nterms():
    # Test if we match without blocks for larger N
    N = 20
    lambd = 0.1
    rho = 10
    np.random.seed(42)
    n_iter = 20
    n_b = 3
    ntraders = [1, n_b]

    for _ in range(n_iter):
        strats = [
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5),
            SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd),
        ]
        groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
        c = CostModelK(groups, RHO)
        if DEBUG:
            print(c)
        L = c.cost_trader(group=B, verbose=DEBUG)

        exp_L = pr.cost_fn_prop_b_approx(strats[A].coeff, strats[B].coeff, lambd * n_b, rho,
                                         verbose=DEBUG) / n_b if lambd != 0 else 0
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

    n_b = 2
    rho = RHO
    ntraders = [1, n_b]
    strats = [
        SinesBlocks(N, blocks=a, coeff=np.zeros(N)),
        SinesBlocks(N, blocks=b, coeff=np.zeros(N), lambd=lambd),
    ]
    groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
    c = CostModelK(groups, rho)
    if DEBUG:
        print(c)
    L = c.cost_trader(group=A, verbose=DEBUG)

    # Calculate expected value, simple, since we only have 2 trades
    # Formula from the ow_exact/ow_equilibrium.py cost_a()
    trd, mkt, exp = c.groups[A].strat, c.mkt, np.exp
    K = (rho + exp(-rho) - 1) / rho ** 2
    J = (1 - exp(-rho)) / rho
    term1 = 0.5 * (mkt.blocks[0] * trd.blocks[0] + mkt.blocks[1] * trd.blocks[1])
    term2 = trd.mu * (mkt.blocks[0] * J + mkt.mu * K)
    term3 = (mkt.blocks[0] * exp(-rho) + mkt.mu * J) * trd.blocks[1]
    L_exp = (term1 + term2 + term3) * trd.lambd * mkt.lambd
    assert np.isclose(L, L_exp, atol=1e-6), f"Implementation Cost is incorrect, exp={L_exp}, got={L}"


@pt.mark.parametrize("a_n, b_n, a_block, b_block, lambd, rho", [
    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 100, 1),
    ((1.0, 0.0), (-0.4, 0.3), (0.0, 0.0), (0.0, 0.0), 2, 10),
    ((1.0, 0.0), (-0.4, 0.3), (1.0, 0.0), (0.3, -0.3), 2, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 0, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 1, 0.1),
])
def test_cost_v_integral(a_n, b_n, a_block, b_block, lambd, rho):
    N = len(a_n)
    n_b = 3
    ntraders = [1, n_b]
    strats = [
        SinesBlocks(N, blocks=a_block, coeff=np.array(a_n)),
        SinesBlocks(N, blocks=b_block, coeff=np.array(b_n), lambd=lambd),
    ]
    groups = [Group(NAMES[i], strats[i], ntraders[i]) for i in range(len(NAMES))]
    c = CostModelK(groups, rho)
    if DEBUG:
        print(c)
    L = c.cost_trader(verbose=DEBUG)

    strat_b = SinesBlocks(N, blocks=b_block, coeff=np.array(b_n), lambd=lambd * n_b)
    strats[B] = strat_b
    groups = [Group(NAMES[i], strats[i], 1) for i in range(len(NAMES))]
    c1 = CostModelK(groups, rho)
    L_int = c1.cost_trader_integral(verbose=DEBUG)
    assert np.isclose(L, L_int, atol=0.001), f"Different Results, exact={L}, int={L_int}"
