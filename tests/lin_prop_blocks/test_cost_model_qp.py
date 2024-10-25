""" Tests fitting the biliear model to the cost function.
    V. Ragulin, 10/24/2024
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

from prop_blocks import SinesBlocks
from cost_model_qp import CostModelQP

# import propagator as pr

DEBUG = False


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
    strat_a = SinesBlocks(N, blocks=(0, 0), coeff=np.array(a))
    strat_b = SinesBlocks(N, blocks=(0, 0), coeff=np.array(b), lambd=lambd)
    c = CostModelQP([strat_a, strat_b], rho)

    if DEBUG:
        for trader in range(2):
            print(f"Trader {trader}:\n", c.qp_coeffs[trader])

    L = c.cost_trader_matrix(trader=c.A)
    exp_L = c.cost_trader(trader=c.A)
    assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


def test_no_blocks_Nterms():
    # Test if we match without blocks for larger N
    N = 20
    lambd = 0.1
    rho = 10
    np.random.seed(42)
    n_iter = 20

    for _ in range(n_iter):
        for trader in range(2):
            strat_a = SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5)
            strat_b = SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd)
            c = CostModelQP([strat_a, strat_b], rho)
            L = c.cost_trader_matrix(trader=trader, verbose=DEBUG)
            exp_L = c.cost_trader(trader=trader, verbose=DEBUG)
            assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


@pt.mark.parametrize("a_n, b_n, a_block, b_block, lambd, rho", [
    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 100, 1),
    ((1.0, 0.0), (-0.4, 0.3), (0.0, 0.0), (0.0, 0.0), 2, 10),
    ((1.0, 0.0), (-0.4, 0.3), (1.0, 0.0), (0.3, -0.3), 2, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 0, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 1, 0.1),
])
def test_cost_v_matrix(a_n, b_n, a_block, b_block, lambd, rho):
    N = len(a_n)
    strat_a = SinesBlocks(N, blocks=a_block, coeff=np.array(a_n))
    strat_b = SinesBlocks(N, blocks=b_block, coeff=np.array(b_n), lambd=lambd)
    c = CostModelQP([strat_a, strat_b], rho=10)
    for trader in range(2):
        L = c.cost_trader_matrix(trader=trader, verbose=DEBUG)
        exp_L = c.cost_trader(trader=trader, verbose=DEBUG)
        assert np.isclose(L, exp_L, atol=1e-6), f"Cost is incorrect, exp={exp_L}, got={L}"


@pt.mark.parametrize("a_n, b_n, a_block, b_block, lambd, rho", [
    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 100, 1),
    ((1.0, 0.0), (-0.4, 0.3), (0.0, 0.0), (0.0, 0.0), 2, 10),
    ((1.0, 0.0), (-0.4, 0.3), (1.0, 0.0), (0.3, -0.3), 2, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 0, 10),
    ((-0.3, -1.1), (0.5, 0.3), (1.0, -2.0), (0.3, -0.3), 1, 0.1),
])
def test_cost_matrix_valid(a_n, b_n, a_block, b_block, lambd, rho):
    N = len(a_n)
    strat_a = SinesBlocks(N, blocks=a_block, coeff=np.array(a_n))
    strat_b = SinesBlocks(N, blocks=b_block, coeff=np.array(b_n), lambd=lambd)
    c = CostModelQP([strat_a, strat_b], rho=10)
    for trader in range(2):
        H = c.qp_coeffs[trader]["H"]
        assert np.allclose(H, H.T), "Hessian is not symmetric"

        # Check that there are no quadratic terms for the other trader
        other = 1 - trader
        other_submat = H[other * (N + 2):(other + 1) * (N + 2), other * (N + 2): (other + 1) * (N + 2)]
        assert np.allclose(other_submat, 0), "Hessian contains quadratic terms for the other trader"


@pt.mark.parametrize("N, lambd, rho", [
    (2, 1, 10),
    (5, 0.1, 10),
    (10, 0.1, 10),
    (20, 0.1, 10),
    # (50, 10, 0.01),
])
def test_cost_matrix_NTerms(N, lambd, rho):
    # Test if the cost matrix is valid for larger N
    # N = 20
    # lambd = 0.1
    # rho = 10
    np.random.seed(42)
    n_iter = 20

    for _ in range(n_iter):
        for trader in range(2):
            strat_a = SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5)
            strat_b = SinesBlocks(N, blocks=(0, 0), coeff=randn(N) / 5, lambd=lambd)
            c = CostModelQP([strat_a, strat_b], rho)
            H = c.qp_coeffs[trader]["H"]
            assert np.allclose(H, H.T), "Hessian is not symmetric"

            # Check that there are no quadratic terms for the other trader
            other = 1 - trader
            other_submat = H[other * (N + 2):(other + 1) * (N + 2), other * (N + 2): (other + 1) * (N + 2)]
            assert np.allclose(other_submat, 0), "Hessian contains quadratic terms for the other trader"
