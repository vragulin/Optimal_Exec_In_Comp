""" Tests for the cost functions for K trader model in the lin_prop_blocks module.
    V. Ragulin, 10/21/2024
"""

import os
import sys
import pytest as pt
import numpy as np
from numpy.random import randn

from prop_blocks import SinesBlocks

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'matrix_utils')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'lin_prop_blocks')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'lin_prop_blocks/K_trader')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'cost_function'))
])

from cost_model_K import CostModelK, Group
import prop_blocks as pb
import propagator as pr

DEBUG = True


# Test for two traders
def test_create_cost_model():
    # Test if we match without blocks
    rho = 10
    N = 2
    strat_a = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array([1, 2]))
    strat_b = pb.SinesBlocks(N, blocks=(0, 0), coeff=np.array([3, 4]), lambd=2)
    c = CostModelK([Group('A', strat_a), Group('B', strat_b)], rho)
    if DEBUG:
        print(c)
    assert c.K == 2, f"Number of traders is incorrect, exp = 2, got = {c.K}"


def test_with_trader_names():
    # Test if we are able to capture and display trader names
    rho = 10
    N = 2
    names = ["Tom", "Dick", "Harry"]
    strats = [
        pb.SinesBlocks(N, blocks=(0, 0)),
        pb.SinesBlocks(N, blocks=(1, 0), lambd=2),
        pb.SinesBlocks(N, blocks=(1, 0), lambd=3)
    ]
    ntraders = [1, 2, 3]
    groups = [Group(names[i], strats[i], ntraders[i]) for i in range(3)]
    c = CostModelK(groups, rho)
    if DEBUG:
        print(c)
    for i, g in enumerate(c.groups):
        assert g.name == names[i], f"Name is incorrect, exp ={names[i]}, got = {g.name}"
        assert g.ntraders == ntraders[i], f"n_traders is incorrect, exp={ntraders[i]}, got = {g.ntraders}"


def test_with_ntraders():
    # Test if we are able to capture #traders of each type and take that into account when
    #  calculating the market strategy
    rho = 10
    N = 2
    names = ["Tom", "Dick", "Harry"]
    strats = [
                pb.SinesBlocks(N, blocks=(1, 0), coeff=np.ones(N)),
                pb.SinesBlocks(N, blocks=(0.1, 0), coeff=np.ones(N) * 10, lambd=2),
                pb.SinesBlocks(N, blocks=(0, 10), coeff=np.ones(N) * 0.1, lambd=3),
            ]
    ntraders = [1, 2, 3]
    groups = [Group(names[i], strats[i], ntraders[i]) for i in range(3)]
    c = CostModelK(groups, rho)
    if DEBUG:
        print(c)
    exp_mkt_lambd = 14
    assert c.mkt.lambd == exp_mkt_lambd, f"Market lambda is incorrect, exp = {exp_mkt_lambd}, got = {c.mkt.lambd}"
    exp_mkt_blocks = (1.4 / 14.0, 90 / 14.0)
    assert c.mkt.blocks == exp_mkt_blocks, f"Market lambda is incorrect, exp = {exp_mkt_blocks}, got = {c.mkt.blocks}"
    exp_mkt_coeff = np.ones(N) * 41.9 / 14.0
    np.testing.assert_allclose(c.mkt.coeff, exp_mkt_coeff, rtol=1e-6), (f"Coeffs are inocrrect, exp = {exp_mkt_coeff}, "
                                                                        f"got = {c.mkt.coeff}")
