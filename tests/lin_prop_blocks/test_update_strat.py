""" Testing the update_strat method of the PropBlocks class
    V. Ragulin - 10/22/2024
"""
import os
import sys
import pytest as pt
import numpy as np

# Add necessary paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../..', 'lin_prop_blocks'),
])

import prop_blocks as pb


def test_update_blocks():
    # Test if we can update the blocks
    N = 10
    np.random.seed(42)
    strat_a = pb.SinesBlocks(N, blocks=(0.1, 0.2), coeff=np.arange(N))
    strat_b = pb.SinesBlocks(N, blocks=(0.3, 0.4), coeff=np.arange(N) + 10, lambd=10)
    new_blocks = (-0.2, -0.3)

    # Update the blocks for trader A
    c = pb.CostModel([strat_a, strat_b], rho=5)
    c.update_strat(blocks=new_blocks)
    assert c.strats[c.A].blocks == new_blocks, (f"Blocks are incorrect, "
                                                f"exp={new_blocks}, got={c.strats[c.A].blocks}")

    # Update the blocks for trader B
    c = pb.CostModel([strat_a, strat_b], rho=5)
    c.update_strat(trader=c.B, blocks=new_blocks)
    assert c.strats[c.B].blocks == new_blocks, (f"Blocks are incorrect, "
                                                f"exp={new_blocks}, got={c.strats[c.B].blocks}")


def test_update_coeffs():
    # Test if we can update the blocks
    N = 10
    np.random.seed(42)
    strat_a = pb.SinesBlocks(N, blocks=(0.1, 0.2), coeff=np.arange(N))
    strat_b = pb.SinesBlocks(N, blocks=(0.3, 0.4), coeff=np.arange(N) + 10, lambd=10)
    new_coeff = -np.arange(N) * np.pi

    # Update the blocks for trader A
    c = pb.CostModel([strat_a, strat_b], rho=5)
    c.update_strat(coeff=new_coeff)
    np.testing.assert_allclose(c.strats[c.A].coeff, new_coeff, atol=1e-6,
                               err_msg=f"Coeffs are incorrect, exp={new_coeff}, got={c.strats[c.A].coeff}")

    # Update the blocks for trader B
    c = pb.CostModel([strat_a, strat_b], rho=5)
    c.update_strat(trader=c.B, coeff=new_coeff)
    np.testing.assert_allclose(c.strats[c.B].coeff, new_coeff, atol=1e-6,
                               err_msg=f"Coeffs are incorrect, exp={new_coeff}, got={c.strats[c.B].coeff}")


def test_update_lambda():
    # Test if we can update the blocks
    N = 10
    np.random.seed(42)
    strat_a = pb.SinesBlocks(N, blocks=(0.1, 0.2), coeff=np.arange(N))
    strat_b = pb.SinesBlocks(N, blocks=(0.3, 0.4), coeff=np.arange(N) + 10, lambd=10)
    new_lambda = 100

    # Update the blocks for trader A
    c = pb.CostModel([strat_a, strat_b], rho=5)
    c.update_strat(lambd=new_lambda)
    assert c.strats[c.A].lambd == new_lambda, (f"Lambda is incorrect, "
                                               f"exp={new_lambda}, got={c.strats[c.A].lambd}")

    # Update the blocks for trader B
    c = pb.CostModel([strat_a, strat_b], rho=5)
    c.update_strat(trader=c.B, lambd=new_lambda)
    assert c.strats[c.B].lambd == new_lambda, (f"Lambda is incorrect, "
                                               f"exp={new_lambda}, got={c.strats[c.B].lambd}")