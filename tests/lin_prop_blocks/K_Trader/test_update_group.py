""" Testing the update_group method of the CostModelK class
    V. Ragulin - 11/01/2024
"""
import os
import sys
import pytest as pt
import numpy as np

# Add necessary paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../../..', 'lin_prop_blocks'),
    os.path.join(CURRENT_DIR, '../../..', 'lin_prop_blocks/K_trader'),
])
from cost_model_K import CostModelK, Group
from prop_blocks import SinesBlocks

# Constants
N = 5

@pt.fixture
def base_model() -> CostModelK:
    rho = 10
    names = ["Tom", "Dick", "Harry"]
    strats = [
        SinesBlocks(N, blocks=(0, 0), coeff=np.arange(N)),
        SinesBlocks(N, blocks=(1, 0), coeff=np.arange(N) * 0.1, lambd=2),
        SinesBlocks(N, blocks=(1, 0), coeff=np.arange(N) * 10, lambd=3)
    ]
    ntraders = [1, 2, 3]
    groups = [Group(names[i], strats[i], ntraders[i]) for i in range(3)]
    return CostModelK(groups, rho)


def test_update_blocks(base_model):
    # Test if we can update the blocks
    new_blocks = (-0.2, -0.3)

    # Update the blocks for different groups
    for i in range(base_model.K):
        c = base_model
        c.update_group(group=i, blocks=new_blocks)
        assert c.groups[i].strat.blocks == new_blocks, (
            f"Blocks are incorrect, exp={new_blocks}, got={c.groups[i].strat.blocks}")


def test_update_coeffs(base_model):
    # Test if we can update the coeffs
    np.random.seed(42)
    new_coeff = -np.arange(N) * np.pi

    # Update the blocks for different traders
    for i in range(base_model.K):
        c = base_model
        c.update_group(group=i, coeff=new_coeff)
        coeff = c.groups[i].strat.coeff
        np.testing.assert_allclose(coeff, new_coeff, atol=1e-6,
                                   err_msg=f"Coeffs are incorrect, exp={new_coeff}, got={coeff}")


def test_update_lambda(base_model):
    # Test if we can update the lambda

    new_lambd = 5

    # Update the blocks for different traders
    for i in range(base_model.K):
        c = base_model
        c.update_group(group=i, lambd=new_lambd)
        lambd = c.groups[i].strat.lambd
        assert lambd == new_lambd, f"Lambda is incorrect, exp={new_lambd}, got={lambd}"


def test_update_ntraders(base_model):
    # Test if we can update the number of traders in a group
    new_ntraders = 25

    for i in range(base_model.K):
        c = base_model
        c.update_group(group=i, ntraders=new_ntraders)
        ntraders = c.groups[i].ntraders
        assert ntraders == new_ntraders, (f"ntraders is incorrect, "
                                          f"exp={new_ntraders}, got={ntraders}")


def test_update_names(base_model):
    # Test if we can update the lambda
    new_name = "New Name"

    for i in range(base_model.K):
        c = base_model
        c.update_group(group=i, name=new_name)
        name = c.groups[i].name
        assert name == new_name, f"Name is incorrect, exp={new_name}, got={name}"
