""" Test for the function update_subset_of_groups in the class CostModelK
    that updates the strategies of a subset of traders in each group (usually just 1)
    V. Ragulin 11/03/2024
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
DEBUG = True
N = 3


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


def test_update_one_trader_group_0(base_model):
    var_groups = []
    b = base_model
    var_blocks, var_sines = (-1, -2), np.ones(N) * 0.7
    for i, s in enumerate(b.groups):
        if i == 0:
            var_strat = SinesBlocks(b.N, blocks=var_blocks, coeff=var_sines, lambd=b.groups[i].strat.lambd)
            var_group = Group(name=b.groups[i].name, strat=var_strat, ntraders=1)
            var_groups.append(var_group)
        else:
            var_groups.append(None)

    new_model = b.update_subset_of_groups(var_groups=var_groups)
    if DEBUG:
        print("\nBase Model:")
        print(base_model)
        print("\nNew Model:")
        print(new_model)

    assert len(new_model.groups) == 3
    assert new_model.groups[0].strat.blocks == var_blocks, (
        f"Blocks are incorrect, exp={var_blocks}, got={new_model.groups[0].strat.blocks}")
    np.testing.assert_allclose(new_model.groups[0].strat.coeff, var_sines), (
        f"Coeffs are incorrect, exp={var_sines}, got={new_model.groups[0].strat.coeff}")


@pt.mark.parametrize("group", [1, 2])
def test_update_one_trader_groups_1_2(base_model, group):
    var_groups = []
    b = base_model
    var_blocks, var_sines = (-1, -2), np.ones(N) * 0.7
    for i, s in enumerate(b.groups):
        if i == group:
            var_strat = SinesBlocks(b.N, blocks=var_blocks, coeff=var_sines, lambd=b.groups[i].strat.lambd)
            new_group = Group(name=b.groups[i].name, strat=var_strat, ntraders=1)
            var_groups.append(new_group)
        else:
            var_groups.append(None)

    new_model = b.update_subset_of_groups(var_groups=var_groups)
    if DEBUG:
        print("\nBase Model:")
        print(base_model)
        print("\nNew Model:")
        print(new_model)
        print("Variation model mapping: ", end="")
        print(new_model.groups_idx)
    idx_base, idx_var = new_model.groups_idx[group]['base'], new_model.groups_idx[group]['variation']
    assert len(new_model.groups) == 4

    # Check that the variation group has been set up correctly
    assert new_model.groups[idx_var].strat.blocks == var_blocks, (
        f"Blocks are incorrect, exp={var_blocks}, got={new_model.groups[group].strat.blocks}")
    assert new_model.groups[idx_var].ntraders == 1, (
        f"ntraders is incorrect, exp={1}, got={new_model.groups[group].ntraders}")
    np.testing.assert_allclose(new_model.groups[idx_var].strat.coeff, var_sines), (
        f"Coeffs are incorrect, exp={var_sines}, got={new_model.groups[idx_var].strat.coeff}")

    # Check that the base group has been set up correctly
    assert (v1 := new_model.groups[idx_base].strat.blocks) == (v2 := b.groups[group].strat.blocks), (
        f"Blocks are incorrect, exp={v2}, got={v1}")
    assert (v1 := new_model.groups[idx_base].ntraders) == (v2 := b.groups[group].ntraders - 1), (
        f"ntraders is incorrect, exp={v2}, got={v1}")
    np.testing.assert_allclose((v1 := new_model.groups[idx_base].strat.coeff), (v2 := b.groups[group].strat.coeff)), (
        f"Coeffs are incorrect, exp={v1}, got={v2}")


@pt.mark.parametrize("change_groups", [
    (0, 1), (0, 2)
])
def test_update_one_trader_two_groups(base_model, change_groups):
    """ Updates groups 0,1 at the same time """
    var_groups = []
    b = base_model
    var_blocks, var_sines = (-1, -2), np.ones(N) * 0.7
    for i, s in enumerate(b.groups):
        if i in change_groups:
            var_strat = SinesBlocks(b.N, blocks=var_blocks, coeff=var_sines, lambd=b.groups[i].strat.lambd)
            new_group = Group(name=b.groups[i].name, strat=var_strat, ntraders=1)
            var_groups.append(new_group)
        else:
            var_groups.append(None)

    new_model = b.update_subset_of_groups(var_groups=var_groups)
    if DEBUG:
        print("\nBase Model:")
        print(base_model)
        print("\nNew Model:")
        print(new_model)
        print("Variation model mapping: ", end="")
        print(new_model.groups_idx)

    assert len(new_model.groups) == 4

    # Group 0
    # Check that the variation group has been set up correctly
    group, idx_var = 0, 0
    assert new_model.groups[idx_var].strat.blocks == var_blocks, (
        f"Blocks are incorrect, exp={var_blocks}, got={new_model.groups[group].strat.blocks}")
    assert new_model.groups[idx_var].ntraders == 1, (
        f"ntraders is incorrect, exp={1}, got={new_model.groups[group].ntraders}")
    np.testing.assert_allclose(new_model.groups[idx_var].strat.coeff, var_sines), (
        f"Coeffs are incorrect, exp={var_sines}, got={new_model.groups[idx_var].strat.coeff}")

    # Group 1
    # Check that the variation group has been set up correctly
    group = change_groups[1]
    idx_base, idx_var = new_model.groups_idx[group]['base'], new_model.groups_idx[group]['variation']
    assert new_model.groups[idx_var].strat.blocks == var_blocks, (
        f"Blocks are incorrect, exp={var_blocks}, got={new_model.groups[group].strat.blocks}")
    assert new_model.groups[idx_var].ntraders == 1, (
        f"ntraders is incorrect, exp={1}, got={new_model.groups[group].ntraders}")
    np.testing.assert_allclose(new_model.groups[idx_var].strat.coeff, var_sines), (
        f"Coeffs are incorrect, exp={var_sines}, got={new_model.groups[idx_var].strat.coeff}")

    # Check that the base group has been set up correctly
    assert (v1 := new_model.groups[idx_base].strat.blocks) == (v2 := b.groups[group].strat.blocks), (
        f"Blocks are incorrect, exp={v2}, got={v1}")
    assert (v1 := new_model.groups[idx_base].ntraders) == (v2 := b.groups[group].ntraders - 1), (
        f"ntraders is incorrect, exp={v2}, got={v1}")
    np.testing.assert_allclose((v1 := new_model.groups[idx_base].strat.coeff), (v2 := b.groups[group].strat.coeff)), (
        f"Coeffs are incorrect, exp={v1}, got={v2}")
