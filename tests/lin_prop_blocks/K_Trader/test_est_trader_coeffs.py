""" Test for estimating the trader cost quadratic form coefficients
V. Ragulin, 11/3/2024
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
from propagator import cost_fn_prop_a_matrix

# Constants
DEBUG = True


def pad_array(arr, k):
    if len(arr) != 2:
        raise ValueError("Input array must be of size 2")
    if k < 2:
        raise ValueError("Target size k must be at least 2")

    pad_width = k - len(arr)
    padded_arr = np.pad(arr, (0, pad_width), mode='edge')
    return padded_arr


def pad_list(in_list:list, k) -> list:
    if k < 2:
        raise ValueError("Target size k must be at least 2")

    padded_list = in_list.copy()
    n_to_add = k - len(in_list)
    if n_to_add > 0:
        padded_list += [padded_list[-1]] * n_to_add

    return padded_list


@pt.mark.parametrize("blocks_arr, lambd_arr, ntraders, group", [
    ([(0, 0), (0, 0)], [1, 1], 1, 0),
    ([(0, 0), (0, 0)], [1, 2], 1, 0),
    ([(0, 0), (0, 0)], [1, 2], 2, 1),
    ([(1, 0), (0, 1)], [1, 1], 2, 0),
    ([(1, 0), (0, 1)], [1, 2], 1, 1),
    ([(1, 2), (3, -1)], [1, 2], 3, 1),
])
def test_two_trader_blocks_only(blocks_arr, lambd_arr, ntraders, group):
    rho = 10
    N = 2

    # Create the model
    strats = []
    for i in range(2):
        strats.append(SinesBlocks(N, blocks=blocks_arr[i], coeff=np.zeros(N), lambd=lambd_arr[i]))
    groups = [Group(f"Trader_{i}", strats[i], ntraders) for i in range(2)]
    c = CostModelK(groups, rho)
    res = c.est_trader_coeffs(group=group, use_sines=False)
    if DEBUG:
        for k, v in res.items():
            print(f"{k}:")
            print(f"{v}\n")

    # Check the results
    cost_from_func = c.cost_trader(group=group)
    H, f, C, var_model = res['H'], res['f'], res['C'], c.var_model
    x_list = []
    for i, g in enumerate(c.groups):
        if isinstance(var_model.groups_idx[i], dict):
            x_list += [np.array(g.strat.blocks)] * len(var_model.groups_idx[i])
        else:
            x_list += [np.array(g.strat.blocks)]
    x = np.concatenate(x_list)
    cost_from_matrix = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_matrix), (
        f"Error in est coeffs exp, exp={cost_from_func}, got={cost_from_matrix}")

    # # Run test for another strategy
    new_blocks = (np.pi, -np.pi)
    c.update_group(group=group, blocks=new_blocks)
    cost_from_func = c.cost_trader(group=group)
    x_list = []
    for i, g in enumerate(c.groups):
        if isinstance(var_model.groups_idx[i], dict):
            x_list += [np.array(g.strat.blocks)] * len(var_model.groups_idx[i])
        else:
            x_list += [np.array(g.strat.blocks)]
    x = np.concatenate(x_list)
    cost_from_matrix = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_matrix), (
        f"Error in est coeffs exp, exp={cost_from_func}, got={cost_from_matrix}")


@pt.mark.parametrize("lambd_arr, ntraders, group", [
    ([1, 1], 1, 0),
    ([5, 2], 1, 0),
    ([1, 0], 2, 1),
    ([3, 2], 2, 0),
    ([10, 4], 1, 1),
    ([4, 1], 3, 1),
])
def test_sines_only(lambd_arr, ntraders, group):
    rho = 10
    N = 2
    ngroups = 3
    # Create the model
    strats = []
    lambd_padded = pad_array(lambd_arr, ngroups)
    for i in range(ngroups):
        strats.append(SinesBlocks(N, blocks=(0, 0), coeff=np.arange(N) + 1, lambd=lambd_padded[i]))
    groups = [Group(f"Trader_{i}", strats[i], ntraders) for i in range(2)]
    c = CostModelK(groups, rho)
    res = c.est_trader_coeffs(group=group, use_sines=True, use_blocks=False)
    if DEBUG:
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                np.set_printoptions(precision=4, suppress=True)
            print(f"{k}:")
            print(f"{v}\n")

    # Check the results
    cost_from_func = c.cost_trader(group=group)
    H, f, C, var_model = res['H'], res['f'], res['C'], c.var_model
    x_list = []
    for i, g in enumerate(c.groups):
        if isinstance(var_model.groups_idx[i], dict):
            x_list += [np.array(g.strat.coeff)] * len(var_model.groups_idx[i])
        else:
            x_list += [np.array(g.strat.coeff)]
    x = np.concatenate(x_list)
    cost_from_matrix = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_matrix), (
        f"Error in est coeffs exp, exp={cost_from_func}, got={cost_from_matrix}")

    # # Run test for another strategy
    new_coeff = np.arange(N) * 0.5
    c.update_group(group=group, coeff=new_coeff)
    cost_from_func = c.cost_trader(group=group)
    x_list = []
    for i, g in enumerate(c.groups):
        if isinstance(var_model.groups_idx[i], dict):
            x_list += [np.array(g.strat.coeff)] * len(var_model.groups_idx[i])
        else:
            x_list += [np.array(g.strat.coeff)]
    x = np.concatenate(x_list)
    cost_from_matrix = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_matrix), (
        f"Error in est coeffs exp, exp={cost_from_func}, got={cost_from_matrix}")


@pt.mark.parametrize("blocks_arr, lambd_arr, ntraders, group", [
    ([(1, 0), (0, 0)], [1, 1], 1, 0),
    ([(0, 1), (0, 0)], [5, 2], 1, 0),
    ([(0.5, 0.5), (-0.5, 0)], [1, 0], 2, 1),
    ([(0.3, 0.2), (-0.1, 0.4)], [3, 2], 2, 0),
    ([(0.3, 0.2), (-0.1, 0.4)], [10, 4], 1, 1),
    ([(-0.1, -0.2), (1.1, 1.2)], [4, 1], 3, 1),
])
def test_blocks_sines(blocks_arr, lambd_arr, ntraders, group):
    rho = 10
    N = 3
    ngroups = 3
    # Create the model
    strats = []
    lambd_padded = pad_list(lambd_arr, ngroups)
    blocks_padded = pad_list(blocks_arr, ngroups)
    for i in range(ngroups):
        strats.append(SinesBlocks(N, blocks=blocks_padded[i], coeff=np.arange(N) + 1, lambd=lambd_padded[i]))
    groups = [Group(f"Trader_{i}", strats[i], ntraders) for i in range(2)]
    c = CostModelK(groups, rho)
    res = c.est_trader_coeffs(group=group, use_sines=True, use_blocks=False)
    if DEBUG:
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                np.set_printoptions(precision=4, suppress=True)
            print(f"{k}:")
            print(f"{v}\n")

    # Check the results
    cost_from_func = c.cost_trader(group=group)
    H, f, C, var_model = res['H'], res['f'], res['C'], c.var_model
    x_list = []
    for i, g in enumerate(c.groups):
        if isinstance(var_model.groups_idx[i], dict):
            x_list += [np.array(g.strat.coeff)] * len(var_model.groups_idx[i])
        else:
            x_list += [np.array(g.strat.coeff)]
    x = np.concatenate(x_list)
    cost_from_matrix = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_matrix), (
        f"Error in est coeffs exp, exp={cost_from_func}, got={cost_from_matrix}")

    # # Run test for another strategy
    new_coeff = np.arange(N) * 0.5
    c.update_group(group=group, coeff=new_coeff)
    cost_from_func = c.cost_trader(group=group)
    x_list = []
    for i, g in enumerate(c.groups):
        if isinstance(var_model.groups_idx[i], dict):
            x_list += [np.array(g.strat.coeff)] * len(var_model.groups_idx[i])
        else:
            x_list += [np.array(g.strat.coeff)]
    x = np.concatenate(x_list)
    cost_from_matrix = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_matrix), (
        f"Error in est coeffs exp, exp={cost_from_func}, got={cost_from_matrix}")
