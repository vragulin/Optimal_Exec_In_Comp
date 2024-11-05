""" Tests for the groups class for K trader model in the lin_prop_blocks module.
    V. Ragulin, 11/2/2024
"""

import os
import sys
from pickle import FALSE

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

import cost_model_K as ck
from prop_blocks import SinesBlocks
import propagator as pr

DEBUG = True


def test_create_print_group():
    g = ck.Group("Tom", SinesBlocks(2, blocks=(0, 0), coeff=np.arange(2)), ntraders=2)
    if DEBUG:
        print(g)
    assert g.name == "Tom", f"Name is incorrect, got={g.name}"
    assert g.ntraders == 2, f"ntraders is incorrect, got={g.ntraders}"
    np.testing.assert_array_equal(g.strat.coeff, np.arange(2)), f"coeff is incorrect, got={g.strat.coeff}"


def test_create_print_group_no_ntraders():
    g = ck.Group("Tom", SinesBlocks(2, blocks=(0, 0), coeff=np.arange(2)))
    if DEBUG:
        print(g)
    assert g.name == "Tom", f"Name is incorrect, got={g.name}"
    assert g.ntraders == 1, f"ntraders is incorrect, got={g.ntraders}"
    np.testing.assert_array_equal(g.strat.coeff, np.arange(2)), f"coeff is incorrect, got={g.strat.coeff}"
