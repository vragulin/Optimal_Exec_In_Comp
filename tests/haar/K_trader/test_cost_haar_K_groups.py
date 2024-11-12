""" Tests for the groups class for K trader model in the Haar module.
    V. Ragulin, 11/10/2024
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
])

import cost_haar_K as ck

DEBUG = True


def test_create_print_group():
    g = ck.Group("Tom", np.arange(2), ntraders=2)
    if DEBUG:
        print(g)
    assert g.name == "Tom", f"Name is incorrect, got={g.name}"
    assert g.ntraders == 2, f"ntraders is incorrect, got={g.ntraders}"
    np.testing.assert_array_equal(g.strat, np.arange(2)), f"coeff is incorrect, got={g.strat}"


def test_create_print_group_no_ntraders():
    g = ck.Group("Tom", np.arange(2))
    if DEBUG:
        print(g)
    assert g.name == "Tom", f"Name is incorrect, got={g.name}"
    assert g.ntraders == 1, f"ntraders is incorrect, got={g.ntraders}"
    np.testing.assert_array_equal(g.strat, np.arange(2)), f"coeff is incorrect, got={g.strat}"
