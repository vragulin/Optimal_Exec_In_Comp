import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs_fast as hf

DEBUG = True


@pt.mark.parametrize("input, expected", [
    (0, None),
    (1, (0, 0)),
    (2, (1, 0)),
    (3, (1, 1)),
    (31, (4, 15)),
    (32, (5, 0)),
])
def test_i_to_jk(input, expected):
    assert hf.i_to_nk(input) == expected


@pt.mark.parametrize("input, expected", [
    ((0, 0), 1),
    ((1, 0), 2),
    ((1, 1), 3),
    ((4, 15), 31),
    ((5, 0), 32),
])
def test_jk_to_i(input, expected):
    assert hf.nk_to_i(*input) == expected
