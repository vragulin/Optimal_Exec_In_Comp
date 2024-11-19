""" Test the utility module
    V. Ragulin - 11/19/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

import g_utils as gu

DEBUG = True


@pt.mark.parametrize("N", [3, 10])
@pt.mark.parametrize("c", [0, 0.8, 1.25, 1.5])
def test_parabolic_add_to_1(c, N):
    # Test the parabolic function that adds to 1
    trades = gu.parabolic(c, N)
    if DEBUG:
        print(f"c={c}, trades: {trades}")
    assert pt.approx(np.sum(trades), abs=1e-6) == 1
