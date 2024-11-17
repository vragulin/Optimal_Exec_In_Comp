""" Test create_lower_triangular_matrix
    V. Ragulin - 11/16/2024
"""
import numpy as np
import pytest as pt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../g_model')))

import g_one_trader as go
import g_multi_trader as gm

DEBUG = True
@pt.mark.parametrize("n, x_diag",
                     [(2, 0.6), (3, 0.7), (10, 0.8)])
def test_n(n, x_diag):
    result = gm.create_lower_triangular_matrix(n, x_diag)

    # Call the function
    for i in range(n):
        for j in range(n):
            if i < j:
                assert result[i,j] == 0
            elif i == j:
                assert result[i,j] == x_diag
            else:
                assert result[i,j] == 1

@pt.mark.parametrize("n,", range(1,5))
def test_default_diag(n):
    n = 2
    result = gm.create_lower_triangular_matrix(n)

    # Call the function
    for i in range(n):
        for j in range(n):
            if i < j:
                assert result[i,j] == 0
            elif i == j:
                assert result[i,j] == 0.5
            else:
                assert result[i,j] == 1
