""" Test functions for adding haar representations.
"""
import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs as hf

DEBUG = True


@pt.mark.parametrize("weights", [(1, 2), (1, -1), (0, 1)])
@pt.mark.parametrize("level", [3, 3, 4, 6])
def test_add_two_vectors(weights, level):
    def func1(t):
        return t

    def func2(t):
        return 1 + t

    def func_add(t):
        return func1(t) * weights[0] + func2(t) * weights[1]

    haar1 = hf.haar_coeff(func1, level=level)
    haar2 = hf.haar_coeff(func2, level=level)
    haar_exp = hf.haar_coeff(func_add, level=level)

    haar_sum = hf.add_haar([haar1, haar2], weights)

    if DEBUG:
        print("haar1:", haar1)
        print("haar2:", haar2)
        print("haar_exp:", haar_exp)
        print("haar_sum:", haar_sum)

    sum_c0, sum_coeffs = haar_sum
    assert sum_c0 == haar_exp[0]
    assert len(sum_coeffs) == len(haar_exp[1])
    assert all([pt.approx(djk, abs=1e-6) == djk_exp
                for (j, k, djk), (j_exp, k_exp, djk_exp)
                in zip(sum_coeffs, haar_exp[1])])


@pt.mark.parametrize("n", [3, 10])
@pt.mark.parametrize("level", [3, 3, 4, 6])
def test_add_n_vectors(n, level):
    def func1(t):
        return t

    def func2(t):
        return 1 + np.cos(2 * np.pi * t)

    weights = [1] * n

    def func_add(t):
        res = func1(t) * weights[0]
        for i in range(1, n):
            res += func2(t) * weights[i]
        return res

    haar_list = [hf.haar_coeff(func1, level=level)]
    for i in range(1, n):
        haar_list.append(hf.haar_coeff(func2, level=level))

    haar_exp = hf.haar_coeff(func_add, level=level)
    haar_sum = hf.add_haar(haar_list, weights)

    if DEBUG:
        print("\nHaar list:")
        for i, h in enumerate(haar_list):
            print(f"haar{i}:", h)
        print("Results:")
        print("haar_exp:", haar_exp)
        print("haar_sum:", haar_sum)

    sum_c0, sum_coeffs = haar_sum
    assert sum_c0 == haar_exp[0]
    assert len(sum_coeffs) == len(haar_exp[1])
    assert all([pt.approx(djk, abs=1e-6) == djk_exp
                for (j, k, djk), (j_exp, k_exp, djk_exp)
                in zip(sum_coeffs, haar_exp[1])])
