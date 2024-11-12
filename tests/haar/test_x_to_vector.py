"""Calculate the function to convert x into a vector of {-1,0,1} of the wavelet signs."""
import pytest as pt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs_fast as hf

DEBUG = True


@pt.mark.parametrize("x, res", [
    (0, [1, 1]),
    (0.2, [1, 1]),
    (0.5, [1, -1]),
    (0.9, [1, -1]),
    (1, [1, 0]),
])
def test_x_to_vector_level1(x, res):
    level = 1
    lmum = hf.calc_lmum(level)
    v = hf.x_to_vector(x, lmum)
    if DEBUG:
        print("v:", v)
    assert np.allclose(v, res)


@pt.mark.parametrize("level", [2, 3, 5])
@pt.mark.parametrize("x", [0, 0.2, 0.5, 0.9, 1])
def test_x_to_vector_nlevels(level, x):
    lmum = hf.calc_lmum(level)
    v = hf.x_to_vector(x, lmum)
    if DEBUG:
        print(f"\nx: {x}, v: {v}")

    # build the expected vector

    expected = np.ones(2 ** level)
    for i in range(1, 2 ** level):
        n, k = hf.i_to_nk(i)
        expected[i] = hf.psi_n_k(x, n, k)
    assert np.allclose(v, expected)
