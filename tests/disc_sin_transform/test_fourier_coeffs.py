import pytest as pt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("../../cost_function"))
sys.path.append(os.path.abspath("../../disc_sin_transform"))
import nc_fourier_coeffs as ncf
import dst_example1 as dst


def test_find_fourier_zero_coeffs():
    def a_function(t, kappa, lambda_, gamma=1):
        return gamma * t

    kapppa = 0
    lambda_ = 0
    N = 5

    c1, c2 = ncf.find_fourier_coefficients(a_function, a_function, 0, lambda_, N)
    np.testing.assert_allclose(c1, 0, atol=1e-9)


def test_reconstruct_func_zero_coefs():
    N = 5
    N_times = 10
    gamma = 1.1

    t = np.linspace(0, 1, N_times)

    coeffs = np.zeros(N)

    f = ncf.reconstruct_function(t, coeffs, N)
    np.testing.assert_allclose(f, t, atol=1e-9)


@pt.mark.skip("Not implemented")
def test_find_fourier_square():
    # ToDo - DST is not working yet.  Let's come back ot this later
    def a_function(t, kappa, lambda_, gamma=1):
        return t * t

    kapppa = 0
    lambda_ = 0
    N = 10
    N_points = 1000

    x = np.linspace(0, 1, N_points)
    y = a_function(x, kapppa, lambda_)

    c1, c2 = ncf.find_fourier_coefficients(a_function, a_function, 0, lambda_, N)
    # expected = dst.sine_transform_approximation(y-x, N_points)
    # np.testing.assert_allclose(c1, expected, atol=1e-9)
    assert False
