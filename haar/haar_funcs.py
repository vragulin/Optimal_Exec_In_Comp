""" Functions reated to Haar wavelet transform
V. Ragulin - 11/08/2024
"""
import numpy as np
from scipy.integrate import quad

phi = lambda x: int((0 <= x < 1))  # scaling function
psi = lambda x: int((0 <= x < 0.5)) - int((0.5 <= x < 1))  # wavelet function
phi_j_k = lambda x, j, k: 2 ** (j / 2) * phi(2 ** j * x - k)
psi_j_k = lambda x, j, k: 2 ** (j / 2) * psi(2 ** j * x - k)


def haar_coeff(f, interval, level):
    """ Old version, for high level quad is not precise enough
    because of a jump in the middle of the interval """
    c0, _ = quad(lambda t: f(t) * phi_j_k(t, 0, 0), *interval)

    coef = []

    def integrand(t, j, k):
        return f(t) * psi_j_k(t, j, k)

    for j in range(0, level):  # xrange -> range for Python 3
        step = 1 / 2 ** j * (interval[1] - interval[0])
        for k in range(0, 2 ** j):
            lb = interval[0] + step * k
            mid = interval[0] + step * (k + 0.5)
            ub = interval[0] + step * (k + 1)
            djk1, _ = quad(integrand, lb, mid-np.finfo(float).eps, (j, k))
            djk2, _ = quad(integrand, mid, ub+np.finfo(float).eps, (j, k))
            djk = djk1 + djk2
            coef.append((j, k, djk))

    return c0, coef


def reconstruct_from_haar(haar_coef, x):
    c0, coef = haar_coef
    s = c0 * phi_j_k(x, 0, 0)
    for j, k, djk in coef:
        s += djk * psi_j_k(x, j, k)
    return s
