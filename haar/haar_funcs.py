""" Functions reated to Haar wavelet transform
V. Ragulin - 11/08/2024
"""
import numpy as np
from scipy.integrate import quad

phi = lambda x: int((0 <= x < 1))  # scaling function
psi = lambda x: int((0 <= x < 0.5)) - int((0.5 <= x < 1))  # wavelet function
phi_j_k = lambda x, j, k: 2 ** (j / 2.0) * phi(2 ** j * x - k)
psi_j_k = lambda x, j, k: 2 ** (j / 2.0) * psi(2 ** j * x - k)


def haar_coeff(f, level: int = 6) -> tuple:
    """ Estimate haar coefficients on the interval [0, 1] for a given function f
    :param f: function to estimate haar coefficients
    :param level: level of the wavelet transform
    :return: c0, [(j, k, djk)] where j, k are the level and position index of the wavelet basis function
    """
    c0, _ = quad(lambda t: f(t) * phi_j_k(t, 0, 0), 0, 1)

    coef = []

    def integrand(t, j, k):
        return f(t) * psi_j_k(t, j, k)

    for j in range(0, level):  # xrange -> range for Python 3
        step = 1 / 2 ** j
        for k in range(0, 2 ** j):
            lb = step * k
            mid = step * (k + 0.5)
            ub = step * (k + 1)
            djk1, _ = quad(integrand, lb, mid - np.finfo(float).eps, (j, k))
            djk2, _ = quad(integrand, mid, ub + np.finfo(float).eps, (j, k))
            djk = djk1 + djk2
            coef.append((j, k, djk))

    return c0, coef


def reconstruct_from_haar(haar_coef: tuple[float, list[tuple[int, int, float]]], x: float) -> float:
    """
    Reconstructs the function value at a given point x from its Haar wavelet coefficients.

    :param haar_coef: tuple containing:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: list of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function
            - k: int, the position index of the wavelet basis function
            - djk: float, the coefficient for the wavelet at (j, k)
    :param x: float, the point at which to reconstruct the function value.
    :return: float, the reconstructed function value at x.
    """
    c0, coef = haar_coef
    s = c0 * phi_j_k(x, 0, 0)
    for j, k, djk in coef:
        s += djk * psi_j_k(x, j, k)
    return s


def integrate_haar_quad(haar_coef: tuple[float, list[tuple[int, int, float]]], a: float, b: float,
                        points: bool = True) -> float:
    """
    Computes the integral of a function represented by Haar wavelets over [a, b] ⊆ [0, 1] using quadrature.

    :param haar_coef: tuple containing:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: list of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function
            - k: int, the position index of the wavelet basis function
            - djk: float, the coefficient for the wavelet at (j, k)
    :param a: float, the lower limit of integration, with 0 ≤ a < b ≤ 1.
    :param b: float, the upper limit of integration, with 0 ≤ a < b ≤ 1.
    :param points: bool, whether to use quadrature points at the Haar wavelet support boundaries.
    :return: float, the result of the integral over [a, b]
    """
    if points:
        level = int(np.round(np.log(len(haar_coef[1]) + 1) / np.log(2)))
        points = list(np.linspace(0, 1, 2 ** level + 1).astype(float))
    else:
        points = None
    s = quad(lambda t: reconstruct_from_haar(haar_coef, t), a, b, points=points)[0]
    return s


def integrate_haar(haar_coef: tuple[float, list[tuple[int, int, float]]], a: float, b: float) -> float:
    """
    Computes the integral of a function represented by Haar wavelets over [a, b] ⊆ [0, 1].

    :param haar_coef: tuple containing:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: list of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function
            - k: int, the position index of the wavelet basis function
            - djk: float, the coefficient for the wavelet at (j, k)
    :param a: float, the lower limit of integration, with 0 ≤ a < b ≤ 1.
    :param b: float, the upper limit of integration, with 0 ≤ a < b ≤ 1.

    :return: float, the result of the integral over [a, b]
    """
    import numpy as np
    c0, coef = haar_coef
    integral = c0 * (b - a)  # Scaling function contribution over [a, b]

    # Loop through each wavelet term in coef
    for (j, k, djk) in coef:
        # Determine the support interval of the Haar wavelet in [0, 1]
        support_start = k / (2 ** j)
        support_mid = (k + 0.5) / (2 ** j)
        support_end = (k + 1) / (2 ** j)

        # Scaling factor for the Haar wavelet at level j
        scaling_factor = 2 ** (j / 2)

        # Positive half interval overlap
        pos_overlap_start = max(a, support_start)
        pos_overlap_end = min(b, support_mid)
        if pos_overlap_start < pos_overlap_end:
            pos_overlap_length = pos_overlap_end - pos_overlap_start
            integral += djk * scaling_factor * pos_overlap_length  # Positive contribution

        # Negative half interval overlap
        neg_overlap_start = max(a, support_mid)
        neg_overlap_end = min(b, support_end)
        if neg_overlap_start < neg_overlap_end:
            neg_overlap_length = neg_overlap_end - neg_overlap_start
            integral -= djk * scaling_factor * neg_overlap_length  # Negative contribution

    return integral


def price_haar(t: float, haar_coeffs: tuple[float, list[tuple[int, int, float]]], rho: float) -> float:
    """
    Calculate the price impact of a displacement at time t using Haar wavelets.
        P = integral_0^t exp(-rho * (t - s)) * m'(s) ds
        where m'(s) is the combined trading intensity by all traders at time s,
        which is approximated by the Haar wavelet

    :param t:  the time at which to calculate the price impact.
    :param haar_coeffs: tuple containing the Haar wavelet coefficients:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: list of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function
            - k: int, the position index of the wavelet basis function
            - djk: float, the coefficient for the wavelet at (j, k)
    :param rho: speed of mean-reversion of the price displacement.
    :return: price displacement at time t.
    """

    # Aliases
    exp = np.exp
    c0, coeffs = haar_coeffs

    integral_value = 0

    # Add the contribution of the average coefficient c0
    if t > 0:
        integral_value += c0 * (1 - np.exp(-rho * t)) / rho

    for level, k, value in coeffs:
        step_size = 1 / 2 ** level
        start = k * step_size
        mid = start + step_size / 2
        end = start + step_size
        mult = 2 ** (level / 2.0)

        if t < mid:
            integral_neg = 0
            if t < start:
                integral_pos = 0
            else:
                integral_pos = (np.exp(rho * t) - np.exp(rho * start))
        else:
            integral_pos = (np.exp(rho * mid) - np.exp(rho * start))
            if t < end:
                integral_neg = (np.exp(rho * t) - np.exp(rho * mid))
            else:
                integral_neg = (np.exp(rho * end) - np.exp(rho * mid))

        integral_value += value * (integral_pos - integral_neg) * mult / rho / exp(rho * t)

    return integral_value
