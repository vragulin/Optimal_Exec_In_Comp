""" Functions reated to Haar wavelet transform
    Represent Haar wavelet in a dense format (np array) to improve performance
V. Ragulin - 11/10/2024
"""
import numpy as np
import math
from scipy.integrate import quad, dblquad
from typing import Tuple, List
from matplotlib import pyplot as plt

# Custom type
PLOT_COST_INT = False
EPS_ABS_QUAD = 1e-4
_L, _M, _U, _MUL = range(4)  # Label LMUM columns


def scaling_function(x: float) -> int:
    """Scaling function phi."""
    return int(0 <= x < 1)


def wavelet_function(x: float) -> int:
    """Wavelet function psi."""
    return int(0 <= x < 0.5) - int(0.5 <= x < 1)


def phi_n_k(x: float, n: int, k: int) -> float:
    """Scaling function at level j and position k."""
    return 2 ** (n / 2.0) * scaling_function(2 ** n * x - k)


def psi_n_k(x: float, n: int, k: int) -> float:
    """Wavelet function at level j and position k."""
    return 2 ** (n / 2.0) * wavelet_function(2 ** n * x - k)


def calc_lmum(level: int) -> np.ndarray:
    """Calculate the bounds and multipliers for the wavelet functions."""
    lmum = np.zeros((2 ** level, 4))
    for i in range(2 ** level):
        if i == 0:
            lmum[0, :] = np.array([0, 2, 4, 1])
        else:
            n, k = i_to_nk(i)
            step = 1 / 2 ** n
            lb = step * k
            mid = step * (k + 0.5)
            ub = step * (k + 1)
            lmum[i] = lb, mid, ub, 2 ** (n / 2.0)
    return lmum.astype(float)


def x_to_vector(x: float, lmum: np.ndarray) -> np.ndarray:
    """Vectorized wavelet function psi for arrays of n and k.
    :param x: float, the point at which to evaluate the wavelet function.
    :param lmum: nx4 array with interval bounds (low, mid, high, multiple) for each wavelet.
    """
    comp = (lmum[:, :3] <= x).astype(int)

    pos = comp[:, _L] - comp[:, _M]
    neg = comp[:, _M] - comp[:, _U]
    return (pos - neg) * lmum[:, _MUL]


def i_to_nk(i: int) -> Tuple[int, int] | None:
    """Convert index i to level j and position k."""
    if i == 0:
        return None  # This is the scaling function
    else:
        n = i.bit_length() - 1
        k = i - 2 ** n
        return n, k


def nk_to_i(n: int, k: int) -> int:
    """Convert level j and position k to index i."""
    return 2 ** n + k


def haar_coeff(f, level: int = 6, **kwargs) -> np.ndarray:
    """
    Estimate Haar coefficients on the interval [0, 1] for a given function f.

    :param f: Function to estimate Haar coefficients.
    :param level: Level of the wavelet transform.
    :return: Tuple containing c0 and a list of tuples (j, k, djk) where j and k
            are the level and position index of the wavelet basis function.
    """
    func_args = kwargs.get('func_args', ())
    c0, _ = quad(f, 0, 1, func_args)
    coef = []

    def integrand(t, j, k):
        return f(t, *func_args) * psi_n_k(t, j, k)

    for j in range(level):

        step = 1 / 2 ** j
        for k in range(2 ** j):
            lb = step * k
            mid = step * (k + 0.5)
            ub = step * (k + 1)
            djk1, _ = quad(integrand, lb, mid - np.finfo(float).eps, (j, k))
            djk2, _ = quad(integrand, mid, ub + np.finfo(float).eps, (j, k))
            djk = djk1 + djk2
            coef.append((j, k, djk))

    haar = np.zeros(2 ** level)
    haar[0] = c0
    for j, k, djk in coef:
        haar[2 ** j + k] = djk
    return haar


def reconstruct_from_haar(x: float, haar_coef: np.ndarray, lmum: np.ndarray | None = None) -> float:
    """
    Reconstruct the function value at a given point x from its Haar wavelet coefficients.

    :param haar_coef: array h containing:
        - h[0]: float, the coefficient for the scaling function over [0, 1].
        - h[1;]: coefficients for the wavelet functions at (n, k), sorted by
                increasing n and k.
    :param x: float, the point at which to reconstruct the function value.
    :param lmum: for speed up, precomputed bounds and multipliers for the wavelet functions.
    :return: float, the reconstructed function value at x.
    """

    if lmum is None:
        level = len(haar_coef).bit_length() - 1
        lmum = calc_lmum(level)
    x_vec = x_to_vector(x, lmum)
    return haar_coef @ x_vec


def integrate_haar_quad(haar_coef: np.ndarray, a: float, b: float, points: bool = True,
                        lmum: np.ndarray | None = None) -> float:
    """
    Compute the integral of a function represented by Haar wavelets over [a, b] using quadrature.

    :param haar_coef: array h containing:
        - h[0]: float, the coefficient for the scaling function over [0, 1].
        - h[1;]: coefficients for the wavelet functions at (n, k), sorted by
                increasing n and k.
    :param haar_coef: np.ndarray, Haar coefficients for the function.:
    :param a: float, the lower limit of integration, with 0 ≤ a < b ≤ 1.
    :param b: float, the upper limit of integration, with 0 ≤ a < b ≤ 1.
    :param points: bool, whether to use quadrature points at the Haar wavelet support boundaries.
    :param lmum: for speed up, precomputed bounds and multipliers for the wavelet functions.
    :return: float, the result of the integral over [a, b].
    """
    if points:
        level = len(haar_coef).bit_length() - 1
        points = list(np.linspace(0, 1, 2 ** level + 1).astype(float))
    else:
        points = None
    s = quad(lambda t: reconstruct_from_haar(t, haar_coef, lmum), a, b, points=points)[0]
    return s


def integrate_haar(haar_coef: np.ndarray, a: float, b: float,
                   lmum: np.ndarray | None = None) -> float:
    """
    Compute the integral of a function represented by Haar wavelets over [a, b].

    :param haar_coef: array h containing:
        - h[0]: float, the coefficient for the scaling function over [0, 1].
        - h[1;]: coefficients for the wavelet functions at (n, k), sorted by
                increasing n and k.
    :param a: float, the lower limit of integration, with 0 ≤ a < b ≤ 1.
    :param b: float, the upper limit of integration, with 0 ≤ a < b ≤ 1.
    :param lmum: for speed up, precomputed bounds and multipliers for the wavelet functions.
    :return: float, the result of the integral over [a, b].
    """

    if lmum is None:
        level = len(haar_coef).bit_length() - 1
        n = 2 ** level
        lmum = calc_lmum(level)

    if a == b:
        return 0
    else:
        assert a < b, "The lower limit of integration should be less than the upper limit."

    # Calculate the integral over the positive component of the wavelet
    pos_overlap_start = np.maximum(lmum[:, _L], a)
    pos_overlap_end = np.minimum(lmum[:, _M], b)
    pos_overlap_length = np.maximum(0, pos_overlap_end - pos_overlap_start)

    neg_overlap_start = np.maximum(lmum[:, _M], a)
    neg_overlap_end = np.minimum(lmum[:, _U], b)
    neg_overlap_length = np.maximum(0, neg_overlap_end - neg_overlap_start)

    integral = ((pos_overlap_length - neg_overlap_length) * lmum[:, _MUL]) @ haar_coef

    return integral


def price_haar(t: float, haar_coef: np.ndarray, rho: float,
               lmum: np.ndarray | None = None) -> float:
    """
    Calculate the price impact of a displacement at time t using Haar wavelets.

    :param t: float, the time at which to calculate the price impact.
    :param haar_coef: array h containing:
        - h[0]: float, the coefficient for the scaling function over [0, 1].
        - h[1;]: coefficients for the wavelet functions at (n, k), sorted by
                increasing n and k.
    :param rho: float, speed of mean-reversion of the price displacement.
    :param lmum: for speed up, precomputed bounds and multipliers for the wavelet functions.
    :return: float, the price displacement at time t.
    """
    exp = np.exp

    if t == 0:
        return 0

    if lmum is None:
        level = len(haar_coef).bit_length() - 1
        lmum = calc_lmum(level)

    # Calculate positive and neg integral based on the overlap of t and waveles
    START, END = range(2)

    pos_overlap = np.zeros((len(lmum), 2))
    pos_overlap[:, START] = lmum[:, _L]
    pos_overlap[:, END] = np.minimum(lmum[:, _M], t)
    pos_integral = np.where(
        pos_overlap[:, START] < pos_overlap[:, END],
        exp(rho * pos_overlap[:, END]) - exp(rho * pos_overlap[:, START]), 0
    )

    neg_overlap = np.zeros((len(lmum), 2))
    neg_overlap[:, START] = lmum[:, _M]
    neg_overlap[:, END] = np.minimum(lmum[:, _U], t)
    neg_integral = np.where(
        neg_overlap[:, START] < neg_overlap[:, END],
        exp(rho * neg_overlap[:, END]) - exp(rho * neg_overlap[:, START]), 0
    )

    integral = ((pos_integral - neg_integral) * lmum[:, _MUL]
                ) @ haar_coef / rho / exp(rho * t)

    return integral


def add_haar(coeffs_list: list[np.ndarray], w: list[float] | None = None) -> np.ndarray:
    """
    Combine Haar coefficients from multiple traders into a single set of coefficients for the market.

    :param coeffs_list: List[np.ndarray], a list of Haar coefficients for each trader.
    :param w: weights, should be the same size as the list of coefficients.
              If None, equal weights of 1 are used.
    :return: np.ndarray, the combined Haar coefficients for the market.
    """
    if w is None:
        w_arr = np.ones(len(coeffs_list))
    else:
        assert len(w) == len(coeffs_list), "Weights should be the same size as the list of coefficients."
        w_arr = np.array(w)

    coeff_stacked = np.column_stack(coeffs_list)
    return coeff_stacked @ w_arr


def cost_quad(trader_coeffs: np.ndarray, mkt_coeffs: np.ndarray, rho: float,
              lmum: np.ndarray | None = None,  **kwargs) -> float:
    """
    Calculate the implementation cost of a trader given the market coefficients and the rho parameter.

    :param trader_coeffs: np.ndarray, Haar coefficients for the trader.
    :param mkt_coeffs: np.ndarray, Haar coefficients for the market.
    :param rho: float, the speed of mean-reversion.
    :param lmum: for speed up, precomputed bounds and multipliers for the wavelet functions.
    :return: float, the implementation cost of the trader.
    """

    if lmum is None:
        level = len(trader_coeffs).bit_length() - 1
        lmum = calc_lmum(level)

    def integrand(t):
        a_prime = reconstruct_from_haar(t, trader_coeffs, lmum)
        p_t = price_haar(t, mkt_coeffs, rho, lmum)
        return a_prime * p_t

    points = kwargs.get('points', None)
    cost = quad(integrand, 0, 1, points=points)[0]

    if PLOT_COST_INT:
        t_values = np.linspace(0, 1, 101)
        cost_values = [integrand(t) for t in t_values]
        cum_cost_values = np.cumsum(cost_values) / 100
        plt.plot(t_values[:-1], cost_values[:-1], label="Cost integrand")
        plt.plot(t_values, cum_cost_values, label="Cumulative cost")
        plt.xlabel("t")
        plt.ylabel("Cost integrand")
        plt.title("Cost integrand and cumulative cost")
        plt.legend()
        plt.show()
    return cost


def cost_quad_x2(trader_coeffs: np.ndarray, mkt_coeffs: np.ndarray, rho: float,
                 lmum: np.ndarray | None = None, **kwargs) -> float:
    """
    ** THIS FUNCTION IS VERY SLOW  - ONLY USE FOR TESTIG ***
    Calculate the implementation cost of a trader given the market coefficients and the rho parameter.
    Calculate the price function via direct integration, rather than taking advantage of the
    Haar wavelet representation.

    :param trader_coeffs: np.ndarray, Haar coefficients for the trader.
    :param mkt_coeffs: np.ndarray, Haar coefficients for the market.
    :param rho: float, the speed of mean-reversion.
    :param lmum: for speed up, precomputed bounds and multipliers for the wavelet functions.
    :return: float, the implementation cost of the trader.
    """

    if lmum is None:
        level = len(trader_coeffs).bit_length() - 1
        lmum = calc_lmum(level)

    points = kwargs.get('points', None)

    def mkt_prime(s):
        return reconstruct_from_haar(s, mkt_coeffs, lmum)

    def integrand(t):
        a_prime = reconstruct_from_haar(t, trader_coeffs, lmum)
        price = quad(lambda s: np.exp(-rho * (t - s)) * mkt_prime(s),
                     0, t, points=points)[0]
        return a_prime * price

    cost = quad(integrand, 0, 1, points=points)[0]

    if PLOT_COST_INT:
        t_values = np.linspace(0, 1, 101)
        cost_values = [integrand(t) for t in t_values]
        cum_cost_values = np.cumsum(cost_values) / 100
        plt.plot(t_values[:-1], cost_values[:-1], label="Cost integrand")
        plt.plot(t_values, cum_cost_values, label="Cumulative cost")
        plt.xlabel("t")
        plt.ylabel("Cost integrand")
        plt.title("Cost integrand and cumulative cost")
        plt.legend()
        plt.show()
    return cost


def cost_dblquad(trader_coeffs: np.ndarray, mkt_coeffs: np.ndarray, rho: float,
                 lmum: np.ndarray | None = None, **kwargs) -> float:
    """
    Calculate the implementation cost of a trader given the market coefficients and the rho parameter.
    Calculate the price function via direct integration, rather than taking advantage of the
    Haar wavelet representation.

    :param trader_coeffs: np.ndarray, Haar coefficients for the trader.
    :param mkt_coeffs: np.ndarray, Haar coefficients for the market.
    :param rho: float, the speed of mean-reversion.
    :param lmum: for speed up, precomputed bounds and multipliers for the wavelet functions.
    :return: float, the implementation cost of the trader.
    """
    if lmum is None:
        level = len(trader_coeffs).bit_length() - 1
        lmum = calc_lmum(level)

    epsabs = kwargs.get('epsabs', 1e-4)

    def mkt_prime(s):
        return reconstruct_from_haar(s, mkt_coeffs, lmum)

    def a_prime(t):
        return reconstruct_from_haar(t, trader_coeffs, lmum)

    def integrand(t, s):
        return a_prime(t) * np.exp(-rho * (t - s)) * mkt_prime(s)

    cost = dblquad(integrand, 0, 1, lambda s: s, 1, epsabs=EPS_ABS_QUAD)[0]

    return cost
