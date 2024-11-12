""" Functions reated to Haar wavelet transform
V. Ragulin - 11/08/2024
"""
import numpy as np
from scipy.integrate import quad, dblquad
from typing import Tuple, List
from matplotlib import pyplot as plt

# Custom type
np.ndarray = Tuple[float, List[Tuple[int, int, float]]]
PLOT_COST_INT = False
EPS_ABS_QUAD = 1e-4

def scaling_function(x: float) -> int:
    """Scaling function phi."""
    return int(0 <= x < 1)



def wavelet_function(x: float) -> int:
    """Wavelet function psi."""
    return int(0 <= x < 0.5) - int(0.5 <= x < 1)


def phi_j_k(x: float, j: int, k: int) -> float:
    """Scaling function at level j and position k."""
    return 2 ** (j / 2.0) * scaling_function(2 ** j * x - k)


def psi_j_k(x: float, j: int, k: int) -> float:
    """Wavelet function at level j and position k."""
    return 2 ** (j / 2.0) * wavelet_function(2 ** j * x - k)


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
        return f(t, *func_args) * psi_j_k(t, j, k)

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

    return c0, coef


def reconstruct_from_haar(haar_coef: np.ndarray, x: float) -> float:
    """
    Reconstruct the function value at a given point x from its Haar wavelet coefficients.

    :param haar_coef: Tuple containing:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: List of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function.
            - k: int, the position index of the wavelet basis function.
            - djk: float, the coefficient for the wavelet at (j, k).
    :param x: float, the point at which to reconstruct the function value.
    :return: float, the reconstructed function value at x.
    """
    c0, coef = haar_coef
    s = c0 * phi_j_k(x, 0, 0)
    for j, k, djk in coef:
        s += djk * psi_j_k(x, j, k)
    return s


def integrate_haar_quad(haar_coef: np.ndarray, a: float, b: float, points: bool = True) -> float:
    """
    Compute the integral of a function represented by Haar wavelets over [a, b] using quadrature.

    :param haar_coef: Tuple containing:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: List of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function.
            - k: int, the position index of the wavelet basis function.
            - djk: float, the coefficient for the wavelet at (j, k).
    :param a: float, the lower limit of integration, with 0 ≤ a < b ≤ 1.
    :param b: float, the upper limit of integration, with 0 ≤ a < b ≤ 1.
    :param points: bool, whether to use quadrature points at the Haar wavelet support boundaries.
    :return: float, the result of the integral over [a, b].
    """
    if points:
        level = int(np.round(np.log(len(haar_coef[1]) + 1) / np.log(2)))
        points = list(np.linspace(0, 1, 2 ** level + 1).astype(float))
    else:
        points = None
    s = quad(lambda t: reconstruct_from_haar(haar_coef, t), a, b, points=points)[0]
    return s


def integrate_haar(haar_coef: np.ndarray, a: float, b: float) -> float:
    """
    Compute the integral of a function represented by Haar wavelets over [a, b].

    :param haar_coef: Tuple containing:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: List of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function.
            - k: int, the position index of the wavelet basis function.
            - djk: float, the coefficient for the wavelet at (j, k).
    :param a: float, the lower limit of integration, with 0 ≤ a < b ≤ 1.
    :param b: float, the upper limit of integration, with 0 ≤ a < b ≤ 1.
    :return: float, the result of the integral over [a, b].
    """
    c0, coef = haar_coef
    integral = c0 * (b - a)

    for j, k, djk in coef:
        support_start = k / (2 ** j)
        support_mid = (k + 0.5) / (2 ** j)
        support_end = (k + 1) / (2 ** j)
        scaling_factor = 2 ** (j / 2)

        pos_overlap_start = max(a, support_start)
        pos_overlap_end = min(b, support_mid)
        if pos_overlap_start < pos_overlap_end:
            pos_overlap_length = pos_overlap_end - pos_overlap_start
            integral += djk * scaling_factor * pos_overlap_length

        neg_overlap_start = max(a, support_mid)
        neg_overlap_end = min(b, support_end)
        if neg_overlap_start < neg_overlap_end:
            neg_overlap_length = neg_overlap_end - neg_overlap_start
            integral -= djk * scaling_factor * neg_overlap_length

    return integral


def price_haar(t: float, haar_coeffs: np.ndarray, rho: float) -> float:
    """
    Calculate the price impact of a displacement at time t using Haar wavelets.

    :param t: float, the time at which to calculate the price impact.
    :param haar_coeffs: Tuple containing the Haar wavelet coefficients:
        - c0: float, the coefficient for the scaling function over [0, 1].
        - coef: List of tuples [(j, k, djk)], where:
            - j: int, the level of the wavelet basis function.
            - k: int, the position index of the wavelet basis function.
            - djk: float, the coefficient for the wavelet at (j, k).
    :param rho: float, speed of mean-reversion of the price displacement.
    :return: float, the price displacement at time t.
    """
    exp = np.exp
    c0, coeffs = haar_coeffs
    integral_value = 0

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


def add_haar(coeffs_list: list[np.ndarray], w: list[float] | None = None) -> np.ndarray:
    """
    Combine Haar coefficients from multiple traders into a single set of coefficients for the market.

    :param coeffs_list: List[np.ndarray], a list of Haar coefficients for each trader.
    :param w: weights, should be the same size as the list of coefficients.
              If None, equal weights of 1 are used.
    :return: np.ndarray, the combined Haar coefficients for the market.
    """
    if w is None:
        w = [1] * len(coeffs_list)
    else:
        assert len(w) == len(coeffs_list), "Weights should be the same size as the list of coefficients."

    c0 = 0

    # Build a set of all waveles that are present in the encodings
    wavelets_set = set()
    for coeffs in coeffs_list:
        wavelets_set |= {(j, k) for j, k, djk in coeffs[1]}

    wavelets_dict = {wavelet: 0 for wavelet in wavelets_set}

    # For each wavelet, sum the coefficients from each trader
    for coeffs, weight in zip(coeffs_list, w):
        c0 += weight * coeffs[0]
        for wavelet in coeffs[1]:
            wavelets_dict[wavelet[:2]] += weight * wavelet[2]

    # Remove wavelets with zero coefficients
    coef = [(j, k, djk) for (j, k), djk in wavelets_dict.items() if djk != 0]
    coef.sort(key=lambda x: (x[0], x[1]))

    return c0, coef


def cost_quad(trader_coeffs: np.ndarray, mkt_coeffs: np.ndarray, rho: float, **kwargs) -> float:
    """
    Calculate the implementation cost of a trader given the market coefficients and the rho parameter.

    :param trader_coeffs: np.ndarray, Haar coefficients for the trader.
    :param mkt_coeffs: np.ndarray, Haar coefficients for the market.
    :param rho: float, the speed of mean-reversion.
    :return: float, the implementation cost of the trader.
    """

    def integrand(t):
        a_prime = reconstruct_from_haar(trader_coeffs, t)
        p_t = price_haar(t, mkt_coeffs, rho)
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


def cost_quad_x2(trader_coeffs: np.ndarray, mkt_coeffs: np.ndarray, rho: float, **kwargs) -> float:
    """
    ** THIS FUNCTION IS VERY SLOW  - ONLY USE FOR TESTIG ***
    Calculate the implementation cost of a trader given the market coefficients and the rho parameter.
    Calculate the price function via direct integration, rather than taking advantage of the
    Haar wavelet representation.

    :param trader_coeffs: np.ndarray, Haar coefficients for the trader.
    :param mkt_coeffs: np.ndarray, Haar coefficients for the market.
    :param rho: float, the speed of mean-reversion.
    :return: float, the implementation cost of the trader.
    """

    points = kwargs.get('points', None)

    def mkt_prime(s):
        return reconstruct_from_haar(mkt_coeffs, s)

    def integrand(t):
        a_prime = reconstruct_from_haar(trader_coeffs, t)
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


def cost_dblquad(trader_coeffs: np.ndarray, mkt_coeffs: np.ndarray, rho: float, **kwargs) -> float:
    """
    Calculate the implementation cost of a trader given the market coefficients and the rho parameter.
    Calculate the price function via direct integration, rather than taking advantage of the
    Haar wavelet representation.

    The cost function is equvalent to a double integral
     int_0^t int_s^1 a'(t) exp(-rho(t-s)) m'(s) ds dt

    :param trader_coeffs: np.ndarray, Haar coefficients for the trader.
    :param mkt_coeffs: np.ndarray, Haar coefficients for the market.
    :param rho: float, the speed of mean-reversion.
    :return: float, the implementation cost of the trader.
    """

    epsabs = kwargs.get('epsabs', 1e-4)

    def mkt_prime(s):
        return reconstruct_from_haar(mkt_coeffs, s)

    def a_prime(t):
        return reconstruct_from_haar(trader_coeffs, t)

    def integrand(t, s):
        return a_prime(t) * np.exp(-rho * (t - s)) * mkt_prime(s)

    cost = dblquad(integrand, 0, 1, lambda s: s, 1, epsabs=EPS_ABS_QUAD)[0]

    return cost
