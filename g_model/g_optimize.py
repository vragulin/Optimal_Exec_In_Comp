""" Optimize execution strategy using the model from the Alfonsi, etc. paper
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1498514
V. Ragulin - 11/15/2024
"""
import numpy as np
from typing import Callable

import numpy as np
from typing import Callable

def v_g(t: np.ndarray, g: Callable[[float], float]) -> np.ndarray:
    """
    Vectorized version of the function.
    :param t: array of time points
    :param t_n: trading times in ascending order, all t in [0,1]
    :param x_n: trade sizes
    :param g: g(t) the resilience function
    :return: array of results
    """
    # Assuming g is a function that can be applied element-wise
    return np.vectorize(g)(t)

def price(t: float, t_n: np.ndarray, x_n: np.ndarray, g: Callable) -> float:#
    """ Price impact at a point in time
    :param t_n: trading times in ascending order, all t in [0,1]
    :param x_n: trade sizes
    :param g: g(t) the resiliance function
    :return: price displacement at time t
    """
    def g_all_range(t):
        if t<0:
            return 0
        else:
            return g(t)

    decay_coef = np.vectorize(g_all_range)(t-t_n)
    return x_n @ decay_coef

def imp_cost(t_n: np.ndarray, x_n: np.ndarray, g: Callable) -> float:
    """ Implementation cost of a trade trajectory
        :param t_n: trading times in ascending order, all t in [0,1]
        :param x_n: trade sizes
        :param g: g(t) the resiliance function
        :return: total implementation cost
        """
    tol = 1e-8
    prices_before = np.array([price(tau-tol, t_n, x_n, g) for tau in t_n])
    return float(np.sum((prices_before + x_n / 2) * x_n))


def imp_cost_matrix(x_n: np.ndarray, g_mat: np.ndarray) -> float:
    """ Implementation cost of a trade trajectory, version using the decay matrix
        :param x_n: trade sizes
        :param g: g_mat NxN matrix of decay coefficients
        :return: total implementation cost
        """
    return float(0.5 * x_n.T @ g_mat @ x_n)


def decay_matrix(t_n: np.ndarray, g: Callable) -> np.ndarray:
    """ Calculate the matrix of decay coefficients
    :param t_n: trading times in ascending order, all t in [0,1]
    :param g: g(t) the resiliance function
    :return: matrix of decay coefficients
    """
    t_abs_diff = np.abs(t_n[:, None] - t_n)
    return np.vectorize(g)(t_abs_diff)

def opt_trades_inv(t_n: np.ndarray, inv_decay_mat: np.ndarray | None
                      ) -> np.ndarray:
    """ Optimize trading path with no constraints
    :param t_n: trading times in ascending order, all t in [0,1]
    :param inv_decay_mat: inverse of the decay matrix
    :return: tuple x_opt - array of optimal trades, dict of other optimization stats
    """
    ones = np.ones(len(t_n))
    denom = ones @ inv_decay_mat @ ones
    x_opt = inv_decay_mat @ ones / denom
    return x_opt

def opt_trades_matrix(t_n: np.ndarray, decay_mat: np.ndarray | None
                      ) -> np.ndarray:
    """ Optimize trading path with no constraints
    :param t_n: trading times in ascending order, all t in [0,1]
    :param decay_mat: the decay matrix
    :return: tuple x_opt - array of optimal trades, dict of other optimization stats
    """
    ones = np.ones(len(t_n))
    numerator = np.linalg.solve(decay_mat, ones)
    denominator = np.sum(numerator)
    x_opt = numerator / denominator
    return x_opt

