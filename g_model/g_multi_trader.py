""" Optimize execution strategy using the model from the Alfonsi, etc. paper
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1498514
for several traders
V. Ragulin - 11/15/2024
"""
import numpy as np
from typing import Callable
from scipy.optimize import minimize, LinearConstraint
import g_one_trader as go


def create_lower_triangular_matrix(n: int, x_diag: float = 0.5):
    # Initialize an nxn matrix with zeros
    matrix = np.zeros((n, n))

    # Set the lower triangular part to 1, excluding the diagonal
    matrix[np.tril_indices(n, -1)] = 1

    # Set the diagonal to 0.5
    np.fill_diagonal(matrix, x_diag)
    return matrix


def decay_matrix_lt(t_n: np.ndarray, g: Callable) -> np.ndarray:
    """ Calculate the matrix of decay coefficients
    :param t_n: trading times in ascending order, all t in [0,1]
    :param g: g(t) the resiliance function
    :return: NxN matrix of decay coefficients, lower triangular
    """
    g_mat = go.decay_matrix(t_n, g)
    mult = create_lower_triangular_matrix(len(t_n))
    return g_mat * mult


def cost_trader(t_n: np.ndarray, x_n: np.ndarray, m_n: np.ndarray,
                g: Callable, trd_in_mkt: bool = True) -> float:
    """ Implementation cost of a trade trajectory, not using the decay matrix
        :param t_n: trading times in ascending order, all t in [0,1]
        :param x_n: the trader's own trade sizes
        :param m_n: the market trade sizes
        :param g: g(t) the resiliance function
        :param trd_in_mkt: whether the m_n includes the trader's trades
        :return: total implementation cost
        """
    tol = 1e-8
    dm_n = np.zeros_like(m_n) if trd_in_mkt else x_n
    total_vol = m_n + dm_n
    prices_before = np.array([go.price(tau - tol, t_n, total_vol, g) for tau in t_n])
    return float(np.sum((prices_before + total_vol / 2) * x_n))


def cost_trader_mat(x_n: np.ndarray, m_n: np.ndarray,
                    decay_mat_lt: np.ndarray, trd_in_mkt: bool = True) -> float:
    """ Implementation cost of a trade trajectory, not using the decay matrix
        :param x_n: the trader's own trade sizes
        :param m_n: the market trade sizes
        :param decay_mat_lt: NxN matrix of decay coefficients, lower triangular
        :param trd_in_mkt: whether the m_n includes the trader's trades
        :return: total implementation cost
        """
    dm_n = np.zeros_like(m_n) if trd_in_mkt else x_n
    total_vol = m_n + dm_n
    return float(x_n.T @ decay_mat_lt @ total_vol)


def best_response(t_n: np.ndarray, b_n: np.ndarray, g: Callable, **kwargs) -> tuple:
    """ Calculate the best response of a trader to the adversary's trades
    :param t_n: trading times in ascending order, all t in [0,1]
    :param b_n: the adversary's trade sizes
    :param g: g(t) the resiliance function
    :return: tuple: optimal trades, optimization stats from scipy
    """
    n = len(t_n)
    init_guess = np.ones(n) / n
    decay_mat_lt = decay_matrix_lt(t_n, g)

    constraint = LinearConstraint(np.ones(n), 1, 1)
    cost = lambda x: cost_trader_mat(t_n, x, b_n, decay_mat_lt, trd_in_mkt=False)

    tol = kwargs.get('tol', 1e-8)
    res = minimize(cost, init_guess, constraints=[constraint], tol=tol)
    return res.x, res


def best_response_mat(b_n: np.ndarray, decay_mat_lt: np.ndarray,
                      **kwargs) -> tuple:
    """ Calculate the best response of a trader to the adversary's trades
    :param b_n: the adversary's trade sizes
    :param decay_mat_lt: NxN matrix of decay coefficients, lower triangular
    :return: tuple: optimal trades, optimization stats from scipy
    """
    ones = np.ones_like(b_n)
    D = decay_mat_lt
    DpDt = decay_mat_lt + decay_mat_lt.T
    denom = np.sum(np.linalg.solve(DpDt, ones))
    numer = 1 + np.sum(np.linalg.solve(DpDt, D @ b_n))
    lambd = numer / denom
    a_opt = np.linalg.solve(DpDt, ones * lambd - D @ b_n)
    return a_opt, {'a_opt': a_opt, 'lambda': lambd}


def two_trader_equilibrium_mat(sizes: list | tuple | np.ndarray, decay_mat_lt: np.ndarray,
                               unit_strat: bool = False) -> tuple:
    """ Calculate the equilibrium of two traders analytically
    :param sizes: target sizes for both traders
    :param decay_mat_lt: NxN matrix of decay coefficients, lower triangular
    :param unit_strat: whether to return the unit stradegies or total trade sizes
    :return: tuple: optimal trades for both traders
    """

    D = decay_mat_lt + decay_mat_lt.T
    n = D.shape[0]
    sizes_np = sizes if isinstance(sizes, np.ndarray) else np.array(sizes)

    ones_nx1 = np.ones((n, 1))
    zeros_nxn = np.zeros((n, n))
    zeros_nx1 = np.zeros((n, 1))
    zeros_1x2 = np.zeros((1, 2))

    foc_a = np.hstack((D, decay_mat_lt, -ones_nx1, zeros_nx1))   # First order conditions (Nash Eq) for trader A
    foc_b = np.hstack((decay_mat_lt, D, zeros_nx1, -ones_nx1))  # First order conditions (Nash Eq) for trader B
    cons_size_a = np.hstack((ones_nx1.T, zeros_nx1.T, zeros_1x2))  # Total size constraint for trader A
    cons_size_b = np.hstack((zeros_nx1.T, ones_nx1.T, zeros_1x2))  # Total size constraint for trader B

    stacked_matrix = np.vstack((foc_a, foc_b, cons_size_a, cons_size_b))

    b = np.vstack((zeros_nx1, zeros_nx1, sizes_np[:, None]))
    x = np.linalg.solve(stacked_matrix, b).ravel()
    if unit_strat:
        return x[:n] / sizes[0], x[n:2 * n] / sizes[1]
    else:
        return x[:n], x[n:2 * n]
