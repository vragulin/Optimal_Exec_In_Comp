"""  Solver the OW equilibrium analytically without optimization
    V. Ragulin - 10/08/2024
"""
import time

import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'matrix_utils')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..')))
# from propagator import cost_fn_prop_a_matrix
# from est_quad_coeffs import find_coefficients
# import prop_cost_to_QP as pcq


def foc_terms_a(n: int, lambd: float, rho: float) -> dict:
    """ Calculate the terms of the first order condition for trader A
    :param n: number of Fourier terms
    :param lambd: size of trader B
    :param rho: decay of the propagator
    :return: dictionary contaiining matrices H,G and vector m
    """

    pi, exp = np.pi, np.exp

    # Constants (later can be precomputed)
    n = np.arange(1, n + 1).reshape([-1, 1])
    N = np.diag(n.reshape(-1)).astype(float)
    n_sq = n ** 2
    i = np.ones(n.shape)
    i_odd = n % 2
    i_mp = i - 2 * i_odd

    m_p_n_odd = (n + n.T) % 2
    msq_nsq = n_sq - n_sq.T
    with np.errstate(divide='ignore', invalid='ignore'):
        M = 2 * np.where(m_p_n_odd, n_sq / msq_nsq, 0)

    D = np.diag((pi * n / (rho ** 2 + (n * pi) ** 2)).reshape(-1))
    h = D @ (i - exp(-rho) * i_mp)

    # Calluate the quadratic coefficients (Hessian)
    H = -rho ** 2 * (D @ i @ h.T + h @ i.T @ D) + pi * rho * N @ D + pi * (N @ M.T @ D + D @ M @ N)

    # Calculate the linear coefficients for the Cross-Term (B)
    G = lambd * (- rho ** 2 * h @ i.T @ D + pi * (0.5 * rho * N @ D + N @ M.T @ D))

    # The constant term
    m = D @ (exp(-rho) * i - i_mp) - (1 + lambd) * h

    return {'H': H, 'G': G, 'm': m[:, 0]}


def foc_terms_b(n: int, lambd: float, rho: float,
                Ha: np.ndarray | None = None, Ga: np.ndarray | None = None) -> dict:
    """ Calculate the terms of the first order condition for trader A
    :param n: number of Fourier terms
    :param lambd: size of trader B
    :param rho: decay of the propagator
    :param Ha: the Hessian matrix of the cost function of trader A
    :param Ga: cross-term multiplier of trader A
    :return: dictionary contaiining matrices H,G and vector m
    """

    pi, exp = np.pi, np.exp

    # Constants (later can be precomputed)
    n = np.arange(1, n + 1).reshape([-1, 1])
    N = np.diag(n.reshape(-1)).astype(float)
    n_sq = n ** 2
    i = np.ones(n.shape)
    i_odd = n % 2
    i_mp = i - 2 * i_odd

    m_p_n_odd = (n + n.T) % 2
    msq_nsq = n_sq - n_sq.T
    with np.errstate(divide='ignore', invalid='ignore'):
        M = 2 * np.where(m_p_n_odd, n_sq / msq_nsq, 0)

    D = np.diag((pi * n / (rho ** 2 + (n * pi) ** 2)).reshape(-1))
    h = D @ (i - exp(-rho) * i_mp)

    # Calluate the quadratic coefficients (Hessian)
    if Ha is None:
        H = -rho ** 2 * (D @ i @ h.T + h @ i.T @ D) + pi * rho * N @ D + pi * (N @ M.T @ D + D @ M @ N)
    else:
        H = lambd ** 2 * Ha

    # Calculate the linear coefficients for the Cross-Term (B)
    if Ga is None:
        G = lambd * (- rho ** 2 * h @ i.T @ D + pi * (0.5 * rho * N @ D + N @ M.T @ D))
    else:
        G = Ga

    # The constant term
    m = lambd * (lambd * D @ (exp(-rho) * i - i_mp) - (1 + lambd) * h)

    return {'H': H, 'G': G, 'm': m[:, 0]}


def solve_equilibrium(n: int, lambd: float, rho: float) -> dict:
    """ Find the equilibrium Fourier parameters for A and B analytically
    """

    # Calculate the first order correction terms
    terms_a = foc_terms_a(n, lambd, rho)
    Ha = terms_a['H']
    Ga = terms_a['G']
    ma = terms_a['m']

    terms_b = foc_terms_b(n, lambd, rho, Ha, Ga)
    Hb = terms_b['H']
    Gb = terms_b['G']
    mb = terms_b['m']

    # Stack matrices to form the combined linear system
    H = np.block([[Ha, Ga], [Gb, Hb]])
    f = np.block([ma, mb])

    # Solve for the stacked coefficients vector
    x = np.linalg.solve(H, -f)

    return {'H': H, 'Ha': Ha, 'Ga': Ga, 'ma': ma, 'Hb': Hb, 'Gb': Gb, 'mb': mb,
            'a': x[:n], 'b': x[n:]}


