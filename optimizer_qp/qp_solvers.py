""" Minimize the Cost Function using a fast QP solver, with or without constraints
    V. Ragulin
"""
import os
import sys
import numpy as np
from qpsolvers import solve_qp  # https://pypi.org/project/qpsolvers/
from typing import List, Tuple, Any
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
from cost_function_approx import cost_fn_a_approx
from sampling import sample_sine_wave
import fourier as fr

# *** Global Defaults, they can be overwritten with kwargs ***
DEFAULT_SOLVER = 'daqp'
T_SAMPLE_PER_SEMI_WAVE = 3


# ToDo: eliminate the qpsolvers interface and call the daqp directly.  This will allow to
#   handle the 2-way constraints bl < Gx < bh, rather than naively duplicating the S matrix
#   to convert the leq and geq constraints to leq only.


def check_constraints(cons: dict) -> bool:
    """ Check that all specified constraints are supported
    """
    SUPPORTED_CONST = ('overbuying', 'short_selling')
    if cons is not None:
        for k in cons.keys():
            if k not in SUPPORTED_CONST:
                raise ValueError(f"Constraint type: {k} not implemented")
    return True


def precalc_obj_func_constants(n_coeffs: int, precomp: dict, **kwargs) -> tuple:
    """ Calculate the constants used in the objective function
        And updates values in the precomp dictionary.
    """
    p, pi = precomp, np.pi
    if p == {}:
        n = np.arange(1, n_coeffs + 1)
        n_sq = (n ** 2)
        n_odd_idx = (n % 2)
        m_p_n_odd = (n[:, None] + n[None, :]) % 2
        mn = n[:, None] @ n[None, :]
        msq_nsq = n_sq[:, None] - n_sq[None, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.where(m_p_n_odd, -mn / msq_nsq, 0)

        # The Hessian matrix
        P = pi * pi * np.diag(n_sq)
        p['n_sq'], p['n_odd_idx'], p['M'], p['P'] = n_sq, n_odd_idx, M, P
    else:
        n_sq, n_odd_idx, M, P = p['n_sq'], p['n_odd_idx'], p['M'], p['P']

    return n_sq, n_odd_idx, P, M


def precalc_constraint_constants(n_coeffs: int, precomp: dict, **kwargs) -> tuple:
    p, pi = precomp, np.pi  # Aliases
    n = np.arange(1, n_coeffs + 1)
    if p == {}:
        t_sample_per_semi_wave = kwargs.get('t_sample_per_semi_wave', T_SAMPLE_PER_SEMI_WAVE)
        t_sample = sample_sine_wave(list(range(1, n_coeffs + 1)), t_sample_per_semi_wave)
        S = np.sin(pi * np.array(t_sample)[:, None] @ n[None, :])
        p['t_sample'], p['S'] = t_sample, S
    else:
        t_sample, S = p['t_sample'], p['S']
    return t_sample, S


def build_obj_func_qp(b_n: np.ndarray, kappa: float, lambd: float,
                      precomp: dict | None = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ Build the matrix and vector pair P,q that describe the objective function
    """
    n_coeffs = len(b_n)
    pi = np.pi

    # Retreive precomputed constants (if values provided), or calculate them
    n = np.arange(1, n_coeffs + 1)
    n_sq, n_odd_idx, P, M = precalc_obj_func_constants(n_coeffs, precomp or {}, **kwargs)

    # Build f as the sum of 4 terms.
    # The impact of the first term is zero
    # The impact of the second term:
    q2 = pi * pi / 2 * lambd * n_sq * b_n

    # The impact of the third term:
    q3 = - 2 * kappa * lambd / pi * n_odd_idx / n

    # The impact of the fourth term:

    q4 = 2 * kappa * lambd * M @ b_n

    # Add all terms to f
    q = q2 + q3 + q4

    return P, q


def build_ineq_cons_qp(b_n: np.ndarray, kappa: float, lambd: float,
                       cons: dict | None = None,
                       precomp: dict | None = None, **kwargs) -> tuple:
    # Replication of the Fourier series can be modelled as a product of a Matrix
    # S of sin(n * pi * t_n) and the coeff vector.  For >- constraints, we need to
    # change the sign of both the matrix and the RHS bound.

    # If we have constraints of both sides, we need to duplicate S with the opposite sign
    if cons is not None:
        # Retreive precomputed constants (if values provided), or calculate them
        t_sample, S = precalc_constraint_constants(len(b_n), precomp or {}, **kwargs)

        check_constraints(cons)
        overbuying = cons.get('overbuying', None)
        short_selling = cons.get('short_selling', None)

        v_overbuy = np.ones(S.shape[0]) * overbuying - t_sample \
            if overbuying is not None else None
        v_short_sell = np.ones(S.shape[0]) * short_selling - t_sample \
            if short_selling is not None else None

        if overbuying is not None and short_selling is not None:
            G = np.vstack((S, -S))
            h = np.concatenate((v_overbuy, -v_short_sell))
        elif overbuying is not None:
            G = S
            h = v_overbuy
        elif short_selling is not None:
            G = -S0

            h = -v_short_sell
        else:
            G, h = None, None
    else:
        G, h = None, None
    return G, h


def min_cost_A_qp(b_n: np.ndarray, kappa: float, lambd: float,
                  cons: dict | None = None, abs_tol: float = 1e-6,
                  precomp: dict | None = None, **kwargs) -> Tuple[np.ndarray, Any]:
    """ Minimize the cost function using the qpsolvers package.
        Problem: min x'Px + q'x,
                 s.t. Gx <= h, Ax = b, lb <= x <= ub
    :param b_n:
    :param kappa:
    :param lambd:
    :param cons:  - dictionary of constraints
    :param abs_tol:
    :param precomp: a dictionary of precomputed matrices or other variablees
    :return:
    """
    # Build the martix and vector describing the objective function
    P, q = build_obj_func_qp(b_n=b_n, kappa=kappa, lambd=lambd, precomp=precomp, **kwargs)
    # Build matrix and vector descrbing inequality constraints
    G, h = build_ineq_cons_qp(b_n=b_n, kappa=kappa, lambd=lambd, cons=cons,
                              precomp=precomp, **kwargs)

    solver = kwargs.get('solver', DEFAULT_SOLVER)
    a_n_opt = solve_qp(P, q, G, h, None, None,
                       solver=solver)

    res_dict = {'x': a_n_opt, 'P': P, 'q': q, 'G': G, 'h': h}
    return a_n_opt, res_dict


def min_cost_B_qp(a_n: np.ndarray, kappa: float, lambd: float,
                  cons: dict | None = None, abs_tol: float = 1e-6,
                  precomp: dict | None = None, **kwargs) -> Tuple[np.ndarray, Any]:
    """
    For minimizing the cost for trader B, rather than writing a seprate specification we can
    use the symmetry between the two traders.

    Say A is trading x(t) and B is trading y(t) scaled by lambda.  Denote:
    C_a(x, y, lambda) the cost of A
    C_b(x, y, lambda) the cost of B

    Then:
    C_b(x, y, lambda) = lambda^2 * C_a(y, x, 1/lambda)

    So, you can calculate and optimize C_b by changing the variables in the Cost_a function.
    Since the function does not calculate the cost, we don't even need to rescale the result
    """
    return min_cost_A_qp(a_n, kappa, 1 / lambd, cons, abs_tol, precomp, **kwargs)
