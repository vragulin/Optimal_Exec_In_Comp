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
import prop_cost_to_QP as c2qp

# *** Global Defaults, they can be overwritten with kwargs ***
DEFAULT_SOLVER = 'daqp'
T_SAMPLE_PER_SEMI_WAVE = 3


# ToDo: eliminate the qpsolvers interface and call the daqp directly.  This will allow to
#   handle the 2-way constraints bl < Gx < bh, rather than naively duplicating the S matrix
#   to convert the leq and geq constraints to leq only.


def precalc_obj_func_constants(n_coeffs: int, precomp: dict, **kwargs) -> tuple:
    """ Calculate the constants used in the objective function
        And updates values in the precomp dictionary.
    """

    return (None,)


def build_obj_func_qp(b_n: np.ndarray, rho: float, lambd: float,
                      precomp: dict | None = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ Build the matrix and vector pair P,q that describe the objective function
    """

    return c2qp.cost_QP_params(b_n=b_n, lambd=lambd, rho=rho)


def reg_adjustment(b_n: np.ndarray, precomp: dict, reg_params: dict,
                   ) -> np.ndarray | float:
    """ Add regularization to the Hessian matrix P
    """
    p, pi, sin = precomp, np.pi, np.sin
    if reg_params is None:
            return 0

    if p == {}:
        t_sample = reg_params['t_sample']
        assert t_sample is not None, "The t_sample parameter is required for the regularization"
        n = np.arange(1, len(b_n) + 1)
        n_sq = n ** 2
        N_sq = np.diag(n_sq)
        n_tau = t_sample[:, None] @ n[None, :]
        sin_n_tau = sin(pi * n_tau)
        a_ddot_mult = -pi *pi * sin_n_tau @ N_sq
        W = a_ddot_mult.T @ a_ddot_mult
        W /= 0.5 * len(t_sample) * len(b_n) ** 3 # Rescale for # of terms, sample size
    else:
        W = p['W']
    return W


def min_cost_A_qp(b_n: np.ndarray, rho: float, lambd: float, abs_tol: float = 1e-6,
                  precomp: dict | None = None, **kwargs) -> Tuple[np.ndarray, Any]:
    """ Minimize the cost function using the qpsolvers package.
        Problem: min x'Px + q'x,
                 s.t. Gx <= h, Ax = b, lb <= x <= ub
    :param b_n:

    :param rho:
    :param lambd:
    :param abs_tol:
    :param precomp: a dictionary of precomputed matrices or other variablees
    :return:
    """
    # Build the martix and vector describing the objective function
    P, q = build_obj_func_qp(b_n=b_n, rho=rho, lambd=lambd, precomp=precomp, **kwargs)

    # Build matrix and vector descrbing inequality constraints
    reg_params = kwargs.get('reg_params')
    factor = kwargs.get('factor', 0)
    dP = reg_adjustment(b_n, precomp, reg_params) * factor

    solver = kwargs.get('solver', DEFAULT_SOLVER)
    a_n_opt = solve_qp(P=P + dP, q=q, solver=solver)

    res_dict = {'x': a_n_opt, 'P': P, 'q': q, 'dP': dP}
    return a_n_opt, res_dict


def min_cost_B_qp(a_n: np.ndarray, rho: float, lambd: float, abs_tol: float = 1e-6,
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
    return min_cost_A_qp(a_n, rho, 1 / lambd, abs_tol, precomp, **kwargs)
