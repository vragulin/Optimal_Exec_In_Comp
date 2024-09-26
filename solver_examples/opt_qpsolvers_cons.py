""" Minimize the cost function using the qpsolvers package.
    This test version only optimizes for Trader A
    V. Ragulin, 24-Sep-2024
"""
import os
import sys
import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import minimize
from typing import List, Tuple, Any
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
from cost_function_approx import cost_fn_a_approx
from sampling import sample_sine_wave
import fourier as fr

# *** Global Parameters ***
ABS_TOL = 1e-6
N = 40
T_SAMPLE_PER_SEMI_WAVE = 1

# Constraints
CONS_OVERBUYING = None  # 1
CONS_SHORTSELLING = None  # -0.5

# Parameters used to set up a test case
DECAY = 0.95
DO_QP = True
DO_SCI = True

t_sample = None


def minimize_cons_qpsolvers(b_n: List[float], kappa: float, lambd: float,
                            cons: dict | None = None, abs_tol: float = 1e-6) -> np.ndarray:
    """ Minimize the cost function using the qpsolvers package.
    :param a_n:
    :param b_n:
    :param kappa:
    :param lambd:
    :param cons:
    :param abs_tol:
    :return:
    """
    n_coeffs = len(b_n)
    H = np.zeros((n_coeffs, n_coeffs))
    f = np.zeros(n_coeffs)

    pi = np.pi

    # Define constants (we can pre-compute them if need speed)
    n = np.arange(1, n_coeffs + 1)
    n_sq = (n ** 2)
    n_odd_idx = (n % 2)
    m_p_n_odd = (n[:, None] + n[None, :]) % 2
    mn = n[:, None] @ n[None, :]
    msq_nsq = n_sq[:, None] - n_sq[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(m_p_n_odd, -mn / msq_nsq, 0)

    # The Hessian matrix
    H = pi * pi * np.diag(n_sq)

    # Build f as the sum of 4 terms.
    # The impact of the first term is zero
    # The impact of the second term:
    f2 = pi * pi / 2 * lambd * n_sq * b_n

    # The impact of the third term:
    f3 = - 2 * kappa * lambd / pi * n_odd_idx / n

    # The impact of the fourth term:

    f4 = 2 * kappa * lambd * M @ b_n

    # Add all terms to f
    f = f2 + f3 + f4

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=SOLVER)

    return a_n_opt


def minimize_cons_scipy(b_n: np.ndarray, kappa: float, lambd: float,
                        cons: dict | None = None, abs_tol: float = 1e-6) -> Tuple[np.ndarray, Any]:
    """ Minimize the cost function using scipy.optimize.minimize
    :param a_n:
    :param b_n:
    :param kappa:
    :param lambd:
    :param cons:
    :param abs_tol:
    :return:
    """

    # Define the objective function for minimize
    def objective(x: np.ndarray) -> float:
        return cost_fn_a_approx(x, b_n, kappa, lambd)

    # Initial guess for a_n
    initial_guess = np.zeros_like(b_n)

    # Constraint function
    def constraint_function(coeff):
        # Sample points
        global t_sample
        if t_sample is None:
            t_sample = sample_sine_wave(list(range(1, len(coeff) + 1)), T_SAMPLE_PER_SEMI_WAVE)

        def val_func(t):
            return fr.reconstruct_from_sin(t, coeff) + t

        vals = np.array([val_func(t) for t in t_sample])

        # Overbuying constraint
        if (ubound := cons.get('overbuying', None)) is not None:
            cons_ubound = ubound - vals
        else:
            cons_ubound = None

        # Short Selling  constraint
        if (lbound := cons.get('short_selling', None)) is not None:
            cons_lbound = vals - lbound
        else:
            cons_lbound = None

        res = [x for x in [cons_ubound, cons_lbound] if x is not None]
        if len(res) > 0:
            return np.concatenate(res)
        else:
            return 1

    # Define the constraint
    constraints = {
        'type': 'ineq',
        'fun': lambda coeff: constraint_function(coeff)
    }

    # Call the minimize function
    result = minimize(objective, initial_guess, constraints=constraints, tol=abs_tol)

    return result.x, result


if __name__ == "__main__":
    # Example usage
    #
    a_n = np.zeros(N)
    b_n = np.random.rand(N)  # [np.cos(2 * i) * np.exp(-DECAY * i) for i in range(N_TERMS)]
    kappa = 10
    lambd = 20
    SOLVER = 'daqp'  # 'quadprog'

    print(f"Initial a_n: {a_n}")
    print(f"Initial b_n: {b_n}")

    # Solve using qpsolvers
    if DO_QP:
        start_time = time.time()
        a_n_opt_qp = minimize_cons_qpsolvers(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        end_time = time.time()
        print(f"Optimal a_n using qpsolvers: {a_n_opt_qp}")
        print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (qpsolvers): {cost_fn_a_approx(a_n_opt_qp, b_n, kappa, lambd):.4f}")
        print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")

    # Solve using scipy.optimize.minimize
    if DO_SCI:
        start_time = time.time()
        a_n_opt_sci, _ = minimize_cons_scipy(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        end_time = time.time()
        print("Optimal a_n using scipy:", a_n_opt_sci)
        print(f"scipy time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (scipy): {cost_fn_a_approx(a_n_opt_sci, b_n, kappa, lambd):.4f}")
        print(f"scipy time taken: {end_time - start_time:.4f} seconds")

    np.testing.assert_allclose(a_n_opt_sci, a_n_opt_qp, atol=2e-3), \
        f"Failed for \na_n_opt_qp={a_n_opt_qp}\n a_n_opt_sci={a_n_opt_sci}\n lambd={lambd}, kappa={kappa}, N={N}"

    print("\n************\nTest passed!\n************")
