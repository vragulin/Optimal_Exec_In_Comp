""" Minimize the cost function using the qpsolvers package.
    Implements the Short Selling and Overbuying constraints.
    V. Ragulin, 24-Sep-2024
"""
import os
import sys
import time
from typing import List

import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import minimize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
import cost_function_approx as ca

# *** Global Parameters ***
ABS_TOL = 1e-6
N = 40
DECAY = 0.95
DO_QP = True
DO_SCI = True
np.random.seed(123)


def minimize_qpsolvers_1(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                         abs_tol: float = 1e-6) -> np.ndarray:
    n_coeffs = len(a_n)
    H = np.zeros((n_coeffs, n_coeffs))
    f = np.zeros(n_coeffs)

    pi = np.pi

    H[0, 0] = pi * pi
    f[0] = lambd * (pi * pi * b_n[0] / 2 - 2 * kappa / pi)

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=SOLVER)

    return a_n_opt


def minimize_qpsolvers_2(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                         abs_tol: float = 1e-6) -> np.ndarray:
    n_coeffs = len(a_n)
    H = np.zeros((n_coeffs, n_coeffs))
    f = np.zeros(n_coeffs)

    pi = np.pi
    n = np.arange(1, n_coeffs + 1)

    # The Hessian matrix
    H[0, 0] = pi * pi
    H[1, 1] = 4 * pi * pi

    # Build f as the sum of 4 terms.
    # The impact of the first term is zero
    # The impact of the second term:
    n_sq = (n ** 2)
    f2 = pi * pi / 2 * lambd * n_sq * b_n

    # The impact of the third term:
    n_odd_idx = (n % 2)
    f3 = - 2 * kappa * lambd / pi * n_odd_idx / n

    # The impact of the fourth term:
    m_p_n_odd = (n[:, None] + n[None, :]) % 2
    mn = n[:, None] @ n[None, :]
    msq_nsq = n_sq[:, None] - n_sq[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(m_p_n_odd, -mn / msq_nsq, 0)

    f4 = 2 * kappa * lambd * M @ b_n

    # Add all terms to f
    f = f2 + f3 + f4

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=SOLVER)

    return a_n_opt


def minimize_qpsolvers(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                       abs_tol: float = 1e-6, solver='daqp') -> np.ndarray:
    n_coeffs = len(a_n)
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

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=solver)

    return a_n_opt


def minimize_scipy(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                   abs_tol: float = 1e-6) -> np.ndarray:
    a_n = np.array(a_n, dtype=np.float64)
    b_n = np.array(b_n, dtype=np.float64)

    # Define the objective function for minimize
    def objective(a_n: np.ndarray) -> float:
        return ca.cost_fn_a_approx(a_n, b_n, kappa, lambd)

    # Initial guess for a_n
    initial_guess = np.zeros_like(a_n)

    # Call the minimize function
    result = minimize(objective, initial_guess, method='BFGS', tol=abs_tol)

    return result.x


# Example usage

if __name__ == '__main__':
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
        match N:
            case -1:  # Legacy test function for case 1
                a_n_opt_qp = minimize_qpsolvers_1(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
            case -2:  # Legacy test function for case 2
                a_n_opt_qp = minimize_qpsolvers_2(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
            case _:  # This should always be used now
                a_n_opt_qp = minimize_qpsolvers(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL, solver=SOLVER)
        end_time = time.time()
        print(f"Optimal a_n using qpsolvers: {a_n_opt_qp}")
        print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (qpsolvers): {ca.cost_fn_a_approx(a_n_opt_qp, b_n, kappa, lambd):.4f}")
        print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")

    # Solve using scipy.optimize.minimize
    if DO_SCI:
        start_time = time.time()
        a_n_opt_sci = minimize_scipy(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        end_time = time.time()
        print("Optimal a_n using scipy:", a_n_opt_sci)
        print(f"scipy time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (scipy): {ca.cost_fn_a_approx(a_n_opt_sci, b_n, kappa, lambd):.4f}")
        print(f"scipy time taken: {end_time - start_time:.4f} seconds")

    np.testing.assert_allclose(a_n_opt_sci, a_n_opt_qp, atol=2e-3), \
        f"Failed for \na_n_opt_qp={a_n_opt_qp}\n a_n_opt_sci={a_n_opt_sci}\n lambd={lambd}, kappa={kappa}, N={N}"

    print("\n************\nTest passed!\n************")