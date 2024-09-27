""" Minimize the cost function using the qpsolvers package.
    This test version only optimizes for Trader A
    V. Ragulin, 24-Sep-2024
    This is a draft version of the code - before refactoring.
    Use the refactored version.
"""
import os
import sys
import numpy as np
from qpsolvers import solve_qp  # https://pypi.org/project/qpsolvers/
from scipy.optimize import minimize
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
from cost_function_approx import cost_fn_a_approx
from sampling import sample_sine_wave
import fourier as fr

# *** Global Parameters ***
# ABS_TOL = 1e-4
# N = 35
T_SAMPLE_PER_SEMI_WAVE = 3
#
# # Constraints
# CONS_OVERBUYING = None  # 1
# CONS_SHORTSELLING = None  # -0.5
#
# # Parameters used to set up a test case
# DECAY = 0.1
# DO_QP = True
# DO_SCI = True
TEST_EXACT_COEFF = False
TEST_CONSTRAINTS = False
np.random.seed(12)
#
t_sample = None


def minimize_cons_qpsolvers(b_n: List[float], kappa: float, lambd: float,
                            cons: dict | None = None, abs_tol: float = 1e-6,
                            solver='daqp') -> Tuple[np.ndarray, Any]:
    """ Minimize the cost function using the qpsolvers package.
        Problem: min x'Px + q'x,
                 s.t. Gx <= h, Ax = b, lb <= x <= ub
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
    P = pi * pi * np.diag(n_sq)

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

    # Build inequality constraints
    # Replication of the Fourier series can be modelled as a product of a Matrix
    # S of sin(n * pi * t_n) and the coeff vector.  For >- constraints, we need to
    # change the sign of both the matrix and the RHS bound.

    # Building the S matrix
    t_sample = sample_sine_wave(list(range(1, n_coeffs + 1)), T_SAMPLE_PER_SEMI_WAVE)
    S = np.sin(pi * np.array(t_sample)[:, None] @ n[None, :])

    # If we have constraints of both sides, we need to duplicate S0 with the opposite sign
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
        G = None
        h = None

    a_n_opt = solve_qp(P, q, G, h, None, None, solver=solver)

    return a_n_opt, None


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


def check_constraints(coeff, cons, t_sample, ABS_TOL):
    def val_func(t):
        return fr.reconstruct_from_sin(t, coeff) + t

    vals = np.array([val_func(t) for t in t_sample])

    # Overbuying constraint
    if (ubound := cons.get('overbuying', None)) is not None:
        cons_ubound = ubound - vals
    else:
        cons_ubound = None

    # Short Selling constraint
    if (lbound := cons.get('short_selling', None)) is not None:
        cons_lbound = vals - lbound
    else:
        cons_lbound = None

    res = [x for x in [cons_ubound, cons_lbound] if x is not None]
    if len(res) > 0:
        return np.all(np.concatenate(res) >= -ABS_TOL)
    else:
        return True


def run_optimization(N, DECAY, kappa, lambd, cons, ABS_TOL, DO_QP, DO_SCI):
    a_n = np.zeros(N)
    b_n = np.random.rand(N) * np.exp(-DECAY * np.arange(N))

    t_sample = sample_sine_wave(list(range(1, N + 1)), T_SAMPLE_PER_SEMI_WAVE)

    qp_time, sci_time = None, None

    if DO_QP:
        start_time_qp = time.time()
        time.sleep(0.0001)  # To avoid pecision issues for small N, where the time is too short to measure
        a_n_opt_qp, _ = minimize_cons_qpsolvers(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)
        end_time_qp = time.time()
        qp_time = end_time_qp - start_time_qp
        # qp_obj = cost_fn_a_approx(a_n_opt_qp, b_n, kappa, lambd)
        is_feasible_qp = check_constraints(a_n_opt_qp, cons, t_sample, ABS_TOL)
        if not is_feasible_qp:
            print("Constraints are not satisfied by the QP solution")
            if TEST_CONSTRAINTS:
                raise ValueError("Constraints are not satisfied by the QP solution")

    if DO_SCI:
        start_time_sci = time.time()
        a_n_opt_sci, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)
        end_time_sci = time.time()
        sci_time = end_time_sci - start_time_sci
        # sci_obj = cost_fn_a_approx(a_n_opt_sci, b_n, kappa, lambd)
        is_feasible_sci = check_constraints(a_n_opt_sci, cons, t_sample, ABS_TOL)
        if not is_feasible_sci:
            print("Constraints are not satisfied by the SciPy solution")
            if TEST_CONSTRAINTS:
                raise ValueError("Constraints are not satisfied by the SciPy solution")

    return qp_time, sci_time


def plot_times(N_values, qp_times, sci_times):
    plt.plot(N_values, qp_times, label='QP Solver Time', color='red')
    plt.plot(N_values, sci_times, label='SciPy Solver Time', color='blue', linestyle="dashed")
    plt.xlabel('N')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('QP Solver vs SciPy Solver Time')
    plt.show()

    plt.plot(N_values, qp_times, label='QP Solver Time', color='red')
    plt.plot(N_values, sci_times, label='SciPy Solver Time', color='blue', linestyle="dashed")
    plt.xlabel('N')
    plt.ylabel('Log(Time (seconds))')
    plt.yscale('log')
    plt.legend()
    plt.title('QP Solver vs SciPy Solver Time (Log Scale)')
    plt.show()

    ratio = [sci_times / qp_times for sci_times, qp_times in zip(sci_times, qp_times)]
    plt.plot(N_values, ratio, label='SciPy/QP Time Ratio', color='green')
    plt.xlabel('N')
    plt.ylabel('time ratio')
    plt.legend()
    plt.title('SciPy/QP Time Ratio')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    N_values = [3, 5, 7, 10, 13, 16, 20, 25, 30, 35, 40, 50, 60, 70]  # Example list of N values
    DECAY = 0.15
    kappa = 10
    lambd = 20
    cons = {'overbuying': 1, 'short_selling': 0}
    ABS_TOL = 1e-4
    DO_QP = True
    DO_SCI = True

    qp_times = []
    sci_times = []
    N_TRIES = 10

    for N in N_values:
        qp_total, sci_total = 0, 0
        t_sample = None
        for t in range(N_TRIES):
            print(f"\nRunning for {N}, try {t+1}")
            qp_time, sci_time = run_optimization(N, DECAY, kappa, lambd, cons, ABS_TOL, DO_QP, DO_SCI)
            qp_total += qp_time
            sci_total += sci_time
            print(f"sci: {sci_time}, qp: {qp_time}")
        qp_times.append(qp_total / N_TRIES)
        sci_times.append(sci_total / N_TRIES)

    print("qp: ", qp_times)
    print("sci: ", sci_times)
    plot_times(N_values, qp_times, sci_times)
