""" Minimize the cost function using the qpsolvers package.
    This test version only optimizes for Trader A
    V. Ragulin, 24-Sep-2024
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
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'optimizer_qp')))
from cost_function_approx import cost_fn_a_approx
from sampling import sample_sine_wave
import fourier as fr
import trading_funcs_qp as tf

# *** Global Parameters ***
ABS_TOL = 1e-4
N = 20
T_SAMPLE_PER_SEMI_WAVE = 3

# Constraints
CONS_OVERBUYING = None  # 1
CONS_SHORTSELLING = None  # -0.5

# Parameters used to set up a test case
DECAY = 0.15
DO_QP = True
DO_SCI = True
TEST_EXACT_COEFF = False
N_POINTS_PLOT = 100
np.random.seed(12)

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


def plot_strategies(a_qp: np.array, a_sci: np.ndarray, b: np.ndarray, cons: dict):
    """ Plot optimal strategies for A and B"""
    t_plot = np.linspace(0, 1, N_POINTS_PLOT)

    a_qp_vals = [fr.reconstruct_from_sin(t, a_qp) + t for t in t_plot]
    a_sci_vals = [fr.reconstruct_from_sin(t, a_sci) + t for t in t_plot]
    b_vals = [fr.reconstruct_from_sin(t, b) + t for t in t_plot]
    plt.plot(t_plot, a_qp_vals, label="a_qp(t)", color="red", alpha=0.6)
    plt.scatter(t_plot, a_sci_vals, s=10, label="a_sci(t)", color="crimson", alpha=0.6)
    plt.plot(t_plot, b_vals, color="blue", label="b(t)", linestyle="dashed")
    plt.plot(t_plot, np.ones(N_POINTS_PLOT) * cons['overbuying'], color="grey", label="overbuy cons.", alpha=0.6)
    plt.plot(t_plot, np.ones(N_POINTS_PLOT) * cons['short_selling'], color="grey", label="short cons", alpha=0.6)
    plt.title(f"Optimal response for A given B\nN terms = {len(b)}")
    # plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    #
    a_n = np.zeros(N)
    # b_n = np.random.rand(N) * np.exp(-DECAY * np.arange(N))
    kappa = 0.1
    lambd = 20
    b_n = fr.sin_coeff(lambda t: tf.equil_2trader(t, kappa=kappa, lambd=lambd, trader_a=True)-t, N)
    SOLVER = 'daqp'  # 'quadprog'

    # print(f"Initial a_n: {a_n}")
    print(f"Initial b_n: {b_n}")

    OVERBUYING = 2
    C = -1
    cons = {'overbuying': OVERBUYING, 'short_selling': C}


    def check_constraints(coeff):
        # Check that all constraints have been satisfied
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
            return np.all(np.concatenate(res) >= -ABS_TOL)
        else:
            return True


    # Solve using qpsolvers
    if DO_QP:
        start_time = time.time()
        a_n_opt_qp, _ = minimize_cons_qpsolvers(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)
        end_time = time.time()
        print(f"Optimal a_n using qpsolvers: {a_n_opt_qp}")
        qp_time = end_time - start_time
        qp_obj = cost_fn_a_approx(a_n_opt_qp, b_n, kappa, lambd)
        print(f"qpsolvers time taken: {qp_time:.4f} seconds")
        print(f"Objective function value (qpsolvers): {qp_obj:.4f}")
        is_feasible_qp = check_constraints(a_n_opt_qp)
        if is_feasible_qp:
            print("Constraints are satisfied by the QP solution")
        else:
            print("Constraints are not satisfied by the QP solution")

    # Solve using scipy.optimize.minimize
    if DO_SCI:
        start_time = time.time()
        a_n_opt_sci, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)
        sci_obj = cost_fn_a_approx(a_n_opt_sci, b_n, kappa, lambd)
        end_time = time.time()
        print("Optimal a_n using scipy:", a_n_opt_sci)
        sci_time = end_time - start_time
        print(f"scipy time taken: {sci_time:.4f} seconds")
        print(f"Objective function value (scipy): {sci_obj:.4f}")
        is_feasible_sci = check_constraints(a_n_opt_sci)
        if is_feasible_sci:
            print("Constraints are satisfied by the scipy solution")
        else:
            print("Constraints are not satisfied by the scipy solution")

    if (qp_obj - sci_obj) >= ABS_TOL * abs(qp_obj):
        print("QP solution is worse than scipy solution by more than ABS_TOL fraction")

    if TEST_EXACT_COEFF:
        np.testing.assert_allclose(a_n_opt_sci, a_n_opt_qp, atol=2e-3), \
            f"Failed for \na_n_opt_qp={a_n_opt_qp}\n a_n_opt_sci={a_n_opt_sci}\n lambd={lambd}, kappa={kappa}, N={N}"

    print("\n************\nTest passed!" +
          f"\nqp is {sci_time / qp_time:.2f} times faster than scipy" +
          "\n************")

    plot_strategies(a_n_opt_qp, a_n_opt_sci, b_n, cons)
