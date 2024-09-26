"""
Test the QPSolvers solver with constraints.
V. Ragulin, 25-Sep-2024
"""
import os
import sys
import pytest as pt
import numpy as np
import time
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..\..', 'cost_function')))
import cost_function_approx as ca
import fourier as fr
from solver_examples.opt_qpsolvers_cons import minimize_cons_qpsolvers, minimize_cons_scipy
from sampling import sample_sine_wave
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..\..', 'optimizer')))
import trading_funcs as tf

# *** Global Parameters ***
ABS_TOL = 1e-6
DO_QP = True
DO_SCI = True
SOLVER = 'daqp'  # 'quadprog'


@pt.mark.skip("Already tested")
@pt.mark.parametrize("N", [10])
@pt.mark.parametrize("kappa", [0.1])
@pt.mark.parametrize("lambd", [5])
def test_no_cons_dict_passed(N, kappa, lambd):
    """ Running functions with no constraints dictionary passed.
        It shoud use "None" by default and run an unconstrained optimization.
    """
    a_n = np.zeros(N)
    b_n = np.random.rand(N)

    cons_overbuying = None  # 1
    cons_short_selling = None  # -0.5

    cons_dict = {'overbuying': cons_overbuying, 'short_selling': cons_short_selling}

    print(f"Initial a_n: {a_n}")
    print(f"Initial b_n: {b_n}")

    # Solve using qpsolvers
    if DO_QP:
        start_time = time.time()
        a_n_opt_qp = minimize_cons_qpsolvers(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        end_time = time.time()
        print(f"Optimal a_n using qpsolvers: {a_n_opt_qp}")
        print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (qpsolvers): {ca.cost_fn_a_approx(a_n_opt_qp, b_n, kappa, lambd):.4f}")

    # Solve using scipy.optimize.minimize
    if DO_SCI:
        start_time = time.time()
        a_n_opt_sci = minimize_cons_scipy(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        end_time = time.time()
        print("Optimal a_n using scipy:", a_n_opt_sci)
        print(f"scipy time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (scipy): {ca.cost_fn_a_approx(a_n_opt_sci, b_n, kappa, lambd):.4f}")

    np.testing.assert_allclose(a_n_opt_sci, a_n_opt_qp, atol=2e-3), \
        f"Failed for \na_n_opt_qp={a_n_opt_qp}\n a_n_opt_sci={a_n_opt_sci}\n lambd={lambd}, kappa={kappa}, N={N}"


@pt.mark.skip("Already tested")
@pt.mark.parametrize("N", [10])
@pt.mark.parametrize("kappa", [0.1])
@pt.mark.parametrize("lambd", [5])
@pt.mark.parametrize("b_arr_n", [1, 0, 0])
def test_none_cons_dict_passed(N, kappa, lambd, b_arr_n):
    """ Running functions with constraints dictionary passed containing None limits for all constraints.
        The program should run an unconstrained optimization.
    """
    """ Running functions with no constraints dictionary passed.
        It shoud use "None" by default and run an unconstrained optimization.
    """

    def is_ndarray_with_size_N(var, N):
        return isinstance(var, np.ndarray) and (var.size == N)

    a_n = np.zeros(N)
    b_n = np.array(b_arr_n)  # np.random.rand(N)

    cons_overbuying = None  # 1
    cons_short_selling = None  # -0.5

    cons_dict = {'overbuying': cons_overbuying, 'short_selling': cons_short_selling}

    print(f"Initial a_n: {a_n}")
    print(f"Initial b_n: {b_n}")

    # Solve using qpsolvers
    if DO_QP:
        start_time = time.time()
        a_n_opt_qp = minimize_cons_qpsolvers(a_n, b_n, kappa, lambd, cons=cons_dict, abs_tol=ABS_TOL)
        end_time = time.time()
        print(f"Optimal a_n using qpsolvers: {a_n_opt_qp}")
        print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (qpsolvers): {ca.cost_fn_a_approx(a_n_opt_qp, b_n, kappa, lambd):.4f}")
        assert is_ndarray_with_size_N(a_n_opt_qp, N), "QP output is not a numpy array with size N"

    # Solve using scipy.optimize.minimize
    if DO_SCI:
        start_time = time.time()
        a_n_opt_sci = minimize_cons_scipy(a_n, b_n, kappa, lambd, cons=cons_dict, abs_tol=ABS_TOL)
        end_time = time.time()
        print("Optimal a_n using scipy:", a_n_opt_sci)
        print(f"scipy time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (scipy): {ca.cost_fn_a_approx(a_n_opt_sci, b_n, kappa, lambd):.4f}")
        assert is_ndarray_with_size_N(a_n_opt_sci, N), "SciPy output is not a numpy array with size N"

    np.testing.assert_allclose(a_n_opt_sci, a_n_opt_qp, atol=2e-3), \
        f"Failed for \na_n_opt_qp={a_n_opt_qp}\n a_n_opt_sci={a_n_opt_sci}\n lambd={lambd}, kappa={kappa}, N={N}"


# @pt.mark.skip("Already tested")
def test_overbuying_constraint_scipy_when_cons_not_binding():
    """ Check that overbuying constraint is working.  Pass parameters for b so that it's optimal for A
        to violate the overbuying constraint.  Make sure that both solvers reach the same solution.
    """
    kappa = 10
    lambd = 10
    N = 10
    py_tol = 1e-2
    PLOT_CHARTS = True

    # Let be trade a risk-neutral strategy
    b_n = np.zeros(N)

    # Deine the constraint function
    overbuy = 7
    cons = {'overbuying': overbuy}

    a_n_opt, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)

    def is_ndarray_with_size_N(var, N):
        return isinstance(var, np.ndarray) and (var.size == N)

    assert is_ndarray_with_size_N(a_n_opt, N), "SciPy output is not a numpy array with size N"

    # Expected values from Mathematica closed from solution
    exp_max_pos_t = 13.0 / 25.0
    exp_max_pos = 169.0 / 25.0
    exp_cost = -427.0 / 3.0

    # Check that the constraint is satisfied
    T_SAMPLE_PER_SEMI_WAVE = 100
    t_sample = sample_sine_wave(list(range(1, N + 1)), T_SAMPLE_PER_SEMI_WAVE)
    vals = [fr.reconstruct_from_sin(t, res.x) + t for t in t_sample]

    if PLOT_CHARTS:
        plt.plot(t_sample, vals, label="Optimal obj function");
        plt.plot(t_sample, np.ones(len(t_sample)) * overbuy, label="Overbuy limit")
        plt.title("Estimated a(t) vs. the Overbuy Constraint")
        plt.grid()
        plt.legend()
        plt.show()

    max_idx = np.argmax(vals)
    max_pos_t = t_sample[max_idx]
    max_pos = vals[max_idx]
    assert max_pos <= overbuy, f"Overbuying constraint violated.  Max value {max_pos} > {overbuy}"
    assert pt.approx(exp_max_pos_t, rel=py_tol) == max_pos_t, f"Expected max pos t {exp_max_pos_t}, got {max_pos_t}"
    assert pt.approx(exp_max_pos, rel=py_tol) == max_pos, f"Expected max value at{exp_max_pos}, got {max_pos}"

    # Check that max_value matches
    min_cost = ca.cost_fn_a_approx(a_n_opt, b_n, kappa, lambd)
    assert pt.approx(min_cost, rel=py_tol) == exp_cost, f"Expected cost {exp_cost}, got {min_cost}"


@pt.mark.skip("Already tested")
def test_scipy_short_selling_constraint_not_binding1():
    """ Check that scipy gives the right answer if there is a sort selling constraint, and it's not binding
        Start with a trivial example where A trades an eager strategy and B trades a risk-neutral strategy.
    """
    """ Check that short selling constraint is working when it's not binding
        for a risk-neutral strategy
        """
    kappa = 10
    lambd = 10
    N = 10
    py_tol = 1e-2
    PLOT_CHARTS = True

    # Let be trade a risk-neutral strategy
    b_n = np.zeros(N)

    # Deine the constraint function
    c = 0
    cons = {'short_selling': c}

    a_n_opt, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)

    def is_ndarray_with_size_N(var, N):
        return isinstance(var, np.ndarray) and (var.size == N)

    assert is_ndarray_with_size_N(a_n_opt, N), "SciPy output is not a numpy array with size N"

    # Expected values from Mathematica closed from solution
    exp_cost = -427.0 / 3.0

    # Check that the constraint is satisfied
    T_SAMPLE_PER_SEMI_WAVE = 100
    t_sample = sample_sine_wave(list(range(1, N + 1)), T_SAMPLE_PER_SEMI_WAVE)
    vals = [fr.reconstruct_from_sin(t, res.x) + t for t in t_sample]

    if PLOT_CHARTS:
        plt.plot(t_sample, vals, label="Optimal obj function");
        plt.plot(t_sample, np.ones(len(t_sample)) * c, label="Short selling limit")
        plt.title("Estimated a(t) vs. the Overbuy Constraint")
        plt.grid()
        plt.legend()
        plt.show()

    min_idx = np.argmin(vals[1:])
    min_pos_t = t_sample[min_idx]
    min_pos = vals[min_idx]
    assert min_pos >= c, f"Short selling constraint violated.  Min value {min_pos} < {c}"
    # assert pt.approx(exp_max_pos_t, rel=py_tol) == max_pos_t, f"Expected max pos t {exp_max_pos_t}, got {max_pos_t}"
    # assert pt.approx(exp_max_pos, rel=py_tol) == max_pos, f"Expected max value at{exp_max_pos}, got {max_pos}"

    # Check that max_value matches
    min_cost = ca.cost_fn_a_approx(a_n_opt, b_n, kappa, lambd)
    assert pt.approx(min_cost, rel=py_tol) == exp_cost, f"Expected cost {exp_cost}, got {min_cost}"

@pt.mark.skip("Already tested")
def test_scipy_short_selling_constraint_not_binding2():
    """ Check that scipy gives the right answer if there is a sort selling constraint, and it's not binding
        An exaple where it's optimal for A to go short, but not to the limit.
    """
    """ Check that short selling constraint is working when it's not binding
        for a risk-neutral strategy
        """
    kappa = 0.1
    lambd = 20
    sigma = 10
    N = 20
    py_tol = 1e-2
    PLOT_CHARTS = True

    # Let be trade an eager risk-neutral strategy. Estimate coeffs from a formula
    def b_func(t):
        params = {'sigma': sigma, 'gamma': 1}
        return tf.b_func_eager(t, params) - t

    b_n = fr.sin_coeff(b_func, N)

    if PLOT_CHARTS:
        t_step = np.linspace(0, 1, 100)
        b_vals = [b_func(t) + t for t in t_step]
        b_approx = [fr.reconstruct_from_sin(t, b_n) + t for t in t_step]
        plt.scatter(t_step, b_vals, s=10, label="b(t)", color="red")
        plt.plot(t_step, b_approx, label="b(t)_approx", color="blue")
        plt.legend()
        plt.grid()
        plt.show()

    # Deine the constraint function
    c = -7
    cons = {'short_selling': c}

    a_n_opt, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)

    def is_ndarray_with_size_N(var, N):
        return isinstance(var, np.ndarray) and (var.size == N)

    assert is_ndarray_with_size_N(a_n_opt, N), "SciPy output is not a numpy array with size N"

    # Expected values from Mathematica closed from solution
    exp_uncons_cost = -369.234

    # Check that the constraint is satisfied
    T_SAMPLE_PER_SEMI_WAVE = 100
    t_sample = sample_sine_wave(list(range(1, N + 1)), T_SAMPLE_PER_SEMI_WAVE)
    vals = [fr.reconstruct_from_sin(t, res.x) + t for t in t_sample]

    if PLOT_CHARTS:
        plt.plot(t_sample, vals, label="Optimal obj function");
        plt.plot(t_sample, np.ones(len(t_sample)) * c, label="Short selling limit")
        plt.title("Estimated a(t) vs. the Overbuy Constraint")
        plt.grid()
        plt.legend()
        plt.show()

    min_idx = np.argmin(vals[1:])
    min_pos_t = t_sample[min_idx]
    min_pos = vals[min_idx]
    assert min_pos >= c-py_tol * 10, f"Short selling constraint violated.  Min value {min_pos} < {c}"
    # assert pt.approx(exp_max_pos_t, rel=py_tol) == max_pos_t, f"Expected max pos t {exp_max_pos_t}, got {max_pos_t}"
    # assert pt.approx(exp_max_pos, rel=py_tol) == max_pos, f"Expected max value at{exp_max_pos}, got {max_pos}"

    # Check that max_value matches
    min_cost = ca.cost_fn_a_approx(a_n_opt, b_n, kappa, lambd)
    assert pt.approx(min_cost, rel=py_tol) == exp_uncons_cost, f"Expected cost {exp_cost}, got {min_cost}"

@pt.mark.skip("Already tested")
def test_scipy_both_constraints_not_binding():
    """ Check that scipy gives the right answer if there are both constraints and they are not binding
    """
    kappa = 10
    lambd = 10
    N = 10
    py_tol = 1e-2
    PLOT_CHARTS = True

    # Let be trade a risk-neutral strategy
    b_n = np.zeros(N)

    # Deine the constraint function
    OVERBUYING = 7
    C= 0
    cons = {'short_selling': C, 'overbuying': OVERBUYING}

    a_n_opt, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)

    def is_ndarray_with_size_N(var, N):
        return isinstance(var, np.ndarray) and (var.size == N)

    assert is_ndarray_with_size_N(a_n_opt, N), "SciPy output is not a numpy array with size N"

    # Expected values from Mathematica closed from solution
    exp_cost = -427.0 / 3.0

    # Check that the constraint is satisfied
    T_SAMPLE_PER_SEMI_WAVE = 100
    t_sample = sample_sine_wave(list(range(1, N + 1)), T_SAMPLE_PER_SEMI_WAVE)
    vals = [fr.reconstruct_from_sin(t, res.x) + t for t in t_sample]

    if PLOT_CHARTS:
        plt.plot(t_sample, vals, label="Optimal obj function");
        plt.plot(t_sample, np.ones(len(t_sample)) * C, label="Short selling limit")
        plt.plot(t_sample, np.ones(len(t_sample)) * OVERBUYING, label="Short selling limit")
        plt.title("Estimated a(t) vs. the Overbuy Constraint")
        plt.grid()
        plt.legend()
        plt.show()

    min_idx = np.argmin(vals[1:])
    min_pos_t = t_sample[min_idx]
    min_pos = vals[min_idx]
    assert min_pos >= C, f"Short selling constraint violated.  Min value {min_pos} < {C}"
    # assert pt.approx(exp_max_pos_t, rel=py_tol) == max_pos_t, f"Expected max pos t {exp_max_pos_t}, got {max_pos_t}"
    # assert pt.approx(exp_max_pos, rel=py_tol) == max_pos, f"Expected max value at{exp_max_pos}, got {max_pos}"

    # Check that max_value matches
    min_cost = ca.cost_fn_a_approx(a_n_opt, b_n, kappa, lambd)
    assert pt.approx(min_cost, rel=py_tol) == exp_cost, f"Expected cost {exp_cost}, got {min_cost}"

@pt.mark.skip("Already tested")
def test_scipy_overbuy_binding():
    kappa = 10
    lambd = 10
    N = 10
    py_tol = 1e-2
    PLOT_CHARTS = True

    # Let be trade a risk-neutral strategy
    b_n = np.zeros(N)

    # Deine the constraint function
    OVERBUYING = 5
    C = 0
    cons = {'short_selling': C, 'overbuying': OVERBUYING}

    a_n_opt, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)

    def is_ndarray_with_size_N(var, N):
        return isinstance(var, np.ndarray) and (var.size == N)

    assert is_ndarray_with_size_N(a_n_opt, N), "SciPy output is not a numpy array with size N"

    # Expected values from Mathematica closed from solution
    uncons_cost = -427.0 / 3.0

    # Check that the constraint is satisfied
    T_SAMPLE_PER_SEMI_WAVE = 100
    t_sample = sample_sine_wave(list(range(1, N + 1)), T_SAMPLE_PER_SEMI_WAVE)
    vals = [fr.reconstruct_from_sin(t, res.x) + t for t in t_sample]

    if PLOT_CHARTS:
        plt.plot(t_sample, vals, label="Optimal obj function");
        plt.plot(t_sample, np.ones(len(t_sample)) * C, label="Short selling limit")
        plt.plot(t_sample, np.ones(len(t_sample)) * OVERBUYING, label="Short selling limit")
        plt.title("Estimated a(t) vs. the Overbuy Constraint")
        plt.grid()
        plt.legend()
        plt.show()

    min_idx = np.argmin(vals[1:])
    min_pos_t = t_sample[min_idx]
    min_pos = vals[min_idx]
    assert min_pos >= C, f"Short selling constraint violated.  Min value {min_pos} < {C}"
    # assert pt.approx(exp_max_pos_t, rel=py_tol) == max_pos_t, f"Expected max pos t {exp_max_pos_t}, got {max_pos_t}"
    # assert pt.approx(exp_max_pos, rel=py_tol) == max_pos, f"Expected max value at{exp_max_pos}, got {max_pos}"

    # Check that max_value matches
    min_cost = ca.cost_fn_a_approx(a_n_opt, b_n, kappa, lambd)
    print(f"Constraned Cost = {min_cost}, Unconstrained cost = {uncons_cost}")
    assert min_cost + 5 * py_tol >= uncons_cost, f"Constrained cost {exp_cost} < uncons cost {min_cost}"


def test_scipy_short_selling_binding():
    """ Check that scipy gives the right answer if there is a sort selling constraint, and it's not binding
            An exaple where it's optimal for A to go short, but not to the limit.
        """
    """ Check that short selling constraint is working when it's not binding
        for a risk-neutral strategy
        """
    kappa = 0.1
    lambd = 20
    sigma = 10
    N = 20
    py_tol = 1e-2
    PLOT_CHARTS = True

    # Let be trade an eager risk-neutral strategy. Estimate coeffs from a formula
    def b_func(t):
        params = {'sigma': sigma, 'gamma': 1}
        return tf.b_func_eager(t, params) - t

    b_n = fr.sin_coeff(b_func, N)

    if PLOT_CHARTS:
        t_step = np.linspace(0, 1, 100)
        b_vals = [b_func(t) + t for t in t_step]
        b_approx = [fr.reconstruct_from_sin(t, b_n) + t for t in t_step]
        plt.scatter(t_step, b_vals, s=10, label="b(t)", color="red")
        plt.plot(t_step, b_approx, label="b(t)_approx", color="blue")
        plt.title("b(t) vs. b(t)_approx")
        plt.legend()
        plt.grid()
        plt.show()

    # Deine the constraint function
    OVERBUYING = 10
    C = -3
    cons = {'overbuying': OVERBUYING, 'short_selling': C}

    a_n_opt, res = minimize_cons_scipy(b_n, kappa, lambd, cons=cons, abs_tol=ABS_TOL)

    def is_ndarray_with_size_N(var, N):
        return isinstance(var, np.ndarray) and (var.size == N)

    assert is_ndarray_with_size_N(a_n_opt, N), "SciPy output is not a numpy array with size N"

    # Expected values from Mathematica closed from solution
    uncons_cost = -369.234

    # Check that the constraint is satisfied
    T_SAMPLE_PER_SEMI_WAVE = 3
    t_sample = sample_sine_wave(list(range(1, N + 1)), T_SAMPLE_PER_SEMI_WAVE)
    vals = [fr.reconstruct_from_sin(t, res.x) + t for t in t_sample]
    b_vals1 = [fr.reconstruct_from_sin(t, b_n) + t for t in t_sample]

    if PLOT_CHARTS:
        plt.plot(t_sample, vals, label="Optimal obj function", color='red');
        plt.plot(t_sample, b_vals1, label="b(t)", color='blue');
        plt.scatter(t_sample, np.ones(len(t_sample)) * C, s=10, label="Short selling limit", color='gray')
        plt.scatter(t_sample, np.ones(len(t_sample)) * OVERBUYING, s=10, label="Short selling limit", color='gray')
        plt.title("Optimal trading strategies vs. the constraints")
        plt.grid()
        plt.legend()
        plt.show()

    assert (pmin:= np.min(vals)) >= C - py_tol * 5, f"Short selling constraint violated.  Min value {pmin} < {C}"
    assert (pmax:= np.max(vals)) <= OVERBUYING + py_tol * 5, f"OVERBUY constraint violated.  Min value {pmax} < {C}"

    # Check that the minimum constrained cost is higher than the unconstrained cost
    opt_cons_cost = ca.cost_fn_a_approx(a_n_opt, b_n, kappa, lambd)
    print(f"Constraned Cost = {opt_cons_cost}, Unconstrained cost = {uncons_cost}")
    assert opt_cons_cost + 5 * py_tol >= uncons_cost, f"Constrained cost {uncons_cost} < uncons cost {opt_cons_cost}"


@pt.mark.skip("Not implemented")
def test_overbuying_constraint_answers_equal():
    """ Check that overbuying constraint is working.  Pass parameters for b so that it's optimal for A
        to violate the overbuying constraint.  Make sure that both solvers reach the same solution.
    """
    assert False
