import os
import sys
import pytest
import numpy as np
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..\..', 'cost_function')))
import cost_function_approx as ca
from solver_examples.opt_qpsolvers import minimize_qpsolvers, minimize_scipy

ABS_TOL = 1e-6
DO_QP = True
DO_SCI = True
SOLVER = 'daqp'  # 'quadprog'


@pytest.mark.parametrize("N", [10, 20, 40])
@pytest.mark.parametrize("kappa", [0.1, 5])
@pytest.mark.parametrize("lambd", [1, 10, 20])
def test_minimization(N, kappa, lambd):
    a_n = np.zeros(N)
    b_n = np.random.rand(N)

    print(f"Initial a_n: {a_n}")
    print(f"Initial b_n: {b_n}")

    # Solve using qpsolvers
    if DO_QP:
        start_time = time.time()
        a_n_opt_qp = minimize_qpsolvers(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        end_time = time.time()
        print(f"Optimal a_n using qpsolvers: {a_n_opt_qp}")
        print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (qpsolvers): {ca.cost_fn_a_approx(a_n_opt_qp, b_n, kappa, lambd):.4f}")

    # Solve using scipy.optimize.minimize
    if DO_SCI:
        start_time = time.time()
        a_n_opt_sci = minimize_scipy(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        end_time = time.time()
        print("Optimal a_n using scipy:", a_n_opt_sci)
        print(f"scipy time taken: {end_time - start_time:.4f} seconds")
        print(f"Objective function value (scipy): {ca.cost_fn_a_approx(a_n_opt_sci, b_n, kappa, lambd):.4f}")

    np.testing.assert_allclose(a_n_opt_sci, a_n_opt_qp, atol=2e-3), \
        f"Failed for \na_n_opt_qp={a_n_opt_qp}\n a_n_opt_sci={a_n_opt_sci}\n lambd={lambd}, kappa={kappa}, N={N}"
