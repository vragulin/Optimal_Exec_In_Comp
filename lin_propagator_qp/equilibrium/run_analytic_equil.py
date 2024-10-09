import time

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'matrix_utils')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..')))
from propagator import cost_fn_prop_a_matrix
from est_quad_coeffs import find_coefficients
import prop_cost_to_QP as pcq
from analytic_solution import solve_equilibrium, foc_terms_a, foc_terms_b
import fourier as fr

# -------------------------------------------------------
# Script to run and test the analytic_solution.py functions
# -------------------------------------------------------
# Global parameters
N = 1000  # Dimension of the vector x
lambd = 1
rho = 1
abs_tol = 1e-6

# Presentation settings
N_COEFF_TO_PRINT = 4
N_PLOT_POINTS = 140


def plot_curves(a_coeffs, b_coeffs, lambd, rho, N, **kwargs) -> None:
    """ Plot curves and and calc stats """
    t_values = np.linspace(0, 1, N_PLOT_POINTS)

    a_curve = [fr.reconstruct_from_sin(t, a_coeffs) + t for t in t_values]
    b_curve = [fr.reconstruct_from_sin(t, b_coeffs) + t for t in t_values]

    # Plot initial guess and optimized functions
    plt.title(f'Optimal Trading Strategies, Linear Propagator (OW) Model\n'
              f'λ={lambd}, ρ={rho}, N={N}', fontsize=12)

    # Top chart
    plt.plot(t_values, a_curve, label=r"$a(t)$", color='red')
    plt.scatter(t_values, b_curve, label=r"$b_\lambda(t)$", s=20, color='blue', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Position over time')
    plt.legend()
    plt.grid()

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


if __name__ == "__main__":
    # Find the equilibrium Fourier coefficients
    start = time.time()
    res_solve = solve_equilibrium(N, lambd, rho)
    print(f"Time to solve for Equilibrium = {(time.time() - start):.4f}s\n")

    a_coeff, b_coeff = res_solve['a'], res_solve['b']
    # print(res_solve)
    print(f"Equilibrium Solved: lambda = {lambd}, rho = {rho}")
    print(f"Number of Fourier Coeffs = {N}, printing the first {N_COEFF_TO_PRINT}")
    print("Trader A: ", a_coeff[:N_COEFF_TO_PRINT])
    print("Trader B: ", b_coeff[:N_COEFF_TO_PRINT])
    plot_curves(a_coeff, b_coeff, lambd, rho, N)

    # Check that each strategy is the best response to the other
    print("\nChecking that the strategy of each trader is the best response to the other")
    sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../ad_hoc')))
    from check_equilibrium import CheckEquilibrium, SCIPY, QP

    init_cost, opt_cost, coeffs_opt, coeff_diff = {}, {}, {}, {}
    for trader in ['A', 'B']:

        if trader == 'A':
            c = CheckEquilibrium(rho=rho, lambd=lambd, N=N, b_coeffs=b_coeff)
            initial_guess = a_coeff
        else:
            c = CheckEquilibrium(rho=rho, lambd=1 / lambd, N=N, b_coeffs=a_coeff, use_trader_b_fn=True)
            initial_guess = b_coeff

        init_cost[trader] = c.compute(initial_guess)
        print(f"Initial {trader} coeffs = {np.round(initial_guess[:N_COEFF_TO_PRINT], 3)}")
        print(f"Initial {trader} guess cost = {init_cost[trader]:.4f}\n")

        # Minimize the cost function
        coeffs_opt[trader], res = c.solve_min_cost(initial_guess, solver=QP, abs_tol=1e-6)

        # Compute the cost with optimized coefficients
        opt_cost[trader] = c.compute(coeffs_opt[trader])

        print(f"Optimized {trader} coeffs = {np.round(coeffs_opt[trader][:N_COEFF_TO_PRINT], 3)}")
        print(f"Optimized cost = {opt_cost[trader]:.4f}")

        coeff_diff[trader] = np.linalg.norm(coeffs_opt[trader] - initial_guess) / np.sqrt(N)
        print(f"Norm Diff {trader}: ", coeff_diff[trader])

        # Find the exact solution and plot curves
        c.plot_curves(initial_guess, coeffs_opt[trader], trader=trader)

    if max(coeff_diff.values()) > abs_tol \
            or abs(opt_cost["A"] - init_cost["A"]) > abs_tol \
            or abs(opt_cost["B"] - init_cost["B"]) > abs_tol:
        print("\nTest failed!  Analytical solution is not optimal")
    else:
        print("\nSuccess! Analytical solution is optimal for both traders")
