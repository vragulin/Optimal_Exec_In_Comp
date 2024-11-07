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
from opt_cost_prop_qp import CostFunction, SCIPY, QP
from analytic_solution import solve_equilibrium, foc_terms_a, foc_terms_b
import fourier as fr

# -------------------------------------------------------
# Script to run and test the analytic_solution.py functions
# -------------------------------------------------------
# Global parameters
N = 5  # Dimension of the vector x
lambd = 5
rho = 0.0001
abs_tol = 1e-6

# Presentation settings
N_COEFF_TO_PRINT = 8
N_PLOT_POINTS = 1000  # Works beest when N is a multiple of 2 * N_PLOT_POINTS
DROP_LAST_SINES = None  # None or integer.  Use to reduce the 'wiggle'
RUN_TESTS = True  # Run tests and plot test results


def ow_size(t, rho):
    """" The optimal size according to the Obizhaeva-Wang (2013) model
    """
    block = 1 / (2 + rho)
    continous_speed = rho / (2 + rho)
    match t:
        case 0:
            return 0
        case 1:
            return 1
        case _:
            return block + continous_speed * t


def plot_curves(a_coeffs, b_coeffs, lambd, rho, N, n_points, **kwargs) -> None:
    """ Plot curves and and calc stats """

    if DROP_LAST_SINES:
        a_coeffs_used = a_coeffs[:-DROP_LAST_SINES]
        b_coeffs_used = b_coeffs[:-DROP_LAST_SINES]
    else:
        a_coeffs_used, b_coeffs_used = a_coeffs, b_coeffs

    t_values = np.linspace(0, 1, n_points + 1)
    a_curve = [fr.reconstruct_from_sin(t, a_coeffs_used) + t for t in t_values]
    b_curve = [fr.reconstruct_from_sin(t, b_coeffs_used) + t for t in t_values]
    ow_curve = [ow_size(t, rho) for t in t_values]

    # Plot initial guess and optimized functions
    title_str = (f'Optimal Trading Strategies, Linear Propagator (OW) Model\n'
                 f'λ={lambd}, ρ={rho}, N={N}')
    if DROP_LAST_SINES:
        title_str += f", dropped Last {DROP_LAST_SINES} Sines"

    plt.title(title_str, fontsize=10)

    # Top chart
    plt.plot(t_values, a_curve, label=r"$a(t)$", color='red')
    # plt.scatter(t_values, b_curve, label=r"$b_\lambda(t)$", s=20, color='green', alpha=0.5)
    plt.plot(t_values, b_curve, label=r"$b_\lambda(t)$", color='blue', alpha=0.5)
    plt.plot(t_values, ow_curve, label=r"OW(t)", color='green', linestyle='dotted')
    plt.xlabel('Time')
    plt.ylabel('Position over time')
    plt.legend()
    plt.grid()

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def plot_spectral_density(a_coeff, b_coeff, lambd, rho, N):
    """ Plot the spectral density of the Fourier coefficients """
    n = np.arange(1, N + 1)
    a_spec = np.abs(a_coeff)
    b_spec = np.abs(b_coeff)

    plt.figure(figsize=(10, 5))
    plt.title(f'Spectral Density of Fourier Coefficients, λ={lambd}, ρ={rho}, N={N}', fontsize=12)

    plt.plot(n, a_spec, label=r"$|a_n|$", color='red', alpha=0.5)
    plt.plot(n, b_spec, label=r"$|b_n|$", color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('Spectral Density')
    # plt.yscale('log')
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
    cond = np.linalg.cond(res_solve['H'])
    print(f"Condition Number of the FOC Matrix = {cond:.2f}")
    # a_coeff = np.zeros(N)
    # a_coeff[0] = 1
    # b_coeff = a_coeff / 2
    plot_curves(a_coeff, b_coeff, lambd, rho, N, n_points=N_PLOT_POINTS)

    if RUN_TESTS:
        # Plot spectral densities of a and b
        plot_spectral_density(a_coeff, b_coeff, lambd, rho, N)

        # Check that each strategy is the best response to the other
        sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../ad_hoc')))
        from check_equilibrium import CheckEquilibrium

        print("\nChecking that the strategy of each trader is the best response to the other")
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
            c.plot_curves(initial_guess, coeffs_opt[trader], n_points=N_PLOT_POINTS, trader=trader)

        if max(coeff_diff.values()) > abs_tol \
                or abs(opt_cost["A"] - init_cost["A"]) > abs_tol \
                or abs(opt_cost["B"] - init_cost["B"]) > abs_tol:
            print("\nTest failed!  Analytical solution is not optimal")
        else:
            print("\nSuccess! Analytical solution is optimal for both traders")
