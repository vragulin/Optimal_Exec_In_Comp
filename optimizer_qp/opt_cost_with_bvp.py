"""
Calculate the best response of a to a passive adversary b using several alternative
approaches (bvp, exact integration of the approx function, analytic formula)
and ensure that the results are consistent.
V. Ragulin 28-Sep-2024
"""
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad, solve_bvp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Tuple, Callable, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'cost_function')))
import fourier as fr
from cost_function_approx import cost_fn_a_approx
import qp_solvers as qp

# Global Parameters
N = 50  # number of Fourier terms
kappa = 10  # permanent impact
lambd = 1  # temporary impact
xi_a = 0  # risk aversion of a -- not sure if it works with the approximate cost function
sigma = 3  # volatility of the stock -- not sure if it works with the approximate cost function
N_PLOT_POINTS = 100  # number of points for plotting

# Select the cost function
QUAD, APPROX = range(2)
COST_FUNCTION = APPROX

# Specify which solver to choose if Fourier Approx is chosen
SCIPY, QP = range(2)
APPROX_SOLVER = QP

# Global variables
b_coeffs = None  # It will be estimated when needed

# Whether to calculate the exact solution using the BVP solver
CALC_EXACT = True


def b_func(t, kappa, lambd) -> float:
    return np.sinh(sigma * t) / np.sinh(sigma)
    # return t


def b_dot_func(t, kappa, lambd) -> float:
    return sigma * np.cosh(sigma * t) / np.sinh(sigma)
    # return 1


def b_dbl_dot_func(t, kappa, lambd) -> float:
    return sigma * sigma * np.sinh(sigma * t) / np.sinh(sigma)
    # return 0


def b_vector(t):
    return (
        b_func(t, kappa, lambd),
        b_dot_func(t, kappa, lambd),
        b_dbl_dot_func(t, kappa, lambd)
    )


def compute_cost_as_integral(a_func, a_dot_func, kappa, lambd, verbose=False):
    def integrand_temp(t):
        _a_dot = a_dot_func(t, kappa, lambd)
        _b_dot = b_dot_func(t, kappa, lambd)
        return (_a_dot + lambd * _b_dot) * _a_dot

    def integrand_perm(t):
        _a = a_func(t, kappa, lambd)
        _b = b_func(t, kappa, lambd)
        _a_dot = a_dot_func(t, kappa, lambd)
        return kappa * (_a + lambd * _b) * _a_dot

    temp_cost = quad(integrand_temp, 0, 1)[0]
    perm_cost = quad(integrand_perm, 0, 1)[0]

    if verbose:
        print("Exact temp_cost: ", temp_cost)
        print("Exact perm_cost: ", perm_cost)

    return temp_cost + perm_cost


def cost_function(a_coeffs):
    if COST_FUNCTION == QUAD:
        return cost_function_exact(a_coeffs)
    elif COST_FUNCTION == APPROX:
        return cost_function_approx(a_coeffs)
    else:
        raise NotImplementedError(f"Unknown cost function")


def cost_function_exact(a_coeffs):
    def a_func(t, kappa, lambd):
        return fr.reconstruct_from_sin(t, a_coeffs) + t

    def a_dot_func(t, kappa, lambd):
        return fr.reconstruct_deriv_from_sin(t, a_coeffs) + 1

    return compute_cost_as_integral(a_func, a_dot_func, kappa, lambd)


def cost_function_approx(a_coeffs):
    global b_coeffs
    if b_coeffs is None:
        b_coeffs = fr.sin_coeff(lambda t: b_func(t, kappa, lambd) - t, N)

    return cost_fn_a_approx(a_coeffs, b_coeffs, kappa, lambd)


def plot_curves(init_guess, opt_coeffs, exact_solution) -> dict:
    """ Plot curves and and calc stats """
    t_values = np.linspace(0, 1, N_PLOT_POINTS)

    init_curve = [fr.reconstruct_from_sin(t, init_guess) + t for t in t_values]
    opt_curve = [fr.reconstruct_from_sin(t, opt_coeffs) + t for t in t_values]
    b_curve = [b_func(t, kappa, lambd) for t in t_values]

    # Plot initial guess and optimized functions
    plt.figure(figsize=(10, 5))

    # plt.plot(t_values, init_curve, label='Initial guess', color='blue')
    plt.plot(t_values, opt_curve, label='Optimal approx a(t)', color='red', linewidth=2)
    if exact_solution is not None:
        plt.plot(t_values, exact_solution, label="Optimal exact a(t)", color="green")
    plt.plot(t_values, b_curve, label="Passive adversary b(t)", color="blue", linestyle="dashed")
    plt.suptitle(f'Best Response to a Passive Adversary')
    plt.title(f'Adversary trading λ={lambd} units, Permanent Impact κ={kappa}', fontsize=11)
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate stats
    if exact_solution is not None:
        diff_approx = opt_curve - exact_solution
        max_diff = np.max(np.abs(diff_approx))
        l2_diff = norm(diff_approx)
    else:
        max_diff, l2_diff = None, None

    return {'max': max_diff, 'L2': l2_diff}


# Define the system of differential equations with exogenous b(t)
def equations(t, y, xi_a):
    a, a_prime = y
    b, b_prime, b_dbl_prime = b_vector(t)
    a_double_prime = -(lambd / 2) * (b_dbl_prime + kappa * b_prime) + xi_a * sigma ** 2 * a
    return np.vstack((a_prime, a_double_prime))


# Boundary conditionss
def boundary_conditions(ya, yb):
    return np.array([ya[0], yb[0] - 1])


def solve_min_cost(cost_func: Callable, init_guess: np.ndarray) -> Tuple[np.ndarray, Any]:
    """ Solve for the optimal cost function using the solver specified """
    if COST_FUNCTION == QUAD or APPROX_SOLVER == SCIPY:
        print("Using SciPy solver")
        res = minimize(cost_func, init_guess)
        return res.x, res
    else:
        print("Using QP solver")
        b_coeff = fr.sin_coeff(lambda t: b_func(t, kappa, lambd) - t, N)
        return qp.min_cost_A_qp(b_coeff, kappa, lambd)


if __name__ == "__main__":
    # Initial guess for a_coeffs
    initial_guess = np.zeros(N)
    initial_cost = cost_function(initial_guess)
    print(f"Initial a_coeffs = {np.round(initial_guess, 3)}")
    print(f"Initial guess cost = {initial_cost:.4f}\n")

    # Minimize the cost function
    start = time.time()
    a_coeff_opt, res = solve_min_cost(cost_function, initial_guess)
    print(f"optimization time = {(time.time() - start):.4f}s")

    # Compute the cost with optimized coefficients
    optimized_cost = cost_function(a_coeff_opt)

    print(f"Optimized a_coeffs = {np.round(a_coeff_opt, 3)}")
    print(f"Optimized cost = {optimized_cost:.4f}")

    # Find the exact solution and plot curves
    if CALC_EXACT:
        t = np.linspace(0, 1, N_PLOT_POINTS)
        y_init = np.zeros((2, t.size))
        sol = solve_bvp(lambda _t, y: equations(_t, y, xi_a), boundary_conditions, t, y_init)
        stats = plot_curves(initial_guess, a_coeff_opt, sol.y[0])
        print(f"Approx - Exact Distance: L2 = {stats['L2']:.4f}, max = {stats['max']:.4f}")
    else:
        stats = plot_curves(initial_guess, a_coeff_opts, None)
