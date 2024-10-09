"""  Confirm that the solution found by an equilibrium script is, indeed, an equilibrium.
    V. Ragulin, 10/05/2024
"""
import os
import sys
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import fourier as fr
from propagator import cost_fn_prop_a_approx, cost_fn_prop_b_approx, prop_price_impact_approx
import qp_prop_solvers as qp
from opt_cost_prop_qp import CostFunction, SCIPY, QP
from tte_optimizer_prop_2drv_qp import State
import config_prop_qp as cfg

# Globol Settings
ITERATION_NUMBER = -1  # The number of the iteration to check, or -1 for the last one
FILE_NAME = 'tte_sim_data_l20_r2_o0_01_n20_g0.01_d0.0.pkl'  # The file with the equilibrium states
N_PLOT_POINTS = 100  # number of points for plotting


class CheckEquilibrium(CostFunction):

    def plot_curves(self, init_guess: np.ndarray, opt_coeffs: np.ndarray,
                    n_points: int, **kwargs) -> None:
        """ Plot curves and and calc stats """
        t_values = np.linspace(0, 1, n_points)

        init_curve = [fr.reconstruct_from_sin(t, init_guess) + t for t in t_values]
        opt_curve = [fr.reconstruct_from_sin(t, opt_coeffs) + t for t in t_values]
        if self.b_func is not None:
            b_curve = [self.b_func(t, **self.b_func_params) for t in t_values]
        else:
            b_curve = [fr.reconstruct_from_sin(t, self.b_coeffs) + t for t in t_values]

        # Plot initial guess and optimized functions
        fig, axs = plt.subplots(2, 1, figsize=(7, 10))
        if 'trader' in kwargs:
            trader_str = f"Trader={kwargs['trader']}, "
        else:
            trader_str = ""

        plt.suptitle(f'Best Response to a Passive Adversary, Linear Propagator Model\n'
                     f'{trader_str}N={self.N}, λ={self.lambd}, ρ={self.rho}', fontsize=14)

        # Top chart
        ax = axs[0]
        ax.scatter(t_values, init_curve, label='Initial guess', s=20, color='green', alpha=0.5)
        ax.plot(t_values, opt_curve, label="Optimal a(t)", color='red', linewidth=2)
        ax.plot(t_values, b_curve, label=r"Passive adversary $b_\lambda(t)$", color="blue", linestyle="dashed")
        ax.set_title(f'Trading Schedules a(t), b(t)', fontsize=11)
        ax.set_xlabel('Time')
        ax.set_ylabel('a(t), b(t)')
        ax.legend()
        ax.grid()

        # Bottom chart
        ax = axs[1]
        ax.set_title(f'Difference between init_guess and optimized', fontsize=11)
        # ax.plot(t_values, a_dot, label="a'(t)", color='red')
        # ax.plot(t_values, b_dot, label="b'(t)", color='blue', linestyle="dashed")
        diff = [o - i for o, i in zip(opt_curve, init_curve)]
        ax.plot(t_values, diff, label="opt-init", color='green')
        ax.set_xlabel('Time')
        # ax.set_ylabel("a'(t), b'(t), dP(0,t)")
        ax.set_ylabel("opt-init")
        ax.legend()
        ax.grid()

        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()


def load_pickled_results(file_name: str) -> dict:
    """ Load the pickled results from the file
    """
    data_dir = os.path.join(current_dir, '..', cfg.SIM_RESULTS_DIR)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'The directory {data_dir} does not exist')

    file_full_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_full_path):
        raise FileNotFoundError(f'The directory {file_full_path} does not exist')

    with open(file_full_path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    data = load_pickled_results(FILE_NAME)

    # Pull out the iteratino of interest
    s = data['iter_hist'][ITERATION_NUMBER]

    # ----------------------------------------------
    # Check that the solution if optimal for A
    # ----------------------------------------------
    c = CheckEquilibrium(rho=s.rho, lambd=s.lambd, N=s.N, b_coeffs=s.b_coeff)
    print(c)

    # Initial guess for a_coeffs
    initial_guess = s.a_coeff
    initial_cost = c.compute(initial_guess)
    print(f"Initial a_coeffs = {np.round(initial_guess, 3)}")
    print(f"Initial guess cost = {initial_cost:.4f}\n")

    # Minimize the cost function
    start = time.time()
    a_coeffs_opt, res = c.solve_min_cost(initial_guess, solver=QP, abs_tol=1e-6)
    print(f"optimization time = {(time.time() - start):.4f}s")

    # Compute the cost with optimized coefficients
    optimized_cost = c.compute(a_coeffs_opt)

    print(f"Optimized a_coeffs = {np.round(a_coeffs_opt, 3)}")
    print(f"Optimized cost = {optimized_cost:.4f}")

    print("Norm Diff: ", np.linalg.norm(a_coeffs_opt - initial_guess) / np.sqrt(len(a_coeffs_opt)))

    # Find the exact solution and plot curves
    c.plot_curves(initial_guess, a_coeffs_opt, N_PLOT_POINTS, trader="A")

    # ----------------------------------------------
    # Check that the solution if optimal for B
    # ----------------------------------------------
    c = CheckEquilibrium(rho=s.rho, lambd=1 / s.lambd, N=s.N, b_coeffs=s.a_coeff,
                         use_trader_b_fn=True)
    print(c)

    # Initial guess for a_coeffs
    initial_guess = s.b_coeff
    initial_cost = c.compute(initial_guess)
    print(f"Initial b_coeffs = {np.round(initial_guess, 3)}")
    print(f"Initial guess cost = {initial_cost:.4f}\n")

    # Minimize the cost function
    start = time.time()
    b_coeffs_opt, res = c.solve_min_cost(initial_guess, solver=QP, abs_tol=1e-6)
    print(f"optimization time = {(time.time() - start):.4f}s")

    # Compute the cost with optimized coefficients
    optimized_cost = c.compute(b_coeffs_opt)

    print(f"Optimized b_coeffs = {np.round(b_coeffs_opt, 3)}")
    print(f"Optimized cost = {optimized_cost:.4f}")

    print("Norm Diff: ", np.linalg.norm(b_coeffs_opt - initial_guess) / np.sqrt(len(b_coeffs_opt)))

    # Find the exact solution and plot curves
    c.plot_curves(initial_guess, b_coeffs_opt, N_PLOT_POINTS, trader="B")

    print('Done')


if __name__ == "__main__":
    main()
