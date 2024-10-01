"""
Miminize cost function given b
"""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Tuple, Callable, Any
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'cost_function')))
import fourier as fr
from cost_function_approx import cost_fn_a_approx
import qp_solvers as qp
import trading_funcs as tf

# Global Parameters
N = 50  # number of Fourier terms
KAPPA = 10  # permanent impact
LAMBD = 1  # temporary impact
SIGMA = 3  # volatility of the stock -- not sure if it works with the approximate cost function

# Specify which solver to use for optimization
SCIPY, QP = range(2)
APPROX_SOLVER = QP

# Plot settings
N_PLOT_POINTS = 100  # number of points for plotting


class CostFunction:
    def __init__(self, **kwargs):
        self.kappa = kwargs.get('kappa', 0)
        self.lambd = kwargs.get('lambd', 1)

        # The trajectory of an adversary can be specified
        # as a function of time or by it Fourier coefficients
        self.b_func = kwargs.get('b_func', None)
        self.b_func_params = kwargs.get('b_func_params', {})

        if 'b_coeffs' in kwargs and 'N' in kwargs:  # If both are provided, use N as the length of b_coeffs
            self._N = kwargs['N']
            self._b_coeffs = kwargs['b_coeffs'][:self._N]
        elif 'b_coeffs' in kwargs:
            self._b_coeffs = kwargs['b_coeffs']
            self._N = len(self._b_coeffs)
        else:
            self._b_coeffs = None
            self._N = kwargs.get('N', 10)

        if self._b_coeffs is None and self.b_func is None:
            raise ValueError("Either b_coeffs or b_func must be provided")

    @property
    def b_coeffs(self) -> np.ndarray:
        if self._b_coeffs is None:
            self._b_coeffs = fr.sin_coeff(lambda t: self.b_func(t, **self.b_func_params) - t, N)
            # b_func = partial(self.b_func, **self.b_func_params) \
            #     if self.b_func_params is not None else self.b_func
            # self._b_coeffs = fr.sin_coeff(b_func, self._N)
        return self._b_coeffs

    @property
    def N(self) -> int:
        return self._N

    def compute(self, a_coeffs):
        return cost_fn_a_approx(a_coeffs, self.b_coeffs, self.kappa, self.lambd)

    def __str__(self):
        s = "Cost Function: \n"
        s += f"N = {self.N}, kappa = {self.kappa}, lambd" + f" = {self.lambd}\n"
        s += f"b_coeffs = {self.b_coeffs}\n"
        s += f"b_func = {self.b_func}\n"
        return s

    def plot_curves(self, init_guess: np.ndarray, opt_coeffs: np.ndarray) -> None:
        """ Plot curves and and calc stats """
        t_values = np.linspace(0, 1, N_PLOT_POINTS)

        init_curve = [fr.reconstruct_from_sin(t, init_guess) + t for t in t_values]
        opt_curve = [fr.reconstruct_from_sin(t, opt_coeffs) + t for t in t_values]
        if self.b_func is not None:
            b_curve = [self.b_func(t, **self.b_func_params) for t in t_values]
        else:
            b_curve = [fr.reconstruct_from_sin(t, self.b_coeffs) + t for t in t_values]

        # Plot initial guess and optimized functions
        plt.figure(figsize=(10, 5))

        plt.plot(t_values, init_curve, label='Initial guess', color='grey', linestyle="dotted")
        plt.plot(t_values, opt_curve, label="Optimal a(t)", color='red', linewidth=2)
        plt.plot(t_values, b_curve, label=r"Passive adversary $b_\lambda(t)$", color="blue", linestyle="dashed")
        plt.suptitle(f'Best Response to a Passive Adversary')
        plt.title(f'Adversary trading λ={self.lambd} units, Permanent Impact κ={self.kappa}', fontsize=11)
        plt.legend()
        plt.grid()
        plt.show()


def solve_min_cost(c: CostFunction, init_guess: np.ndarray) -> Tuple[np.ndarray, Any]:
    """ Solve for the optimal cost function using the solver specified """
    if APPROX_SOLVER == SCIPY:
        print("Using SciPy solver")
        res = minimize(c.compute, init_guess)
        return res.x, res
    elif APPROX_SOLVER == QP:
        print("Using QP solver")
        return qp.min_cost_A_qp(c.b_coeffs, c.kappa, c.lambd)
    else:
        raise ValueError("Unknown solver")


def main():
    # Set up the environment
    c = CostFunction(kappa=KAPPA, lambd=LAMBD, N=N,
                     b_func=tf.risk_averse, b_func_params={'sigma': SIGMA})
    print(c)

    # Initial guess for a_coeffs
    initial_guess = np.zeros(N)
    initial_cost = c.compute(initial_guess)
    print(f"Initial a_coeffs = {np.round(initial_guess, 3)}")
    print(f"Initial guess cost = {initial_cost:.4f}\n")

    # Minimize the cost function
    start = time.time()
    a_coeffs_opt, res = solve_min_cost(c, initial_guess)
    print(f"optimization time = {(time.time() - start):.4f}s")

    # Compute the cost with optimized coefficients
    optimized_cost = c.compute(a_coeffs_opt)

    print(f"Optimized a_coeffs = {np.round(a_coeffs_opt, 3)}")
    print(f"Optimized cost = {optimized_cost:.4f}")

    # Find the exact solution and plot curves
    c.plot_curves(initial_guess, a_coeffs_opt)


if __name__ == "__main__":
    main()
