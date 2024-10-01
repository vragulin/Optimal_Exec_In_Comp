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
from sampling import sample_sine_wave

# Global Parameters
N = 20  # number of Fourier terms
KAPPA = 10  # permanent impact
LAMBD = 20  # temporary impact
SIGMA = 3  # volatility of the stock -- not sure if it works with the approximate cost function

# Parameters used to set up a test case
np.random.seed(12)
DECAY = 0.15

# Constraints
OVERBUYING = 2
C = -1
CONS = {'overbuying': OVERBUYING, 'short_selling': C}
T_SAMPLE_PER_SEMI_WAVE = 3  # number of points to sample constraints

# Specify which solver to choose if Fourier Approx is chosen
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

        # Attributes used for optimization
        self.t_sample = None

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

    @property
    def b_func_params(self) -> dict:
        return self._b_func_params

    @b_func_params.setter
    def b_func_params(self, params: dict):
        # If input does not contain values for lambda and kappa, add them
        # This way we don't have to specify them in the b_func_params explicitly
        self._b_func_params = params
        self._b_func_params.setdefault('kappa', self.kappa)
        self._b_func_params.setdefault('lambd', self.lambd)

    def compute(self, a_coeffs):
        return cost_fn_a_approx(a_coeffs, self.b_coeffs, self.kappa, self.lambd)

    def __str__(self):
        s = "Cost Function: \n"
        s += f"N = {self.N}, kappa = {self.kappa}, lambd" + f" = {self.lambd}\n"
        s += f"b_coeffs = {self.b_coeffs}\n"
        s += f"b_func = {self.b_func}\n"
        s += f"b_func_params = {self.b_func_params}\n"
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

    def solve_min_cost(self, init_guess: np.ndarray, **kwargs) -> Tuple[np.ndarray, Any]:
        """ Solve for the optimal cost function using the solver specified """

        cons = kwargs.get('constraints', None)
        if APPROX_SOLVER == SCIPY:
            print("Using SciPy solver")
            constraints = self.build_scipy_constraints(cons, T_SAMPLE_PER_SEMI_WAVE)
            res = minimize(self.compute, init_guess, constraints=constraints)
            return res.x, res
        elif APPROX_SOLVER == QP:
            print("Using QP solver")
            return qp.min_cost_A_qp(self.b_coeffs, self.kappa, self.lambd, cons=cons)
        else:
            raise ValueError("Unknown solver")

    def build_scipy_constraints(self, cons: dict, t_sample_per_semi_wave: int = 3) -> dict:

        # Constraint function
        def constraint_function(coeff):
            # Sample points
            if self.t_sample is None:
                self.t_sample = sample_sine_wave(list(range(1, len(coeff) + 1)),
                                                 t_sample_per_semi_wave)

            def val_func(t):
                return fr.reconstruct_from_sin(t, coeff) + t

            vals = np.array([val_func(t) for t in self.t_sample])

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

        return constraints


def main():
    # Set up the environment
    # -------------------------------------------------------------------------
    # Example 1: set up with a function from the trading_funcs_qp module
    # ... or it can be any other function defined in this or another module.
    # c = CostFunction(kappa=KAPPA, lambd=LAMBD, N=N,
    #                  b_func=tf.risk_averse, b_func_params={'sigma': SIGMA})
    # -------------------------------------------------------------------------
    # Example 2 - Another example - b following an eager strategy
    # c = CostFunction(kappa=0.1, lambd=20, N=N,
    #                  b_func=tf.eager, b_func_params={'sigma': SIGMA})
    # -------------------------------------------------------------------------
    # Example 3 - Another example - b following an equilibrium strategy
    c = CostFunction(kappa=0.1, lambd=20, N=N,
                     b_func=tf.equil_2trader, b_func_params={'trader_a': True})
    # -------------------------------------------------------------------------
    # Example 4 - Initalize b(t) with Fourier coefficients
    # b_coeffs = np.random.rand(N) * np.exp(-DECAY * np.arange(N))
    # c = CostFunction(kappa=KAPPA, lambd=LAMBD, N=N, b_coeffs=b_coeffs)
    # -------------------------------------------------------------------------
    print(c)
    print(c.b_func(0.5))

    # Initial guess for a_coeffs
    initial_guess = np.zeros(N)
    initial_cost = c.compute(initial_guess)
    print(f"Initial a_coeffs = {np.round(initial_guess, 3)}")
    print(f"Initial guess cost = {initial_cost:.4f}\n")

    # Minimize the cost function
    start = time.time()
    a_coeffs_opt, res = c.solve_min_cost(initial_guess, constraints=CONS)
    print(f"optimization time = {(time.time() - start):.4f}s")

    # Compute the cost with optimized coefficients
    optimized_cost = c.compute(a_coeffs_opt)

    print(f"Optimized a_coeffs = {np.round(a_coeffs_opt, 3)}")
    print(f"Optimized cost = {optimized_cost:.4f}")

    # Find the exact solution and plot curves
    c.plot_curves(initial_guess, a_coeffs_opt)


if __name__ == "__main__":
    main()
