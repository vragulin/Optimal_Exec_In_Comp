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
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'optimizer_qp')))
import fourier as fr
from propagator import cost_fn_prop_a_approx, prop_price_impact_approx
import trading_funcs_qp as tf

# Global Parameters
N = 200  # number of Fourier terms
RHO = 10  # propagator decay
LAMBD = 5  # size of trader B
SIGMA = 3  # volatility of the stock -- not sure if it works with the approximate cost function

# Specify which solver to use for optimization
SCIPY, QP = range(2)
APPROX_SOLVER = SCIPY  # QP not yet implemented

# Plot settings
N_PLOT_POINTS = 100  # number of points for plotting

# Random coefficients test parameters
np.random.seed(12)
DECAY = 0.15  # decay of the Fourier coefficients for the random coefficient test


class CostFunction:
    def __init__(self, **kwargs):
        self.rho = kwargs.get('rho', 0)
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
        return cost_fn_prop_a_approx(a_coeffs, self.b_coeffs, self.lambd, self.rho)

    def __str__(self):
        s = "Cost Function: \n"
        s += f"N = {self.N}, rho = {self.rho}, lambd" + f" = {self.lambd}\n"
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
        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        plt.suptitle(f'Best Response to a Passive Adversary, Linear Propagator Model\n'
                     f'N={self.N}, λ={self.lambd}, ρ={self.rho}', fontsize=14)

        # Top chart
        ax = axs[0]
        ax.plot(t_values, init_curve, label='Initial guess', color='grey', linestyle="dotted")
        ax.plot(t_values, opt_curve, label="Optimal a(t)", color='red', linewidth=2)
        ax.plot(t_values, b_curve, label=r"Passive adversary $b_\lambda(t)$", color="blue", linestyle="dashed")
        ax.set_title(f'Trading Schedules a(t), b(t)', fontsize=11)
        ax.set_xlabel('Time')
        ax.set_ylabel('a(t), b(t)')
        ax.legend()
        ax.grid()

        # Bottom chart
        ax = axs[1]
        a_dot = [fr.reconstruct_deriv_from_sin(t, opt_coeffs) + 1 for t in t_values]
        b_dot = [self.lambd * (fr.reconstruct_deriv_from_sin(t, self.b_coeffs) + 1) for t in t_values]
        dp = [prop_price_impact_approx(t, opt_coeffs, self.b_coeffs, self.lambd, self.rho) for t in t_values]

        ax.set_title(f'Temporary Price Impact', fontsize=11)
        # ax.plot(t_values, a_dot, label="a'(t)", color='red')
        # ax.plot(t_values, b_dot, label="b'(t)", color='blue', linestyle="dashed")
        ax.plot(t_values, dp, label="dP(0,t)", color='green')
        ax.set_xlabel('Time')
        # ax.set_ylabel("a'(t), b'(t), dP(0,t)")
        ax.set_ylabel("dP(0,t)")
        ax.legend()
        ax.grid()

        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()


def solve_min_cost(c: CostFunction, init_guess: np.ndarray) -> Tuple[np.ndarray, Any]:
    """ Solve for the optimal cost function using the solver specified """
    if APPROX_SOLVER == SCIPY:
        print("Using SciPy solver")
        res = minimize(c.compute, init_guess)
        return res.x, res
    elif APPROX_SOLVER == QP:
        raise NotImplementedError("Quadratic programming solver not yet implemented")
    else:
        raise ValueError("Unknown solver")


def main():
    # ToDo - plot price impact over time, maybe on a separate plot, also show function derivs
    #  Make it a 2x1 grid
    # Set up the environment  with a function from the trading_funcs_qp module
    # or it can be any other function defined in this or another module.
    # -------------------------------------------------------------------------
    # Example 1: Trader B following a risk-neutral strategy
    # c = CostFunction(rho=RHO, lambd=LAMBD, N=N, b_func=lambda t: t)
    # -------------------------------------------------------------------------
    # Example 2: Trader B following a risk-averse strategy
    # c = CostFunction(rho=RHO, lambd=LAMBD, N=N,
    #                  b_func=tf.risk_averse, b_func_params={'sigma': SIGMA})
    # -------------------------------------------------------------------------
    # Example 3: Trader B following an eager strategy
    # c = CostFunction(rho=RHO, lambd=LAMBD, N=N,
    #                  b_func=tf.eager, b_func_params={'sigma': SIGMA})
    # -------------------------------------------------------------------------
    # Example 4: Trader B following an equilibrium strategy
    c = CostFunction(rho=RHO, lambd=LAMBD, N=N,
                     b_func=tf.equil_2trader, b_func_params={'trader_a': True, 'kappa': 10})
    # -------------------------------------------------------------------------
    # Example 5: Trader B following a random strat defined by a vector of random Fourier coeffs.
    # b_coeffs = np.random.rand(N) * np.exp(-DECAY * np.arange(N))
    # c = CostFunction(rho=RHO, lambd=LAMBD, N=N, b_coeffs=b_coeffs)
    # -------------------------------------------------------------------------
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