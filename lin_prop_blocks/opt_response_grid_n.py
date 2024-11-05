"""  Solve for best response within the OW-with-Blocks model
    Run the model for a range of parameters and plot the results as a grid.  Here we change N.
    V. Ragulin - 10/21/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '..', 'cost_function'),
    os.path.join(CURRENT_DIR, '..', 'optimizer_qp')
])

from prop_blocks import SinesBlocks, CostModel
import trading_funcs as tf

# Parameters
N_LIST = [20, 50, 100, 200]  # number of Fourier terms
RHO_LIST = [0.1, 1, 10, 100]  # propagator decay
LAMBDA = 5
FUNC = tf.risk_averse  # tf.risk_neutral tf.eager
FUNC_PARAMS = {}  # {"sigma": 3}  # {}
OTHER_STRAT_CODE = "an OW-Equilibrium"  # "a Risk-Averse"  # "a Risk-Neutral" "an Eager"

# Constants
A, B = range(2)


def other(trader: int) -> int:
    return 1 - trader


def trader_code(trader: int) -> str:
    return 'A' if trader == A else 'B'


def trader_color(trader: int) -> str:
    return 'red' if trader == A else 'blue'


def trader_func_name(trader: int) -> str:
    return 'a(t)' if trader == A else r'$b_\lambda(t)$'


def plot_curves(c: CostModel, ax: Any, **kwargs) -> None:
    n_points = kwargs.get('n_points', 100)
    init_guess = kwargs.get('init_guess', SinesBlocks(N=c.N))
    trader = kwargs.get('trader', A)

    """ Plot curves and and calc stats """
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [init_guess.calc(t) for t in t_values]
    opt_curve = [c.strats[trader].calc(t) for t in t_values]
    other_curve = [c.strats[other(trader)].calc(t) for t in t_values]

    # Plot initial guess and optimized functions

    # Top chart
    ax.plot(t_values, init_curve, label='Initial guess ' + trader_func_name(trader), color='grey', linestyle="dotted")
    ax.plot(t_values, opt_curve, label="Optimal " + trader_func_name(trader),
            color=trader_color(trader), linewidth=2)
    ax.plot(t_values, other_curve, label="Adversary " + trader_func_name(other(trader)),
            color=trader_color(other(trader)), linestyle="dashed")
    ax.set_title(f'N={c.N}, ρ={c.rho}', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{trader_func_name(A)}, {trader_func_name(B)}')
    ax.legend()
    ax.grid()


def plot_curves_grid(rho_list: list[float], func: callable, N_list: list[float],
                     lambd: float = 1, **kwargs) -> None:
    """ Plot curves for a range of parameters """

    trader = kwargs.get('trader', A)
    params = kwargs.get('func_params', {})

    fig, axs = plt.subplots(len(rho_list), len(N_list), figsize=(15, 15))
    title_str = (f'Best Response to {OTHER_STRAT_CODE} Passive Adversary, Linear Propagator Model\n'
                 f'trader={trader_code(trader)}, λ={lambd}')
    if "sigma" in params:
        title_str += f', σ={params["sigma"]}'
    plt.suptitle(title_str, fontsize=14)

    for i, rho in enumerate(rho_list):
        for j, N in enumerate(N_list):
            # Set up the environment  with a function from the trading_funcs_qp module
            # or it can be any other function defined in this or another module.
            # -------------------------------------------------------------------------
            # Example 1: Trader B following a risk-neutral strategy
            # c = CostFunction(rho=RHO, lambd=LAMBD, N=N, b_func=lambda t: t)
            # -------------------------------------------------------------------------
            # Example 2: Trader B following a risk-averse strategy
            # strat_b = SinesBlocks.from_func(func=func, N=N, params=params, lambd=lambd)
            # strats = [SinesBlocks(N=strat_b.N), strat_b]
            # c = CostModel(strats, rho=rho)
            # -------------------------------------------------------------------------
            # Example 3: Trader B following an eager strategy
            # strat_b = SinesBlocks.from_func(func=tf.eager, N=10, params={"sigma": 3}, lambd=5.0)
            # strats = [SinesBlocks(N=strat_b.N), strat_b]
            # c = CostModel(strats, rho=0.1)
            # -------------------------------------------------------------------------
            # Example 4: Trader B following an OW equilibrium  strategy
            strats = [SinesBlocks(N=int(N)), SinesBlocks(N=int(N), blocks=(0.9661, 0.9231), lambd=lambd)]
            c = CostModel(strats, rho=rho)
            print(c)

            # Initial guess for a_coeffs
            initial_cost = c.cost_trader()
            print(f"Initial strategy for trader A:")
            print(c.strats[A])
            print(f"Initial cost = {initial_cost:.4f}")

            # Minimize the cost function
            start = time.time()
            strat_a_opt, res = c.solve_min_cost()
            print(f"optimization time = {(time.time() - start):.4f}s")

            # Compute the cost with optimized coefficients
            c.strats = [strat_a_opt, c.strats[B]]
            optimized_cost = c.cost_trader()

            print("Optimized strategy for trader A:")
            print(strat_a_opt)
            print(f"Optimized cost = {optimized_cost:.4f}")

            # Plot curves
            plot_curves(c, ax=axs[i, j])

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def main():
    plot_curves_grid(rho_list=RHO_LIST, lambd=LAMBDA,
                     func=FUNC, func_params=FUNC_PARAMS, N_list=N_LIST)


if __name__ == "__main__":
    main()
