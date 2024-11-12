"""  Solve for best response within the OW-with-Blocks model
    Use the Quadratic Programming representation of the problem
    V. Ragulin - 10/21/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '..', 'cost_function'),
    os.path.join(CURRENT_DIR, '..', 'optimizer_qp')
])

from prop_blocks import SinesBlocks, CostModel
from cost_model_qp import CostModelQP
import trading_funcs as tf

# Parameters
N = 50  # number of Fourier terms
RHO = 1  # propagator decay
LAMBD = 2  # size of trader B
SIGMA = 3  # risk aversion or eagerness coefficient
FUNC = tf.risk_averse  # tf.risk_averse tf.eager tf.risk_neutral
FUNC_PARAMS = {"sigma": SIGMA}  # {} {"sigma": SIGMA}
OTHER_STRAT_CODE = "a Risk-Averse" # "a Risk-Neutral" "an Eager" "a Risk-Averse" "an OW-Equilibrium"

REG_PARAMS = {'wiggle': 0, 'wiggle_exp': 4}  # Regularization parameters - dict or None

# Constants
A, B = range(2)
USE_QP = True  # If True, use Quadratic Programming, else Scipy minimize


def other(trader: int) -> int:
    return 1 - trader


def trader_code(trader: int) -> str:
    return 'A' if trader == A else 'B'


def trader_color(trader: int) -> str:
    return 'red' if trader == A else 'blue'


def trader_func_name(trader: int) -> str:
    return 'a(t)' if trader == A else r'$b_\lambda(t)$'


def plot_curves(c: CostModel, **kwargs) -> None:
    n_points = kwargs.get('n_points', 100)
    init_guess = kwargs.get('init_guess', SinesBlocks(N=c.N))
    trader = kwargs.get('trader', A)

    """ Plot curves and and calc stats """
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [init_guess.calc(t) for t in t_values]
    opt_curve = [c.strats[trader].calc(t) for t in t_values]
    other_curve = [c.strats[other(trader)].calc(t) for t in t_values]

    # Plot initial guess and optimized functions
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plt.suptitle(f'Best Response to {OTHER_STRAT_CODE} Passive Adversary, Linear Propagator Model\n'
                 f'trader={trader_code(trader)}, N={c.N}, '
                 f'λ=[{c.strats[A].lambd, c.strats[B].lambd}], '
                 f'ρ={c.rho}', fontsize=14)

    # Top chart
    ax = axs[0]
    ax.plot(t_values, init_curve, label='Initial guess ' + trader_func_name(trader), color='grey', linestyle="dotted")
    ax.plot(t_values, opt_curve, label="Optimal " + trader_func_name(trader),
            color=trader_color(trader), linewidth=2)
    ax.plot(t_values, other_curve, label="Passive adversary " + trader_func_name(other(trader)),
            color=trader_color(other(trader)), linestyle="dashed")
    ax.set_title(f'Trading Schedules for A and B', fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{trader_func_name(A)}, {trader_func_name(B)}')
    ax.legend()
    ax.grid()

    # Bottom chart
    ax = axs[1]
    dp = [c.price(t) for t in t_values]
    dp[0] = 0
    dp[-1] = dp[-1] + c.mkt.blocks[-1] * c.mkt.lambd

    ax.set_title(f'Temporary Price Displacement', fontsize=11)
    ax.plot(t_values, dp, label="dP(0,t)", color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel("dP(0,t)")
    ax.legend()
    ax.grid()

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def main():
    # Set up the environment  with a function from the trading_funcs_qp module
    # or it can be any other function defined in this or another module.
    # -------------------------------------------------------------------------
    # Example 1: Trader B following a risk-neutral strategy
    # c = CostFunction(rho=RHO, lambd=LAMBD, N=N, b_func=lambda t: t)
    # -------------------------------------------------------------------------
    # Example 2: Trader B following a risk-averse strategy
    strat_b = SinesBlocks.from_func(func=FUNC, N=N, params=FUNC_PARAMS, lambd=LAMBD)
    strats = [SinesBlocks(N=strat_b.N), strat_b]
    c = CostModelQP(strats, rho=RHO)
    # -------------------------------------------------------------------------
    # Example 3: Trader B following an eager strategy
    # strat_b = SinesBlocks.from_func(func=tf.eager, N=10, params={"sigma": 3}, lambd=5.0)
    # strats = [SinesBlocks(N=strat_b.N), strat_b]
    # c = CostModel(strats, rho=0.1)
    # -------------------------------------------------------------------------
    # Example 4: Trader B following an equilibrium  strategy
    # -------------------------------------------------------------------------
    # Example 5: Trader B defined explicitly with parameters
    # -------------------------------------------------------------------------
    # strats = [SinesBlocks(N=N), SinesBlocks(N=N, blocks=(0.9661, 0.9231), lambd=LAMBD)]
    # c = CostModelQP(strats, rho=RHO, reg_params=REG_PARAMS)
    # print(c)

    # Initial guess for a_coeffs
    initial_cost = c.cost_trader()
    print(f"Initial strategy for trader A:")
    print(c.strats[A])
    print(f"Initial cost = {initial_cost:.4f}")

    # Minimize the cost function
    start = time.time()
    if USE_QP:
        strat_a_opt, res = c.solve_min_cost_qp()
    else:
        strat_a_opt, res = c.solve_min_cost()

    print(f"optimization time = {(time.time() - start):.4f}s")

    # Compute the cost with optimized coefficients
    c.strats = [strat_a_opt, c.strats[B]]
    optimized_cost = c.cost_trader()

    print("Optimized strategy for trader A:")
    print(strat_a_opt)
    print(f"Optimized cost = {optimized_cost:.4f}")

    # Find the exact solution and plot curves
    plot_curves(c)


if __name__ == "__main__":
    main()
