"""  Solve for best response within the OW-with-Blocks model
    V. Ragulin - 10/21/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../..', 'cost_function'),
    os.path.join(CURRENT_DIR, '../..', 'optimizer_qp')
])

from prop_blocks import SinesBlocks
from cost_model_K import CostModelK, Group
import trading_funcs as tf

# Parameters
N = 2  # number of Fourier terms
RHO = 0.1  # propagator decay
NGROUPS = 2  # number of groups
LAMBD = [1, 2]  # sizes
NTRADERS = [1, 1]  # number of traders in each group
TRADER_TO_OPT = 0  # trader to optimize
SIGMA = 3  # risk aversion or eagerness coefficient

FUNC = tf.risk_averse  # tf.risk_averse tf.eager tf.risk_neutral
FUNC_PARAMS = {"sigma": SIGMA}  # {"sigma": SIGMA} {}
OTHER_STRAT_CODE = "a Risk-Averse"  # "a Risk-Neutral" "an Eager", "a Risk-Neutral"

# Regularization parameters
REG_PARAMS = {'wiggle': 0, 'wiggle_exp': 4}  # Regularization parameters - dict or None
STRAT_TYPE = {'blocks': True, 'sines': True}  # Types of strategies to use
abs_tol = 1e-6

# Presentation parameters
N_PLOT_POINTS = 100  # number of points to plot, works best with N//2
N_COEFF_TO_PRINT = 4  # Number of coefficients to print


# Constants
def trader_code(trader: int) -> str:
    return f"{trader}"


def trader_color(trader: int) -> str:
    colors = {0: 'red',
              1: 'blue',
              2: 'green',
              3: 'purple',
              4: 'orange'}
    return colors[trader % len(colors)]


def trader_func_name(trader: int) -> str:
    return fr'$a_{{{trader}}}(t)$'


def lambd_list():
    return ', '.join([str(l) for l in LAMBD])


def ntraders_list():
    return ', '.join([str(n) for n in NTRADERS])


def plot_curves(c: CostModelK, var_group: int = 0, **kwargs) -> None:
    n_points = kwargs.get('n_points', 100)
    init_guess = kwargs.get('init_guess', SinesBlocks(N=c.N))

    # Plot initial guess and optimized functions
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plt.suptitle(f'Best Response by a Trader in Group {var_group} in the Linear Propagator Model\n'
                 f'N={c.N}, λ=[{lambd_list()}], ntraders=[{ntraders_list()}], ρ={c.rho}',
                 fontsize=14)

    # Top chart
    ax = axs[0]
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [init_guess.calc(t) for t in t_values]
    ax.plot(t_values, init_curve, label='Initial guess ', color='grey', linestyle="dotted")

    for i, g in enumerate(c.groups):
        curve = [g.strat.calc(t) for t in t_values]
        if i == var_group:
            ax.plot(t_values, curve, label=trader_func_name(i), color='red', linewidth=2, marker='o')
        else:
            ax.plot(t_values, curve, label=trader_func_name(i), color=trader_color(i))

    ax.set_title(f'Trading Strategies for Trader Groups', fontsize=11)
    ax.set_xlabel('t')
    ax.set_ylabel(r"$a_i(t)$")
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
    # Create the model.  The starting strategy are not important, just the structure
    strats = [SinesBlocks.from_func(func=FUNC, N=N, params=FUNC_PARAMS, lambd=LAMBD[i])
              for i in range(NGROUPS)]
    # strats = [SinesBlocks(N, lambd=LAMBD[i]) for i in range(NGROUPS)]
    groups = [Group(trader_code(i), strats[i], NTRADERS[i]) for i in range(NGROUPS)]
    c = CostModelK(groups, RHO)
    print(c)

    # Calculate the objective function for the initial strategy
    group_opt = TRADER_TO_OPT
    initial_cost = c.cost_trader(group_opt)
    print(f"Initial cost for a trader from group {group_opt}: {initial_cost:.4f}")

    # Minimize the cost function
    start = time.time()
    strat_opt, res = c.solve_min_cost(group_opt=group_opt, strat_type=STRAT_TYPE,
                                      abs_tol=abs_tol)
    print(f"optimization time = {(time.time() - start):.4f}s")

    # Compute the cost with optimized coefficients
    optimized_cost = res['scipy_output'].fun

    print(f"Optimized strategy for trader from group {group_opt}:")
    print(strat_opt)
    print(f"Optimized cost = {optimized_cost:.4f}")

    # Find the exact solution and plot curves
    plot_curves(res['var_model'], group=res['var_trader_idx'], n_points=N_PLOT_POINTS)


if __name__ == "__main__":
    main()