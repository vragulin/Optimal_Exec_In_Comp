"""  Solve for best response within the OW-with-Blocks model
    V. Ragulin - 10/21/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Filter out integration warnings
import warnings
from scipy.integrate import IntegrationWarning

warnings.filterwarnings("ignore", category=IntegrationWarning)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../..', 'optimizer_qp')
])

from cost_classes import CostHaarK, Group, HaarStrat, SCIPY, ANALYTIC
import trading_funcs as tf

# Parameters
# LEVEL = 7  # level of the Haar Wavelet System (i.e. N= 2**LEVEL)
RHO = 10  # propagator decay
NGROUPS = 2
# number of groups
LAMBD = [1, 2]  # sizes
NTRADERS = [1, 1]  # number of traders in each group
TRADER_TO_OPT = 0  # trader to optimize
SIGMA = 3  # risk aversion or the eagerness coefficient

FUNC = tf.risk_averse  # tf.risk_averse tf.eager tf.risk_neutral
FUNC_ARGS = {'sigma': 3}  # Expects a dictionary
OTHER_STRAT_CODE = "a Risk-Averse"  # "a Risk-Neutral" "an Eager", "a Risk-Neutral"

# Optimization parameters
abs_tol = 1e-12
SOLVER = ANALYTIC  # SCIPY or ANALYTIC

# Presentation parameters
N_PLOT_POINTS = 100  # number of points to plot, works best with N//2
N_COEFF_TO_PRINT = 4  # Number of coefficients to print
MARKERSIZE = 3  # Marker for the strategy being optimized


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


def plot_curves(axs, c: CostHaarK, var_group: int = 0, **kwargs) -> None:
    n_points = kwargs.get('n_points', 100)
    opt_cost = kwargs.get('opt_cost', None)

    # Plot initial guess and optimized functions
    ax = axs[0]
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [t for t in t_values]
    ax.plot(t_values, init_curve, label='Initial guess ', color='grey', linestyle="dotted")

    for i, g in enumerate(c.groups):
        curve = [g.strat.calc(t) for t in t_values]
        if i == var_group:
            ax.plot(t_values, curve, label=trader_func_name(i), color='red', linewidth=1,
                    marker='o', markersize=MARKERSIZE)
        else:
            ax.plot(t_values, curve, label=trader_func_name(i), color=trader_color(i))

    ax.set_title(f'Position Trajectories, Opt Cost={opt_cost:.2f}\nlevel={c.level}, N={c.N}', fontsize=11)
    ax.set_xlabel('t')
    ax.set_ylabel(r"$a_i(t)$")
    ax.legend()
    ax.grid()

    # Middle chart - trading intensity
    ax = axs[1]
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [t for t in t_values]
    ax.plot(t_values, np.ones(n_points + 1), label='Initial guess ', color='grey', linestyle="dotted")

    for i, g in enumerate(c.groups):
        curve = [g.strat.deriv(t) for t in t_values]
        if i == var_group:
            ax.plot(t_values[:-1], curve[:-1], label=trader_func_name(i), color='red', linewidth=1,
                    marker='o', markersize=MARKERSIZE)
        else:
            ax.plot(t_values[:-1], curve[:-1], label=trader_func_name(i), color=trader_color(i))

    ax.set_title(f'Trading Intensities, level={c.level}, N={c.N}', fontsize=11)
    ax.set_xlabel('t')
    ax.set_ylabel(r"$a_i'(t)$")
    ax.legend()
    ax.grid()

    # Bottom chart
    ax = axs[2]
    dp = [c.price(t) for t in t_values]
    dp[0] = 0
    dp[-1] = dp[-1]

    ax.set_title(f'Price Displacement, level={c.level}, N={c.N}', fontsize=11)
    ax.plot(t_values, dp, label="dP(0,t)", color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel("dP(0,t)")
    ax.legend()
    ax.grid()


def main():
    levels = [3, 4, 5, 6, 7]
    # levels = [3, 3, 4]
    fig, axs = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle('Best Response vs. Haar Wavelet Resolution (Level)\n' +
                 fr'$\rho$={RHO}, $\lambda$={LAMBD[1]}', fontsize=16)

    for idx, level in enumerate(levels):
        global LEVEL
        LEVEL = level

        def other_trajectory(t: float) -> float:
            return FUNC(t, **FUNC_ARGS)

        strats = [HaarStrat.from_func(func=other_trajectory, level=LEVEL, lambd=LAMBD[i])
                  for i in range(NGROUPS)]
        groups = [Group(trader_code(i), strats[i], NTRADERS[i]) for i in range(NGROUPS)]
        c = CostHaarK(groups, RHO)
        print(c)

        # Calculate the objective function for the initial strategy
        group_opt = TRADER_TO_OPT
        initial_cost = c.cost_trader(group_opt)
        print(f"Initial cost for a trader from group {group_opt} at level {level}: {initial_cost:.4f}")

        # Minimize the cost function
        start = time.time()
        strat_opt, res = c.solve_min_cost(group_opt=group_opt, abs_tol=abs_tol, solver=SOLVER)
        print(f"optimization time for level {level} = {(time.time() - start):.4f}s")

        # Compute the cost with optimized coefficients
        optimized_cost = res['fun']

        print(f"Optimized strategy for trader from group {group_opt} at level {level}:")
        print(strat_opt)
        print(f"Optimized cost at level {level} = {optimized_cost:.4f}")

        # Find the exact solution and plot curves
        row = idx // 5
        col = idx % 5
        plot_curves(axs[:, col], res['var_model'], group=res['var_trader_idx'], n_points=N_PLOT_POINTS,
                    opt_cost=optimized_cost)

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


if __name__ == "__main__":
    main()
