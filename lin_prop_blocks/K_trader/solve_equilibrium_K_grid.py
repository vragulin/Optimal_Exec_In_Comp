"""  Solve for the equilirbium within the OW-with-Blocks model
    V. Ragulin - 10/22/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Any

from cvxopt.misc import use_C

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../..', 'cost_function'),
    os.path.join(CURRENT_DIR, '../..', 'optimizer_qp')
])

from prop_blocks import SinesBlocks
from cost_model_K import CostModelK, Group
import trading_funcs as tf

# Parameters
RHO_LIST = [0.1, 0.5, 2.5]  # 5 levels of rho
# N_LIST = [10, 25, 50, 75]  # 2 levels of NNGROUPS = 3  # number of groups
N_LIST = [49, 50, 99, 100]
NGROUPS = 2  # number of groups
LAMBD = [0.25, 0.75]  # sizes
NTRADERS = [1, 1]  # number of traders in each group

# Regularization parameters
REG_PARAMS = {'wiggle': 0, 'wiggle_exp': 4}  # Regularization parameters - dict or None
STRAT_TYPE = {'blocks': False, 'sines': True}  # Types of strategies to use

# Exponent for the wiggle penalty
RUN_TESTS = False  # Check that the solution satisfies the Nash Equilibrium conditions
abs_tol = 1e-6

# Presentation parameters
N_PLOT_POINTS = 1000  # number of points to plot, works best with N//2
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


def plot_curves(c: CostModelK, ax: Any, **kwargs) -> None:
    """ Plot the trading strategies and the temporary price displacement """
    n_points = kwargs.get('n_points', 100)
    init_guess = kwargs.get('init_guess', SinesBlocks(N=c.N))

    # Top Chart plot curves
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [init_guess.calc(t) for t in t_values]
    ax.plot(t_values, init_curve, label='Initial guess ', color='grey', linestyle="dotted")
    for i, g in enumerate(c.groups):
        curve = [g.strat.calc(t) for t in t_values]
        ax.plot(t_values, curve, label=trader_func_name(i), color=trader_color(i))

    # Plot market
    mkt = [c.mkt.calc(t) for t in t_values]
    ax.plot(t_values, mkt, label='Market', color='cyan', linestyle='dashed')

    ax.set_xlabel('t')
    ax.set_ylabel(r"$a_i(t)$")
    ax.legend()
    ax.grid()


def main():
    fig, axs = plt.subplots(len(RHO_LIST), len(N_LIST), figsize=(15, 15))
    plt.suptitle(f'Nash Equilibrium in the Linear Propagator Model\n'
                 f'λ=[{lambd_list()}], ntraders=[{ntraders_list()}]',
                 fontsize=14)
    for i, rho in enumerate(RHO_LIST):
        for j, N in enumerate(N_LIST):
            # Create the model. The starting strategy is not important, just the structure
            strats = [SinesBlocks(N, lambd=LAMBD[k]) for k in range(NGROUPS)]
            groups = [Group(trader_code(k), strats[k], NTRADERS[k]) for k in range(NGROUPS)]
            c = CostModelK(groups, rho)

            # Solve for the equilibrium
            start_time = time.time()
            res = c.solve_equilibrium(strat_type=STRAT_TYPE)
            print(f"Time to solve for rho={rho}, N={N}: {time.time() - start_time:.2f} seconds")

            # Compute the cost with optimized coefficients
            opt_cost = {k: c.cost_trader(k) for k in range(NGROUPS)}

            # Print the results
            print(f"\nEquilibrium Strategies for rho={rho}, N={N}:")
            print(c)
            print("\nOptimized costs:")
            print({k: round(float(v), 4) for k, v in opt_cost.items()})
            mat_cond = np.linalg.cond(res['H'])
            print(f"Condition of FOC matrix: {mat_cond:.2f}")

            # Plot the curves
            plot_curves(c, n_points=N_PLOT_POINTS, ax=axs[i, j])
            axs[i, j].set_title(fr'$\rho$={rho}, N={N}, cond={mat_cond:.2e}', fontsize=10)

    plt.tight_layout(rect=(0, 0.01, 1, 0.97))
    plt.show()


if __name__ == "__main__":
    main()



