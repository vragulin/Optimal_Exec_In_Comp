""" Calculate the equilibrium fair share (Equivalent of Fig. 5 from paper #3).
    V. Ragulin 11/05/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../..', 'cost_function'),
    os.path.join(CURRENT_DIR, '../..', 'optimizer_qp')
])

from prop_blocks import SinesBlocks
from cost_model_K import CostModelK, Group
import trading_funcs as tf

# Parameters
N = 5  # number of Fourier terms
RHO_LIST = [0.1, 1, 5]  # propagator decay
NGROUPS = 2  # number of groups
TOTAL_TRADERS_LIST = [2, 20]  # total number of trader
MKT_SHARE_LIST = [0.01, 0.12, 0.25, 0.50, 0.75, 0.88, 0.99]  # Market share of the second group

# Regularization parameters
REG_PARAMS = {'wiggle': 0, 'wiggle_exp': 4}  # Regularization parameters - dict or None
STRAT_TYPE = {'blocks': True, 'sines': True}  # Types of strategies to use

# Exponent for the wiggle penalty
abs_tol = 1e-6

# Presentation parameters
N_PLOT_POINTS = 100  # number of points to plot, works best with N//2
N_COEFF_TO_PRINT = 4  # Number of coefficients to print
VERBOSE = True


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


def fair_share_deviation(n: int, rho: float, mkt_share: float) -> tuple[float, dict]:
    """ Calculate fair share deviation for one set of parameters """

    # Create the model.  The starting strategy are not important, just the structure
    lambd_arr = [mkt_share, (1 - mkt_share) / (n - 1)]
    ntraders_arr = [1, n - 1]
    strats = [SinesBlocks(N, lambd=lambd_arr[i]) for i in range(NGROUPS)]
    groups = [Group(trader_code(i), strats[i], ntraders_arr[i]) for i in range(NGROUPS)]
    c = CostModelK(groups, rho)

    # Solve for the equilibrium
    start_time = time.time()
    print(f"\nSolving for n={n}, rho={rho}, mkt_share={mkt_share}")
    _ = c.solve_equilibrium(strat_type=STRAT_TYPE)
    print(f"Done.  Time to solve: {time.time() - start_time:.2f} seconds")

    # Compute the cost with optimized coefficients
    opt_cost = {k: c.cost_trader(k) for k in range(NGROUPS)}

    # Plot the curves
    total_cost = opt_cost[0] + opt_cost[1] * (n - 1)
    fair_share_dev = float(opt_cost[0] - total_cost * mkt_share)

    res_dict = {'opt_cost': opt_cost, 'model': c, 'n': n, 'rho': rho,
                'mkt_share': mkt_share, 'fair_share_dev': fair_share_dev}

    # Print the results
    if VERBOSE:
        # print("\nEquilibrium Strategies:")
        # print(c)
        print("\nOptimized costs:")
        print({k: round(float(v), 4) for k, v in opt_cost.items()})
        print(f"Fair share deviation: {fair_share_dev:.4f}")
    return fair_share_dev, res_dict


def plot_grids(data: dict):
    """ Plot the results of the fair share deviation calculation """
    fig, axs = plt.subplots(len(TOTAL_TRADERS_LIST), len(RHO_LIST), figsize=(12, 6))
    axs = np.atleast_2d(axs)
    used_palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)

    for i, n in enumerate(TOTAL_TRADERS_LIST):
        for j, rho in enumerate(RHO_LIST):
            ax = axs[i, j]
            ax.set_title(fr"n={n}, $\rho$={rho}")
            ax.set_xlabel('Market share')
            ax.set_ylabel('Deviation (%)')
            d_fair_share = data[(n, rho)]
            # ax.bar(MKT_SHARE_LIST, d_fair_share, color=['red' if d > 0 else 'blue' for d in d_fair_share])
            _ = sns.barplot(x=MKT_SHARE_LIST, y=d_fair_share, hue=d_fair_share,
                            palette=used_palette, ax=ax, legend=False)
    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def main():
    data = {}
    for n in TOTAL_TRADERS_LIST:
        for rho in RHO_LIST:
            data[(n, rho)] = []
            for mkt_share in MKT_SHARE_LIST:
                d_fair_share, _ = fair_share_deviation(n, rho, mkt_share)
                data[(n, rho)].append(d_fair_share)
    print(data)
    plot_grids(data)


if __name__ == "__main__":
    main()
