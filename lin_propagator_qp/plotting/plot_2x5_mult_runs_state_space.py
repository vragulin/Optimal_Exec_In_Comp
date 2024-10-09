""" Generate a 5x2 plot showing the trading trajectoris and full state-space path for multiple scenarios
    V. Ragulin - 10/08/2024
"""
import os
import sys
import numpy as np
import pickle
import time
from typing import List, Any, Dict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from tte_optimizer_prop_2drv_qp import State
import config_prop_qp as cfg
import plotting_utils as pu
import fourier as fr

FILE_NAMES = (
    'tte_sim_data_l20_r2_oNA_n200_g0.001_d0.0.pkl',
    'tte_sim_data_l20_r2_oNA_n200_g0.01_d0.0.pkl',
    'tte_sim_data_l20_r2_oNA_n200_g0.02_d0.0.pkl',
    'tte_sim_data_l20_r2_o0_025_n200_g0.02_d0.0007.pkl',
    'tte_sim_data_l20_r2_o0_025_n200_g0.5_d0.0007.pkl',
)
N_PLOT_POINTS = 100  # number of points for plotting

# Plot settings
NCOLS = 5
INCLUDE_SUPTITLE = False
PLOT_BOTH = False
TO_HIGHLIGHT_FREQ = 50
X_TICK_ROTATION = 0  # 45


def idx_from_iter(iter_num) -> int:
    return 2 * iter_num + 1


def plot_solution_strategies(s: State, iter_hist: List["State"], ax: Any) -> Dict[str, Any]:
    """Check the solution against theoretical values.

    :param iter_hist: list of historical iteration values
    :param ax: Matplotlib axis object for plotting
    :return: Statistics including function values and norms of differences
    """
    t_values = np.linspace(0, 1, N_PLOT_POINTS)

    a_approx = [fr.reconstruct_from_sin(t, s.a_coeff) + t for t in t_values]
    b_approx = [fr.reconstruct_from_sin(t, s.b_coeff) + t for t in t_values]
    both_approx = [(a + b * s.lambd) / (1 + s.lambd) for a, b in zip(a_approx, b_approx)]

    if ax is not None:
        ax.plot(t_values, a_approx, label=r"$a^*(t)$", color="red", linestyle="-")
        ax.plot(t_values, b_approx, label=r"$b^*_{\lambda}(t)$", color="blue", linestyle="-")
        if PLOT_BOTH:
            ax.scatter(t_values, both_approx,
                       label=r"$\frac{(a^(t) + \lambda b^*_{\lambda}(t)}{1+\lambda}$",
                       color="green", s=30, alpha=0.5)
        ax.plot(t_values, t_values, label="initial guess", color="grey", linestyle="--")
        n_iter = len(iter_hist) // 2
        title_str = ('Equilibrium trading strategies for A and B\n' +
                     r'$\rho$' + f'={s.rho}, ' + r'$\lambda$' + f'={s.lambd}, ' +
                     f'N={s.N}, ' + r'$\gamma=$' + f'{s.gamma:.3f} :: ' +
                     f'{n_iter} iterations\n')
        if 'iter_adjustment' not in s.reg_params:
            title_str += f"No regularization"
        else:
            other = s.reg_params['iter_adjustment']['max_other_cost_inc']
            title_str += f'Reg: max other increase {other:.3f}, '

            gamma_decay = s.reg_params.get('gamma_decay', 0)
            if gamma_decay > 0:
                gamma_half_life = np.log(2) / gamma_decay
                title_str += r'$\gamma_{hl}$' + f'={gamma_half_life:.0f}'

        ax.set_title(title_str, fontsize=12)
        ax.legend()
        ax.set_xlabel('t')
        ax.set_ylabel(r'$a(t), b_{\lambda}(t)$')
        ax.grid()

    return {
        "a_approx": a_approx,
        "b_approx": b_approx,
    }


def plot_state_space(s: State, iter_hist: List["State"], ax: Any, **kwargs) -> None:
    """ Plot evolution of costs for A and B over iterations
    """

    a_costs = [i.a_cost for i in iter_hist]
    b_costs = [i.b_cost for i in iter_hist]

    # Plot the points as circles
    ax.scatter(a_costs[1:-1], b_costs[1:-1], color='grey', s=20, label='(cost A, cost B)', alpha=0.3)

    # Connect the points with lines
    ax.plot(a_costs, b_costs, color='darkblue', linestyle='-', linewidth=1, alpha=0.3)

    # Highlight the starting and ending points
    ax.scatter(a_costs[0], b_costs[0], color='green', s=300, marker='<', label='init guess', facecolors='none')
    ax.scatter(a_costs[-1], b_costs[-1], color='darkblue', s=300, marker='*', label='solution', facecolors='none')

    # Highlight every HIGHLIGHT_FREQ point in orange
    to_highlight = list(range(TO_HIGHLIGHT_FREQ, len(a_costs) - 1, TO_HIGHLIGHT_FREQ))
    ax.scatter([a_costs[i] for i in to_highlight], [b_costs[i] for i in to_highlight],
               color='red', s=70, marker='^',
               label=f'every {TO_HIGHLIGHT_FREQ}th iter', alpha=0.4)

    # Rotate x-axis tick marks for this specific axis
    if 'x_tick_rotation' in kwargs:
        ticks = np.linspace(np.min(a_costs), np.max(a_costs), 10)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=kwargs['x_tick_rotation'])
        formatter = ticker.FormatStrFormatter('%.4f')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel('Trader A cost')
    ax.set_ylabel('Trader B cost')

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    else:
        ax.set_title(f'Convergence to Equilibrium - Full Path\n' +
                     f'State Space Diagram: (x,y) = (cost A, cost B)')
    ax.relim()
    ax.autoscale_view()

    ax.legend()


def plot_one_run(s: State, iter_hist: List["State"], axs: Any, idx: int) -> None:
    """ Plot run indexed by `idx` on the grid of axes `axs`"""
    _ = plot_solution_strategies(s, iter_hist, axs[0, idx])
    plot_state_space(s, iter_hist, axs[1, idx])


def plot_mult_scenarios(runs: List[dict], **kwargs) -> None:
    fig, axs = plt.subplots(2, NCOLS, figsize=(25, 10))
    if INCLUDE_SUPTITLE:
        plt.suptitle("Two-Trader Equilibrium Strategies", fontsize=16)

    for i, run in enumerate(runs):
        plot_one_run(run['state'], run['iter_hist'], axs, i)

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def main():
    runs = [pu.load_pickled_results(f) for f in FILE_NAMES]
    plot_mult_scenarios(runs)
    print('Loaded the data')


if __name__ == "__main__":
    main()
