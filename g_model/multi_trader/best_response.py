""" Best Response to Passive Adversary, G-Model
V. Ragulin - 11/17/2024
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.abspath(os.path.join(current_dir, '..'))
])

import g_one_trader as go
import g_multi_trader as gm

# Global parameters
N = 100
RHO = 10
LAMBD = 10
TOL = 1e-6
G_FUNC = 'exp'
# 'concave', 'exp', 'power'

# Presentation parameters
N_POINTS_PLOT = 10 * N
N_COEFF_TO_SHOW = 4
PLOT_MATRIX_SOLUTION = True

# Define the resilience function g
def g(t):
    match G_FUNC:
        case 'concave':
            return g_concave(t)
        case 'exp':
            return g_exp(t)
        case 'power':
            return g_power(t)
        case _:
            raise ValueError(f'Unknown resilience function: {G_FUNC}')


def g_concave(t: float) -> float:
    # return np.exp(-RHO*t)
    return 1 / (1 + (RHO * t) ** 2)


def g_exp(t: float) -> float:
    return 0.5* np.exp(-RHO * t) + 0.5


def g_power(t: float) -> float:
    return 1 / (1 + RHO * t) ** 0.4


def plot_curves(t_n, a_opt, b_n) -> None:
    """Plot price impact
    :param t_n: array of trade times
    :param a_opt: array of trade sizes for the trader
    :param b_n: array of trade sizes for the adversary
    """

    t_vals = np.linspace(0, 1, N_POINTS_PLOT)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.suptitle(f'Optimal Response and Trades in a Two-Trader Market\n'
                 fr'Resilience function: {G_FUNC}, $\lambda$={LAMBD}, N={N}')

    # Position trajector
    ax = axs[0]
    a_pos = np.array([go.position(tau, t_n, a_opt) for tau in t_vals])
    b_pos = np.array([go.position(tau, t_n, b_n) for tau in t_vals])
    init_pos = np.array([go.position(tau, t_n, 1 / N * np.ones(N)) for tau in t_vals])
    ax.plot(t_vals, a_pos, label='Trader position', color='red')
    ax.plot(t_vals, b_pos, label='Adversary position', color='blue', linestyle='dashed')
    ax.plot(t_vals, init_pos, label='Initial position', color='grey', linestyle='dotted')
    ax.set_ylabel('Position')
    ax.legend()

    # Top plot: trades as bars
    bar_width = 1 / N
    ax = axs[1]
    ax.bar(t_n, a_opt, width=bar_width, align='center', label='trader', alpha=0.3, color='red')
    ax.bar(t_n, b_n, width=bar_width, align='center', label='adversary', alpha=0.3, color='blue')
    ax.plot(t_vals, 1 / (len(a_opt) * np.ones_like(t_vals)), label='init_guess', color='grey', linestyle='dotted')
    ax.set_xlabel('t')
    ax.set_ylabel('Trade size')
    ax.legend()

    # Bottom plot: price impact
    ax = axs[2]
    price_vals = np.array(
        [go.price(tau, t_n, a_opt + b_n * LAMBD, g) for tau in t_vals]
    )
    ax.plot(t_vals, price_vals, label='Price impact')
    ax.set_ylabel('Price impact')
    ax.legend()
    ax.set_title('Price impact of a trade trajectory')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up the model
    t_n = np.linspace(0, 1, N)
    b_n = np.ones(N) * 1 / N  # Adversary trades

    # Try the risk-netral strategy
    init_guess = np.ones(N) / N
    cost_init = gm.cost_trader(t_n, init_guess, b_n * LAMBD, g, trd_in_mkt=False)
    print(f"Initial guess cost: {cost_init:.4f}")

    a_opt, stats = gm.best_response(t_n, b_n * LAMBD, g, tol=1e-10)
    print(f"Optimal cost: {stats.fun:.4f}")
    print(f"Optimal trades: {a_opt[:N_COEFF_TO_SHOW]}")

    #  Check that I get the same result with the matrix version
    decay_mat_lt = gm.decay_matrix_lt(t_n, g)
    a_opt_mat, stats_mat = gm.best_response_mat(b_n * LAMBD, decay_mat_lt)
    cost_mat = gm.cost_trader_mat(t_n, a_opt_mat, b_n * LAMBD, decay_mat_lt, trd_in_mkt=False)
    print(f"Optimal cost matrix: {cost_mat:.4f}")
    print(f"Optimal trades matrix: {a_opt_mat[:N_COEFF_TO_SHOW]}")

    # Plot the results
    if PLOT_MATRIX_SOLUTION:
        plot_curves(t_n, a_opt_mat, b_n)
    else:
        plot_curves(t_n, a_opt, b_n)
