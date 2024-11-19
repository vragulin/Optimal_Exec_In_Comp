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
import g_utils as gu

# Global parameters
N = 100
RHO = 1
LAMBD = 1
TOL = 1e-6
G_FUNC = 'exp'  # 'concave', 'exp', 'exp+perm', exp_mix', 'power'
B_STRAT = 'eager'  # 'risk_neutral', 'eager', 'conservative'

# Presentation parameters
N_POINTS_PLOT = 10 * N
N_COEFF_TO_SHOW = 6
RUN_SCIPY = False
PLOT_MATRIX_SOLUTION = True


# Define the resilience function g
def g(t):
    match G_FUNC:
        case 'concave':
            return 1 / (1 + (RHO * t) ** 2)
        case 'exp':
            return np.exp(-RHO * t)
        case 'exp+perm':
            return 0.5 * np.exp(-RHO * t) + 0.5
        case 'power':
            return 1 / (1 + 10 * t) ** 0.4
        case 'exp_mix':
            return 0.5 * np.exp(-RHO * t) + 0.5 * np.exp(-25 * RHO * t)
        case _:
            raise ValueError(f'Unknown resilience function: {G_FUNC}')


def g_str():
    """String representation of the resilience function"""
    match G_FUNC:
        case 'concave':
            return f'1 / (1 + ({RHO}*t)**2)'
        case 'exp':
            return f'exp(-{RHO}*t)'
        case 'exp+perm':
            return f'0.5 * exp(-{RHO}*t) + 0.5'
        case 'power':
            return '1 / (1 + 10 * t) ** 0.4'
        case 'exp_mix':
            return f'0.5 * exp(-{RHO} * t) + 0.5 * exp({-25 * RHO} * t)'
        case _:
            raise ValueError(f'Unknown resilience function: {G_FUNC}')


def plot_curves(t_n, a_opt, b_n) -> None:
    """Plot price impact
    :param t_n: array of trade times
    :param a_opt: array of trade sizes for the trader
    :param b_n: array of trade sizes for the adversary
    """

    t_vals = np.linspace(0, 1, N_POINTS_PLOT)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    sup_s = f'Optimal Response and Trades in a Two-Trader Market\n'
    sup_s += fr'$\lambda$={LAMBD}, N={N}, '
    sup_s += f'g(t)={g_str()}'
    plt.suptitle(sup_s)

    # Position trajector
    ax = axs[0]
    a_pos = np.array([go.position(tau, t_n, a_opt) for tau in t_vals])
    b_pos = np.array([go.position(tau, t_n, b_n) for tau in t_vals])
    init_pos = np.array([go.position(tau, t_n, 1 / N * np.ones(N)) for tau in t_vals])
    ax.plot(t_vals, a_pos, label='trader position', color='red')
    ax.plot(t_vals, b_pos, label='adversary position', color='blue', linestyle='dashed')
    ax.plot(t_vals, init_pos, label='risk-neutral', color='grey', linestyle='dotted')
    ax.set_ylabel('Position')
    ax.legend()

    # Top plot: trades as bars
    bar_width = 1 / N
    ax = axs[1]
    ax.bar(t_n, a_opt, width=bar_width, align='center', label='trader trades', alpha=0.3, color='red')
    ax.bar(t_n, b_n, width=bar_width, align='center', label='adversary trades', alpha=0.3, color='blue')
    ax.plot(t_vals, 1 / (len(a_opt) * np.ones_like(t_vals)), label='risk-neutral', color='grey', linestyle='dotted')
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


    def b_strat():
        match B_STRAT:
            case 'risk_neutral':
                return np.ones_like(t) * 1 / N
            case 'eager':
                return gu.parabolic(c=1.5, N=N)  # Eager adversary
            case 'conservative':
                return gu.parabolic(c=0, N=N)  # Eager adversary
            case _:
                raise ValueError(f'Unknown adversary strategy: {B_STRAT}')


    b_n = b_strat()  # Adversary trades

    # Try the risk-netral strategy
    init_guess = np.ones(N) / N
    cost_init = gm.cost_trader(t_n, init_guess, b_n * LAMBD, g, trd_in_mkt=False)
    print(f"Initial guess cost: {cost_init:.4f}")

    if RUN_SCIPY:  # Run scipy as a check to the matrix calculation
        a_opt, stats = gm.best_response(t_n, b_n * LAMBD, g, tol=1e-10)
        print(f"Optimal cost: {stats.fun:.4f}")
        print(f"Optimal trades: {[f'{x:.3e}' for x in a_opt[:N_COEFF_TO_SHOW]]}")

    #  Check that I get the same result with the matrix version
    decay_mat_lt = gm.decay_matrix_lt(t_n, g)
    a_opt_mat, stats_mat = gm.best_response_mat(b_n * LAMBD, decay_mat_lt)
    cost_mat = gm.cost_trader_mat(a_opt_mat, b_n * LAMBD, decay_mat_lt, trd_in_mkt=False)
    print(f"Optimal cost matrix: {cost_mat:.4f}")
    print(f"Optimal trades matrix: {[f'{x:.3e}' for x in a_opt_mat[:N_COEFF_TO_SHOW]]}")

    # Plot the results
    if PLOT_MATRIX_SOLUTION or not RUN_SCIPY:
        plot_curves(t_n, a_opt_mat, b_n)
    else:
        plot_curves(t_n, a_opt, b_n)
