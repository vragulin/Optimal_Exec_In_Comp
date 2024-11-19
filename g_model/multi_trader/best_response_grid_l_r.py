""" Best response for a two-trader market with different lambda and rho values
    V. Ragulin - 11/19/2024
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
TOL = 1e-6
G_FUNC = 'exp'  # 'concave', 'exp', 'exp+perm', exp_mix', 'power'
B_STRAT = 'risk-neutral'  # 'risk_neutral', 'eager', 'conservative'

# Presentation parameters
N_POINTS_PLOT = 10 * N
N_COEFF_TO_SHOW = 4


# Define the resilience function g
def g(t, **kwargs):
    rho = kwargs.get('rho', 1)
    match G_FUNC:
        case 'concave':
            return 1 / (1 + (rho * t) ** 2)
        case 'exp':
            return np.exp(-rho * t)
        case 'exp+perm':
            return 0.5 * np.exp(-rho * t) + 0.5
        case 'power':
            return 1 / (1 + 10 * t) ** 0.4
        case 'exp_mix':
            return 0.5 * np.exp(-rho * t) + 0.5 * np.exp(-25 * rho * t)
        case _:
            raise ValueError(f'Unknown resilience function: {G_FUNC}')


def g_str(**kwargs):
    """String representation of the resilience function"""
    rho = kwargs.get('rho', 1)
    match G_FUNC:
        case 'concave':
            return f'1 / (1 + ({rho}*t)**2)'
        case 'exp':
            return f'exp(-{rho}*t)'
        case 'exp+perm':
            return f'0.5 * exp(-{rho}*t) + 0.5'
        case 'power':
            return '1 / (1 + 10 * t) ** 0.4'
        case 'exp_mix':
            return f'0.5 * exp(-{rho} * t) + 0.5 * exp({-25 * rho} * t)'
        case _:
            raise ValueError(f'Unknown resilience function: {G_FUNC}')


def plot_curves(ax, t_n, a_opt, b_n, lambd, rho) -> None:
    """Plot price impact
    :param ax: axis to plot on
    :param t_n: array of trade times
    :param a_opt: array of trade sizes for the trader
    :param b_n: array of trade sizes for the adversary
    :param lambd: position size of the adversary
    :param rho: g-function parameter
    """

    t_vals = np.linspace(0, 1, N_POINTS_PLOT)

    sup_s = fr'$\lambda$={lambd}, '
    sup_s += f'g(t)={g_str(rho=rho)}'
    ax.set_title(sup_s)

    # Position trajectory
    a_pos = np.array([go.position(tau, t_n, a_opt) for tau in t_vals])
    b_pos = np.array([go.position(tau, t_n, b_n) for tau in t_vals])
    init_pos = np.array([go.position(tau, t_n, 1 / N * np.ones(N)) for tau in t_vals])
    ax.plot(t_vals, a_pos, label='trader position', color='red')
    ax.plot(t_vals, b_pos, label='adversary position', color='blue', linestyle='dashed')
    ax.plot(t_vals, init_pos, label='risk-neutral', color='grey', linestyle='dotted')
    ax.set_ylabel('Position')
    ax.legend()


def main():

    def b_strat():
        match B_STRAT:
            case 'risk-neutral':
                return np.ones(N) * 1 / N
            case 'eager':
                return gu.parabolic(c=1.5, N=N)  # Eager adversary
            case 'conservative':
                return gu.parabolic(c=0, N=N)  # Eager adversary
            case _:
                raise ValueError(f'Unknown adversary strategy: {B_STRAT}')

    t_n = np.linspace(0, 1, N)
    b_n = b_strat()  # Adversary trades

    lambd_values = [1, 5, 10, 50]
    rho_values = [0.1, 1, 10, 50]
    fig, axs = plt.subplots(len(lambd_values), len(rho_values), figsize=(15, 10))
    fig.suptitle(fr'Optimal Response and Trades for Different $\lambda$ and $\rho$'
                 f'\nadversary strat = {B_STRAT}, N={N}', fontsize=16)

    for i, lambd in enumerate(lambd_values):
        for j, rho in enumerate(rho_values):

            def _g(t):
                return g(t, rho=rho)

            # Try the risk-neutral strategy
            init_guess = np.ones(N) / N
            cost_init = gm.cost_trader(t_n, init_guess, b_n * lambd, _g, trd_in_mkt=False)
            print(f"Initial guess cost: {cost_init:.4f}")

            # Check that I get the same result with the matrix version
            decay_mat_lt = gm.decay_matrix_lt(t_n, _g)
            a_opt, stats = gm.best_response_mat(b_n * lambd, decay_mat_lt)
            cost_mat = gm.cost_trader_mat(a_opt, b_n * lambd, decay_mat_lt, trd_in_mkt=False)
            print(f"Optimal cost matrix: {cost_mat:.4f}")
            print(f"Optimal trades matrix: {a_opt[:N_COEFF_TO_SHOW]}")

            # Plot the results
            plot_curves(axs[i, j], t_n, a_opt, b_n, lambd, rho)

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


if __name__ == "__main__":
    main()
