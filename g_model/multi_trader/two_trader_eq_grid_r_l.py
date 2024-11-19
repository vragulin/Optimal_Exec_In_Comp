""" Test two trader equilibrium grid
    V. Ragulin - 11/15/2024
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
N = 50
RHO = 1
LAMBD = 5
TOL = 1e-6
G_FUNC = 'exp_mix'  # 'concave', 'exp', 'exp+perm', exp_mix', 'power'
RUN_TESTS = True

# Presentation parameters
N_POINTS_PLOT = 10 * N
N_COEFF_TO_SHOW = min(4, N)
ALPHA = 0.6


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
            return 1 / (1 + rho * t) ** 0.4
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
            return f'1 / (1 + {rho} * t) ** 0.4'
        case 'exp_mix':
            return f'rho={rho}'
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
    sup_s += f'{g_str(rho=rho)}'
    ax.set_title(sup_s)

    # Position trajectory
    a_pos = np.array([go.position(tau, t_n, a_opt) for tau in t_vals])
    b_pos = np.array([go.position(tau, t_n, b_n) for tau in t_vals])
    init_pos = np.array([go.position(tau, t_n, 1 / N * np.ones(N)) for tau in t_vals])
    ax.plot(t_vals, a_pos, label='a(t)', color='red', marker='o', alpha=ALPHA)
    ax.plot(t_vals, b_pos, label='b(t)', color='blue', linestyle='dashed', alpha=ALPHA)
    ax.plot(t_vals, init_pos, label='risk-neutral', color='grey', linestyle='dotted', alpha=ALPHA)
    ax.set_ylabel('Position')
    ax.legend()


def main():
    # Define the input arrays
    t_n = np.linspace(0, 1, N)
    lambd_values = [1, 2, 5, 25]
    rho_values = [1, 10, 50]
    fig, axs = plt.subplots(len(lambd_values), len(rho_values), figsize=(15, 10))
    fig.suptitle(r'Two-Trader Competitive Execution Equilibrium for Different $\lambda$ and $\rho$'
                 f'\nN={N}, g(t)={G_FUNC}', fontsize=16)

    for i, lambd in enumerate(lambd_values):
        for j, rho in enumerate(rho_values):
            def _g(t):
                return g(t, rho=rho)

            D = gm.decay_matrix_lt(t_n, _g)
            a_opt, b_opt = gm.two_trader_equilibrium_mat(sizes=[1, lambd], decay_mat_lt=D, unit_strat=True)

            # Solve for the equilibrium
            cost_a_opt = gm.cost_trader_mat(a_opt, b_opt * lambd, D, trd_in_mkt=False)
            cost_b_opt = gm.cost_trader_mat(b_opt * lambd, a_opt, D, trd_in_mkt=False)
            print(f"Cost at equilibrium for λ={lambd}, ρ={rho}: c_a = {cost_a_opt:.4f}, c_b = {cost_b_opt:.4f}")

            # Plot trades:
            print(f"Trader A unit trades for λ={lambd}, ρ={rho}: {a_opt[:N_COEFF_TO_SHOW]}")
            print(f"Trader B unit trades for λ={lambd}, ρ={rho}: {b_opt[:N_COEFF_TO_SHOW]}")

            plot_curves(axs[i, j], t_n, a_opt, b_opt, lambd, rho)

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


if __name__ == '__main__':
    main()
