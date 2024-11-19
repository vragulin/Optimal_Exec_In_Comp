""" Test two trader equilibrium
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
N = 100
RHO = 10
LAMBD = 10
TOL = 1e-6
G_FUNC = 'power'  # 'concave', 'exp', 'exp+perm', 'power'
RUN_TESTS = True

# Presentation parameters
N_POINTS_PLOT = 10 * N
N_COEFF_TO_SHOW = min(4,N)
ALPHA = 0.6


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


def plot_curves(t_n, a_opt, b_n) -> None:
    """Plot price impact
    :param t_n: array of trade times
    :param a_opt: array of trade sizes for the trader
    :param b_n: array of trade sizes for the adversary
    """

    t_vals = np.linspace(0, 1, N_POINTS_PLOT)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.suptitle(f'Equlirbirum in a Two-Trader Market\n'
                 fr'Resilience function: {G_FUNC}, $\lambda$={LAMBD}, N={N}')

    # Position trajector
    ax = axs[0]
    a_pos = np.array([go.position(tau, t_n, a_opt) for tau in t_vals])
    b_pos = np.array([go.position(tau, t_n, b_n) for tau in t_vals])
    init_pos = np.array([go.position(tau, t_n, 1 / N * np.ones(N)) for tau in t_vals])
    ax.plot(t_vals, a_pos, label='a(t)', color='red', marker='o', alpha=ALPHA)
    ax.plot(t_vals, b_pos, label='b(t)', color='blue', linestyle='dashed', alpha=ALPHA)
    ax.plot(t_vals, init_pos, label='risk-neutral', color='grey', linestyle='dotted', alpha=ALPHA)
    ax.set_ylabel('Position')
    ax.legend()

    # Top plot: trades as bars
    bar_width = 1 / N
    ax = axs[1]
    ax.bar(t_n, a_opt, width=bar_width, align='center', label='trader A', alpha=ALPHA, color='red')
    ax.bar(t_n, b_n, width=bar_width, align='center', label='trader B', alpha=ALPHA, color='blue')
    ax.plot(t_vals, 1 / (len(a_opt) * np.ones_like(t_vals)), label='risk-neutral', color='grey', linestyle='dotted')
    ax.set_xlabel('t')
    ax.set_ylabel('Trade size')
    ax.legend()

    # Bottom plot: price impact
    ax = axs[2]
    price_vals = np.array(
        [go.price(tau, t_n, a_opt + b_n * LAMBD, g) for tau in t_vals]
    )
    price_vals[0] = 0
    ax.plot(t_vals, price_vals, label='Price impact')
    ax.set_ylabel('Price impact')
    ax.legend()
    ax.set_title('Price impact of a trade trajectory')

    plt.tight_layout()
    plt.show()


def test_tte():
    # Define the input arrays
    t_n = np.linspace(0, 1, N)
    D = gm.decay_matrix_lt(t_n, g)
    a_opt, b_opt = gm.two_trader_equilibrium_mat(sizes=[1, LAMBD], decay_mat_lt=D, unit_strat=True)

    # Optimal Cost at Equilibrium
    cost_a_opt = gm.cost_trader_mat(a_opt, b_opt * LAMBD, D, trd_in_mkt=False)
    cost_b_opt = gm.cost_trader_mat(b_opt * LAMBD, a_opt, D, trd_in_mkt=False)
    print(f"Cost at equilibrium: c_a = {cost_a_opt:.4f}, c_b = {cost_b_opt:.4f}")

    # Plot trades:
    print(f"Trader A unit trades: {a_opt[:N_COEFF_TO_SHOW]}")
    print(f"Trader B unit trades: {b_opt[:N_COEFF_TO_SHOW]}")

    plot_curves(t_n, a_opt, b_opt)

    if RUN_TESTS:  # Confirm that each trader's cost is best reponse to the other
        for i, trd in enumerate([a_opt, b_opt]):
            adv = b_opt if i == 0 else a_opt
            size_trd, size_adv = [1, LAMBD] if i == 0 else [LAMBD, 1]

            cost = gm.cost_trader_mat(trd * size_trd, adv * size_adv, D, trd_in_mkt=False)
            trd_opt, stats = gm.best_response_mat(adv * size_adv/size_trd, D)
            cost_opt = gm.cost_trader_mat(trd_opt*size_trd, adv * size_adv, D, trd_in_mkt=False)
            print(f"Cost of trader {i}:  c_equil = {cost:.4f}, c_opt = {cost_opt:.4f}")
            print(f"Optimal trades: {trd_opt[:N_COEFF_TO_SHOW]}")
            assert np.isclose(cost, cost_opt, atol=TOL), f"Trader {i} costs do not match"
            assert np.allclose(trd, trd_opt, atol=TOL), f"Trader {i} trades do not match"
            print(f"Trader {i} trades match the best response")


if __name__ == '__main__':
    test_tte()
    print("test_tte passed")
