""" Example with optimal trades for a given resiliance function
V. Ragulin - 11/15/2024
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

import g_optimize as go

# Global parameters
N = 100
RHO = 10
N_POINTS_PLOT = 10*N
TOL = 1e-6
# Define the resilience function g
def g(t: float) -> float:
    # return np.exp(-RHO*t)
    return 1/(1+(RHO*t)**2)

def plot_curves(t_n, x_n) -> None:
    # Plot price impact
    t_vals = np.linspace(0, 1, N_POINTS_PLOT)
    price_vals = np.array(
        [go.price(tau, t_n, x_n, g) for tau in t_vals]
    )

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.suptitle(f'Price impact and trades, N={N}')
    # Top plot: price impact
    axs[0].plot(t_vals, price_vals, label='Price impact')
    axs[0].set_ylabel('Price impact')
    axs[0].legend()
    axs[0].set_title('Price impact of a trade trajectory')

    # Bottom plot: trades as bars
    bar_width = 1 / N
    axs[1].bar(t_n, x_n, width=bar_width, align='center', label='Trades')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Trade size')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the input arrays
    t_n = np.linspace(0, 1, N)
    x_n = np.ones(N) * 1/N

    # plot_curves(t_n, x_n)

    # Calc cost of the initial guess
    cost_twap = go.imp_cost(t_n, x_n, g)
    print(f'TWAP Cost: {cost_twap}')

    # Optimize the trades
    def is_symmetric(m: np.ndarray) -> bool:
        max_diff = np.max(np.abs(m - m.T))
        return max_diff < 1e-10

    # This code works fine, but for g(t) = 1/(1+(RHO*t)**2) the condition number of the decay matrix is huge
    # so the matrix inversion is not stable
    g_mat = go.decay_matrix(t_n, g).astype(np.float64)
    assert is_symmetric(g_mat), 'Decay matrix is not symmetric'
    print(f"Condition number of the decay matrix: {np.linalg.cond(g_mat)}")
    x_n_opt = go.opt_trades_matrix(t_n, g_mat)
    cost_opt = go.imp_cost_matrix(x_n_opt, g_mat)

    print(f'Opt Cost: {cost_opt}')
    plot_curves(t_n, x_n_opt)