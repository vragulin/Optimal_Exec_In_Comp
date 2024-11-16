""" An example script to calculate price impact and cost of a trade trajectory
V. Ragulin - 11/15/2024
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from g_optimize import price, v_g, imp_cost

# Global parameters
N = 5
RHO = 1
N_POINTS_PLOT = 100
TOL = 1e-6
# Define the resilience function g
def g(t: float) -> float:
    return np.exp(-RHO*t)


if __name__ == "__main__":
    # Define the input arrays
    t_n = np.linspace(0, 1, N)
    x_n = np.ones(N) * 1/N

    # Plot price impact
    t_vals = np.linspace(0, 1, N_POINTS_PLOT)
    price_vals = np.array(
        [price(tau, t_n, x_n, g) for tau in t_vals]
    )
    plt.plot(t_vals, price_vals, label='Price impact')
    plt.xlabel('t')
    plt.ylabel('Price simpact')
    plt.title('Price impact of a trade trajectory')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calc cost
    cost = 0
    for i in range(N):
        cost_i = (price((t_n[i]-TOL), t_n, x_n, g) + x_n[i]/2) * x_n[i]
        cost += cost_i
        print(f'Cost of trade {i}: {cost_i}')

    cost2 = imp_cost(t_n, x_n, g)
    print(f'Total cost (loop): {cost}, cost (func): {cost2}')