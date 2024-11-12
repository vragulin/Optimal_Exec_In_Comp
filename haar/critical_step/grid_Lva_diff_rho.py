""" Calculate the critival value of time step when arbitrage becomes possible
V. Ragulin 11/12/2024
"""

import numpy as np
from scipy.integrate import quad
import os
import sys
import matplotlib.pyplot as plt

from price_impact import P, cost_a

if __name__ == "__main__":

    rho_list = [0.1, 1, 1.5, 2]
    step_list = [0.1, 0.25, 0.5, 1]

    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    plt.suptitle("Trader A Cost vs. Trading Intensity\nTrader B Intensity = 1")

    for i, rho in enumerate(rho_list):
        for j, step in enumerate(step_list):
            b = 1
            a_values = np.linspace(0, 2/(rho * step), 20)
            cost_values = [cost_a(a, b, rho, step) for a in a_values]

            min_cost_idx = np.argmin(cost_values)
            min_cost_a = a_values[min_cost_idx]
            min_cost_value = cost_values[min_cost_idx]

            ax = axs[i, j]
            ax.plot(a_values, cost_values, label="cost_a")
            ax.plot(min_cost_a, min_cost_value, 'ro', label=rf"min cost={min_cost_a:.2f}")
            ax.set_title(fr"$\rho$={rho}, step={step}")
            ax.set_xlabel("a'(t)")
            ax.set_ylabel("cost_a")
            ax.legend()
            ax.grid()

    plt.tight_layout()
    plt.show()


