""" Script to demonstrate the critical step in the Haar wavelet transform.
    V. Ragulin 11/12/2024
"""

""" Calculate the critival value of time step when arbitrage becomes possible
V. Ragulin 11/12/2024
"""

import numpy as np
from scipy.integrate import quad
import os
import sys
import matplotlib.pyplot as plt

from price_impact import P, cost_a, critical_step

if __name__ == "__main__":

    b_values = [0.5, 1, 2, 10]
    rho_list = [0.1, 0.2, 0.5, 1, 2]

    for b in b_values:
        crit_step_list = [critical_step(1, b, rho)/2 for rho in rho_list]
        plt.plot(rho_list, crit_step_list, label=f"b={b}")

    plt.title("Critical Wavelet Length vs. Propagator Decay Speed\n"
              "(arbitrage is possible if step < critical)")
    plt.xlabel(r"$\rho$")
    plt.ylabel("Critical Wavelet Length")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

