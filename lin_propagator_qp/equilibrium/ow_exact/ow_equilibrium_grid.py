""" Generate a grid plot of exact OW solutions for different values of rho and lambda
    V. Ragulin - 13/08/2024
"""
import numpy as np
import matplotlib.pyplot as plt
from ow_equilibrium import OW
from typing import Any

# Parameters
RHO_VALUES = [0.1, 1, 10, 100]
LAMBD_VALUES = [0.1, 1, 20, 100]
N_ROWS = 2


def plot_curves(ow: OW, ax: Any) -> None:
    """ Plot the equilibrium trading strategies """
    eq = ow.solve_nash()
    nash = OW(rho=rho, lambd=lambd, a=eq[:2], b=eq[2:])

    t_sample = np.linspace(0, 1, ow.n_to_plot+1)
    a_curve = np.array([nash.ow_strat(t, nash.a) for t in t_sample])
    b_curve = np.array([nash.ow_strat(t, nash.b) for t in t_sample])
    ow_curve = np.array([nash.ow_paper(t) for t in t_sample])

    ax.set_title(f"rho={ow.rho}, lambd={ow.lambd}")
    ax.plot(t_sample, a_curve, label='a(t)', color='red')
    ax.plot(t_sample, b_curve, label=r'$b_\lambda(t)$', color='blue', linestyle='dashed')
    ax.plot(t_sample, t_sample, label='t', color='grey', linestyle='dotted')
    ax.plot(t_sample, ow_curve, label='OW(t)', color='green', linestyle='dotted')
    ax.set_title(f"rho={ow.rho}, lambd={ow.lambd}")
    ax.set_xlabel('t')
    ax.set_ylabel('pos over time')
    ax.legend()
    ax.grid()


if __name__ == "__main__":
    # Generate the grid

    fig, axs = plt.subplots(len(RHO_VALUES), len(LAMBD_VALUES), figsize=(12, 12))
    for i, rho in enumerate(RHO_VALUES):
        for j, lambd in enumerate(LAMBD_VALUES):
            ow = OW(rho, lambd)
            plot_curves(ow, axs[i, j])

    plt.tight_layout(rect=(0., 0.01, 1., 0.96))
    plt.suptitle("Obizhaeva-Wang Model: Equilibrium Trading Strategies", fontsize=16)
    plt.show()
