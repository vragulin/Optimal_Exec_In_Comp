""" Price Displacement when trading intesity is given as a function \
    V. Ragulin, 11/09/2024
"""

import numpy as np
from scipy.integrate import quad
import os
import sys
import matplotlib.pyplot as plt
from typing import Callable
from codetiming import Timer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import haar_funcs as hf
from trading_intensity_funcs import linear, eager, quadratic, bucket, ow_approx, sin_approx

# Global Parameters
FUNC = 'sin_1_term'
RHO = 2.0
LEVEL = 6

# Example usage
if __name__ == "__main__":
    func_dict = {'constant': linear,
                 'linear': linear,
                 'eager': eager,
                 'quadratic': quadratic,
                 'bucket': bucket,
                 'ow_approx': ow_approx,
                 'sin_1_term': sin_approx,
                 }

    func_args = {'constant': {'slope': 0},
                 'linear': {'slope': 1},
                 'eager': {'a': 10},
                 'quadratic': {},
                 'bucket': {'start': 0.1, 'end': 0.9, 'max_pos': 2},
                 'ow_approx': {'start': 0.05, 'end': 0.95, 'start_pos': 0.5},
                 'sin_1_term': {'amplitude': 1, 'frequency': 1},
                 }

    # Define the trading intensity function
    func_kwargs = func_args[FUNC]


    def func(t: float) -> float:
        return func_dict[FUNC](t, **func_kwargs)


    haar_coeff = hf.haar_coeff(func, level=5)

    # Plots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    # axs = np.atleast_2d(axs)
    fig.suptitle(f"Trading Intensity and Price Displacement\n"
                 f"strategy = {FUNC}, level = {LEVEL}, "
                 rf"$\rho$={RHO}", fontsize=14)

    t_values = np.linspace(0, 1, 101)

    # Plot the trading intensity function
    ax = axs[0]
    m_prime_approx = [hf.reconstruct_from_haar(haar_coeff, t) for t in t_values]
    m_prime = [func(t) for t in t_values]
    ax.plot(t_values[:-1], m_prime[:-1], label="m'(t), exact", color='blue')
    ax.plot(t_values[:-1], m_prime_approx[:-1], label="m'(t), approx", color='red', marker='o')
    ax.set_title("Market"
                 " Trading Intensity Function")
    ax.set_xlabel("t")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.set_ylabel("m'(t)")
    ax.legend()

    # Plot position trajectory
    ax = axs[1]
    with Timer(text="Position Trajectory Calculation (Haar): {:.3f} seconds"):
        pos_approx = [hf.integrate_haar(haar_coeff, 0, t) for t in t_values]
    if FUNC in {'ow_approx', 'bucket'}:
        points = [func_kwargs['start'], func_kwargs['end']]
    else:
        points = None
    with Timer(text="Position Trajectory Calculation (Exact): {:.3f} seconds"):
        pos = [quad(lambda s: func(s), 0, t, points=points
                    )[0] for t in t_values]
    ax.plot(t_values, t_values, label="t", linestyle='--', color='gray')
    ax.plot(t_values, pos, label="m(t), exact", color='blue')
    ax.plot(t_values, pos_approx, label="m(t), approx", color='red', marker='o')
    ax.set_title("Position Trajectory")
    ax.set_xlabel("t")
    ax.set_ylabel("m(t)")
    ax.legend()

    # Plot the price displacement
    ax = axs[2]
    with Timer(text="Price Displacement Calculation (Haar): {:.3f} seconds"):
        p_approx = [hf.price_haar(t, haar_coeff, RHO) for t in t_values]
    with Timer(text="Price Displacement Calculation (Exact): {:.3f} seconds"):
        p_exact = [quad(lambda s: func(s) * np.exp(-RHO * (t - s)), 0, t)[0] for t in t_values]
    ax.plot(t_values, p_exact, label="P(t), exact", color='blue')
    ax.plot(t_values, p_approx, label="P(t), approx", color='red', marker='o')
    ax.set_title("Price Displacement")
    ax.set_xlabel("t")
    ax.set_ylabel("P(t)")
    ax.legend()

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()
