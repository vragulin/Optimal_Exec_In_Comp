""" Calculate the critival value of time step when arbitrage becomes possible
V. Ragulin 11/12/2024
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import os
import sys
import matplotlib.pyplot as plt


def P(t: float, a: float, b: float, rho: float, step: float) -> float:
    """ Price impact, A is trading a wavelet from [0,2t], B is trading uniform
    :param t:  time
    :param a:  intensity of A, i.e. +a for i in [0,step), and -a for [step, 2*step).
    :param b:  intensity of B, trading uniform on [0, 2*step)
    :param rho: propagator decay speed
    :param step: half of the wavelet
    :return:  price at t
     """

    exp = np.exp

    def p_uniform(t: float, mkt: float, rho: float) -> float:
        """ Impact of a uniform trading strategy from 0 to t """
        return mkt / rho * (1 - exp(-rho * t))

    if t <= step:
        return p_uniform(t, a + b, rho)
    else:
        tau = t - step
        p_step = p_uniform(step, a + b, rho)
        return p_step * exp(-rho * tau) + p_uniform(tau, -a + b, rho)


def cost_a(a: float, b: float, rho: float, step: float) -> float:
    """ Cost of A from the strategy """

    def integrand(t: float) -> float:
        a_deriv = a if t < step else -a
        return a_deriv * P(t, a, b, rho, step)

    cost, _ = quad(integrand, 0, 2 * step, points=[step])

    return cost


def critical_step(a: float, b: float, rho: float) -> float:
    """ Solve for the value of step when cost_a = 0
    :param a:  intensity of A
    :param b:  intensity of B
    :param rho: propagator decay speed
    :return:  critical step value
    """

    return brentq(lambda step: cost_a(a, b, rho, step), 1e-5, 1e10)


# Example use
if __name__ == "__main__":
    step = 0.5
    rho = 1
    a = 1
    b = 1
    n_steps = 20

    t_values = np.linspace(0, 2 * step, 2 * n_steps + 1)
    p_values = [P(t, a, b, rho, step) for t in t_values]
    plt.plot(t_values, p_values, label="price_impact")
    plt.title("Price Impact")
    plt.legend()
    plt.show()

    L_a = cost_a(a, b, rho, step)
    print(f"L_a: {L_a}")
