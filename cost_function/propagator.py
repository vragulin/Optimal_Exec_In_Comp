""" Implement the linear propagator for the cost function.
    In line with the Chriss(2024) paper.
    V. Ragulin, 09/30/2024
"""
import numpy as np
import fourier as fr
from scipy.integrate import quad


# from functools import reduce
# from itertools import product


def prop_price_impact_integral(t: float, a_n: np.ndarray, b_n: np.ndarray, lambd: float,
                               rho: float) -> float:
    """ Compute using exact integration the price impact of the propagator at time t.
        dP[0,t] = int_0^t exp(-rho(t-s)) dot{a(s)} ds
    :param t: time since the start of trading
    :param a_n:  Fourier coefficients of the trading rate a(t)
    :param b_n: Fourier coefficients of the trading rate b(t)
    :param lambd: size of trader B
    :param rho:  decay of the propagator
    :return: price impact at time t
    """

    def integrand(s):
        joint_coeff = a_n + b_n * lambd
        return np.exp(-rho * (t - s)) * (fr.reconstruct_deriv_from_sin(s, joint_coeff) + (1 + lambd))

    return quad(integrand, 0, t)[0]


def prop_price_impact_approx(t: float, a_n: np.ndarray, b_n: np.ndarray, lambd: float,
                             rho: float, **kwargs) -> float:
    """ Compute using the Fourier approximation the price impact of the propagator at time t.
        dP[0,t] = int_0^t exp(-rho(t-s)) dot{a(s)} ds
    :param t: time since the start of trading
    :param a_n:  Fourier coefficients of the trading rate a(t)
    :param b_n: Fourier coefficients of the trading rate b(t)
    :param lambd: size of trader B
    :param rho:  decay of the propagator
    :return: price impact at time t
    """

    pi, exp, sin, cos = np.pi, np.exp, np.sin, np.cos
    n = np.arange(1, len(a_n) + 1)

    if 'precomp' in kwargs:
        precomp = kwargs['precomp']
        sin_n_pi_t = precomp.get(['sin_n_pi_t'], sin(n * pi * t))
        cos_n_pi_t = precomp.get(['cos_n_pi_t'], cos(n * pi * t))
    else:
        sin_n_pi_t = sin(n * pi * t)
        cos_n_pi_t = cos(n * pi * t)

    # First term
    t1 = (1 + lambd) * (1 - exp(-rho * t)) / rho

    # Second term
    d_n = (a_n + lambd * b_n) / (rho ** 2 + (n * pi) ** 2)
    trig_term = rho * (cos_n_pi_t - exp(-rho * t)) + n * pi * sin_n_pi_t

    t2 = d_n @ (pi * n * trig_term)

    return t1 + t2


def cost_fn_prop_a_exact(a_n: np.ndarray, b_n: np.ndarray, lambd, rho, verbose=False):
    """ Compute the exact value of the cost function for the given Fourier coefficients.
            dP[0,t] = int_0^t exp(-rho(t-s)) \dot{a(s)} ds
    Args:
        a_n (np.ndarray): Fourier coefficients for the a(t) function.
        b_n (np.ndarray): Fourier coefficients for the b(t) function.
        lambd (float): size of trader B.
        rho (float): exponential decay of the propagator
        verbose (bool, optional): If True, print the intermediate results. Defaults to False.

    Returns:
        float: The value of the cost function.
    """
    ...
