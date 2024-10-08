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
    :param kwargs: can pass precomputed values via 'precomp' key
    :return: price impact at time t
    """

    pi, exp, sin, cos = np.pi, np.exp, np.sin, np.cos
    n = np.arange(1, len(a_n) + 1)

    if 'precomp' in kwargs:
        precomp = kwargs['precomp']
        sin_n_pi_t = precomp.get('sin_n_pi_t', sin(n * pi * t))
        cos_n_pi_t = precomp.get('cos_n_pi_t', cos(n * pi * t))
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


def cost_fn_prop_a_integral(a_n: np.ndarray, b_n: np.ndarray, lambd, rho, verbose=False):
    """ Compute the exact value of the cost function for the given Fourier coefficients
        using the exact integration of both the propagator and cost per time.
            dP[0,t] = int_0^1 dot{a(t)} P_{a,b_{lambda}}(t) dt
    Args:
        a_n (np.ndarray): Fourier coefficients for the a(t) function.
        b_n (np.ndarray): Fourier coefficients for the b(t) function.
        lambd (float): size of trader B.
        rho (float): exponential decay of the propagator
        verbose (bool, optional): If True, print the intermediate results. Defaults to False.

    Returns:
        float: The value of the cost function.
    """

    def a_dot(t):
        return fr.reconstruct_deriv_from_sin(t, a_n) + 1

    def integrand(t):
        return a_dot(t) * prop_price_impact_approx(t, a_n, b_n, lambd, rho)

    return quad(integrand, 0, 1)[0]


def cost_fn_prop_a_integ_approx(a_n: np.ndarray, b_n: np.ndarray, lambd, rho, verbose=False):
    """ Compute the exact value of the cost function for the given Fourier coefficients
        using a combination of the Fourier approximation of the propagator and
        exact integration of the cost per time.
            dP[0,t] = int_0^1 dot{a(t)} P_{a,b_{lambda}}(t) dt
    Args:
        a_n (np.ndarray): Fourier coefficients for the a(t) function.
        b_n (np.ndarray): Fourier coefficients for the b(t) function.
        lambd (float): size of trader B.
        rho (float): exponential decay of the propagator
        verbose (bool, optional): If True, print the intermediate results. Defaults to False.

    Returns:
        float: The value of the cost function.
    """

    def a_dot(t):
        return fr.reconstruct_deriv_from_sin(t, a_n) + 1

    def integrand(t):
        return a_dot(t) * prop_price_impact_integral(t, a_n, b_n, lambd, rho)

    return quad(integrand, 0, 1)[0]


def cost_fn_prop_a_approx(a_n: np.ndarray, b_n: np.ndarray, lambd, rho, verbose=False, **kwargs):
    """ Compute the exact value of the cost function of Trader A for the given Fourier coefficients
        using the analytic formula in terms of the Fourier coefficients from the Chriss(24) paper.

    :param a_n: the Fourier coefficients for the a(t) function.
    :param b_n: the Fourier coefficients for the b(t) function.
    :param lambd: Size of trader B.
    :param rho: Exponential decay of the propagator.
    :param verbose: If True, print the intermediate results. Defaults to False.
    :param kwargs: can pass precomputed values via 'precomp' key

    :return: The value of the cost function.
    """
    # ToDo - add logic to precompute the constants and pass them via kwargs

    pi, exp = np.pi, np.exp

    # Constants (later can be precomputed)
    n = np.arange(1, len(a_n) + 1)
    n_sq = n ** 2
    n_odd = n % 2
    neg_one_to_n = np.ones(n.shape) - 2 * n_odd
    m_p_n_odd = (n[:, None] + n[None, :]) % 2
    msq_nsq = n_sq[:, None] - n_sq[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(m_p_n_odd, n_sq[:, None] / msq_nsq, 0)

    # Mega-variables
    c_n = a_n + lambd * b_n
    d_n = (c_n * n * pi) / (rho ** 2 + (n * pi) ** 2)
    K = (1 + lambd) / rho + rho * np.sum(d_n)

    # Term 1
    t1 = (exp(-rho) - 1) / rho * K

    # Term 2
    t2 = 2 * np.sum(d_n * n_odd)

    # Term 3
    t3 = (1 + lambd) / rho

    # Term 4
    t4 = -K * pi * rho * np.sum((a_n * n) / (rho ** 2 + (n * pi) ** 2)
                     * (1 - neg_one_to_n * exp(-rho)))

    # Term 5
    t5_1 = 0.5 * pi * rho * np.sum(a_n * n * d_n)
    t5_2 = 2 * pi * np.sum(d_n[:, None] * (a_n * n)[None, :] * M)
    t5 = t5_1 + t5_2

    # Term 6 = 0

    if verbose:
        print(f"Term 1: {t1}")
        print(f"Term 2: {t2}")
        print(f"Term 3: {t3}")

        print(f"Term 4: {t4}")
        print(f"Term 5: {t5}, terms 5_1: {t5_1}, terms 5_2: {t5_2}")
        print(f"Sum: {t1 + t2 + t3 + t4 + t5}")
    return t1 + t2 + t3 + t4 + t5


def cost_fn_prop_b_approx(a_n: np.ndarray, b_n: np.ndarray, lambd, rho, verbose=False, **kwargs):
    """ Compute the exact value of the cost function of Trader B for the given Fourier coefficients
        using the analytic formula in terms of the Fourier coefficients from the Chriss(24) paper.

    :param a_n: the Fourier coefficients for the a(t) function.
    :param b_n: the Fourier coefficients for the b(t) function.
    :param lambd: Size of trader B.
    :param rho: Exponential decay of the propagator.
    :param verbose: If True, print the intermediate results. Defaults to False.
    :param kwargs: can pass precomputed values via 'precomp' key

    :return: The value of the cost function.
    """
    return lambd ** 2 * cost_fn_prop_a_approx(b_n, a_n, 1/lambd, rho, verbose, **kwargs)


def cost_fn_prop_a_matrix(a_n: np.ndarray, b_n: np.ndarray, lambd, rho, verbose=False, **kwargs):
    """
    The matrix version of the cost function.
    Compute the exact value of the cost function of Trader A for the given Fourier coefficients
    using the analytic formula in terms of the Fourier coefficients from the Chriss(24) paper.

    :param a_n: the Fourier coefficients for the a(t) function.
    :param b_n: the Fourier coefficients for the b(t) function.
    :param lambd: Size of trader B.
    :param rho: Exponential decay of the propagator.
    :param verbose: If True, print the intermediate results. Defaults to False.
    :param kwargs: can pass precomputed values via 'precomp' key

    :return: The value of the cost function.
    """
    pi, exp = np.pi, np.exp

    # Constants (later can be precomputed)
    n = np.arange(1, len(a_n) + 1)
    N = np.diag(n).astype(float)
    n_sq = n ** 2
    i = np.ones(n.shape)
    i_odd = n % 2
    i_mp = i - 2 * i_odd

    m_p_n_odd = (n[:, None] + n[None, :]) % 2
    msq_nsq = n_sq[:, None] - n_sq[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        M = 2 * np.where(m_p_n_odd, n_sq[:, None] / msq_nsq, 0)

    D = np.diag(pi * n / (rho ** 2 + (n * pi) ** 2))

    d = D @ (a_n + lambd * b_n)
    K = (1 + lambd) / rho + rho * np.sum(d)

    # Terms
    t1 = (exp(-rho) - 1) / rho * K
    t2 = 2 * d.T @ i_odd
    t3 = (1 + lambd) / rho
    t4 = -rho * K * a_n.T @ D @ (i - exp(-rho) * i_mp)
    t5_1 = 0.5 * pi * rho * a_n.T @ N @ d
    # t5_2 = pi * d.T @ M @ N @ a_n
    t5_2 = pi * a_n.T @ N @ M.T @ d

    t5 = t5_1 + t5_2

    if verbose:
        print(f"Term 1: {t1}")
        print(f"Term 2: {t2}")
        print(f"Term 3: {t3}")

        print(f"Term 4: {t4}")
        print(f"Term 5: {t5}, terms 5_1: {t5_1}, terms 5_2: {t5_2}")
        print(f"Sum: {t1 + t2 + t3 + t4 + t5}")
    return t1 + t2 + t3 + t4 + t5
