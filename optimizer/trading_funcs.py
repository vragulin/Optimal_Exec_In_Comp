# Investor Trading Schedules - eventually move into a separate files
import numpy as np


# ---------------------------------------------
# Risk-neutral
# ---------------------------------------------
def b_func_risk_neutral(t: float, params: dict) -> float:
    return params['gamma'] * t


def b_dot_func_risk_neutral(t: float, params: dict) -> float:
    return params['gamma']


# ---------------------------------------------
# Risk-averse
# ---------------------------------------------
def b_func_risk_averse(t: float, params: dict) -> float:
    gamma, sig = params['gamma'], params['sigma']
    return gamma * np.sinh(sig * t) / np.sinh(sig)


def b_dot_func_risk_averse(t: float, params: dict) -> float:
    gamma, sig = params['gamma'], params['sigma']
    return gamma * sig * np.cosh(sig * t) / np.sinh(sig)


# ---------------------------------------------
# Eager
# ---------------------------------------------
def b_func_eager(t: float, params: dict) -> float:
    gamma, sig = params['gamma'], params['sigma']
    return gamma * (np.exp(-sig * t) - 1) / (np.exp(-sig) - 1)


def b_dot_func_eager(t: float, params: dict) -> float:
    gamma, sig = params['gamma'], params['sigma']
    return -gamma * sig * np.exp(-sig * t) / (np.exp(-sig) - 1)


# ---------------------------------------------
# Parabolic
# ---------------------------------------------
def b_func_parabolic(t: float, params: dict) -> float:
    c, gamma = params['c'], params['gamma']
    return gamma * t + t * (t - c) / (1 - c)


def b_dot_func_parabolic(t: float, params: dict) -> float:
    c, gamma = params['c'], params['gamma']
    return gamma - (2 * t - c) / (1 - c)


# ---------------------------------------------
# Equilibrium
# ---------------------------------------------
def equil_2trader(t: float, params: dict) -> float:
    """  2-Trader equilibrium tradign schedules for a and b
         which one is chosen is controlled by parameter
         trader_a
    """
    kappa, lambd = params['kappa'], params['lambd']
    trader_a: bool = params['trader_a']
    exp = np.exp

    term1 = (1 - exp(-kappa * t / 3))
    term2 = exp(kappa / 3) * (exp(kappa / 3) + exp(2 * kappa / 3) + 1) * (lambd + 1)
    term3 = (lambd - 1) * exp(kappa * t / 3)
    term4 = (lambd - 1) * exp(2 * kappa * t / 3)
    term5 = (lambd - 1) * exp(kappa * t)
    term6 = 2 * (exp(kappa)-1)

    if trader_a:
        return - term1 * (-term2 + term3 + term4 + term5) / term6
    else:
        return term1 * (term2 + term3 + term4 + term5) / (term6 * lambd)


def equil_2trader_dot(t, params: dict):
    kappa, lambd = params['kappa'], params['lambd']
    trader_a: bool = params['trader_a']
    exp = np.exp

    term1 = exp(-kappa * t / 3) * kappa
    term2 = 3 * exp(4 * t * kappa / 3) * (lambd-1)
    term3 = (exp(kappa / 3) + exp(2*kappa / 3) + exp(kappa)) * (1+lambd)
    term4 = 6 * (-1 + exp(kappa))

    if trader_a:
        return term1 * (-term2 + term3) / term4
    else:
        return term1 * (term2 + term3) / (term4 * lambd)




