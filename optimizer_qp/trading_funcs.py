# Investor Trading Schedules - eventually move into a separate files
import numpy as np


# ---------------------------------------------
# Risk-neutral
# ---------------------------------------------
def risk_neutral(t: float, **kwargs) -> float:
    return t


def risk_neutral_deriv(t: float, **kwargs) -> float:
    return params['gamma']


# ---------------------------------------------
# Risk-averse
# ---------------------------------------------
def risk_averse(t: float, **kwargs) -> float:
    sig = kwargs.get("sigma", 1)
    return np.sinh(sig * t) / np.sinh(sig)


def risk_averse_deriv(t: float, **kwargs) -> float:
    sig = kwargs.get("sigma", 1)
    return sig * np.cosh(sig * t) / np.sinh(sig)


# ---------------------------------------------
# Eager
# ---------------------------------------------
def eager(t: float, **kwargs) -> float:
    sig = kwargs.get('sigma', 1)
    return (np.exp(-sig * t) - 1) / (np.exp(-sig) - 1)


def eager_deriv(t: float, params: dict) -> float:
    sig = kwargs.get('sigma', 1)
    return -sig * np.exp(-sig * t) / (np.exp(-sig) - 1)


# ---------------------------------------------
# Parabolic
# ---------------------------------------------
def parabolic(t: float, **kwargs) -> float:
    c = kwargs.get('c', 0.5)
    try:
        return t + t * (t - c) / (1 - c)
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"Trying to divide by (1-c), with c=1), {e}")


def parabolic_deriv(t: float, params: dict) -> float:
    c = kwargs.get('c', 0.5)
    try:
        return 1 + (2 * t - c) / (1 - c)
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"Trying to divide by (1-c), with c=1), {e}")


# ---------------------------------------------
# Equilibrium
# ---------------------------------------------
def equil_2trader(t: float, **kwargs) -> float:
    """  2-Trader equilibrium tradign schedules for a and b
         which one is chosen is controlled by parameter
         trader_a
    """
    kappa, lambd = kwargs.get('kappa', 1), kwargs.get('lambd', 1)
    trader_a: bool = kwargs.get("trader_a", True)
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


def equil_2trader_deriv(t, **kwargs):
    kappa, lambd = kwargs.get('kappa', 1), kwargs.get('lambd', 1)
    trader_a: bool = kwargs.get("trader_a", True)
    exp = np.exp

    term1 = exp(-kappa * t / 3) * kappa
    term2 = 3 * exp(4 * t * kappa / 3) * (lambd-1)
    term3 = (exp(kappa / 3) + exp(2*kappa / 3) + exp(kappa)) * (1+lambd)
    term4 = 6 * (-1 + exp(kappa))

    if trader_a:
        return term1 * (-term2 + term3) / term4
    else:
        return term1 * (term2 + term3) / (term4 * lambd)




