# Investor Trading Schedules - eventually move into a separate files
import numpy as np


# ---------------------------------------------
# Risk-neutral
# ---------------------------------------------
def b_func_risk_neutral(t: float, params: dict):
    return params['gamma'] * t


def b_dot_func_risk_neutral(t: float, params: dict):
    return params['gamma']


# ---------------------------------------------
# Risk-averse
# ---------------------------------------------
def b_func_risk_averse(t: float, params: dict):
    gamma, sig = params['gamma'], params['sigma']
    return gamma * np.sinh(sig * t) / np.sinh(sig)


def b_dot_func_risk_averse(t: float, params: dict):
    gamma, sig = params['gamma'], params['sigma']
    return gamma * sig * np.cosh(sig * t) / np.sinh(sig)


# ---------------------------------------------
# Eager
# ---------------------------------------------
def b_func_eager(t: float, params: dict):
    gamma, sig = params['gamma'], params['sigma']
    return gamma * (np.exp(-sig * t) - 1) / (np.exp(-sig) - 1)


def b_dot_func_eager(t: float, params: dict):
    gamma, sig = params['gamma'], params['sigma']
    return -gamma * sig * np.exp(-sig * t) / (np.exp(-sig) - 1)


# ---------------------------------------------
# Parabolic
# ---------------------------------------------
def b_func_parabolic(t, params):
    c, gamma = params['c'], params['gamma']
    return gamma * t + t * (t - c) / (1 - c)


def b_dot_func_parabolic(t, params: dict):
    c, gamma = params['c'], params['gamma']
    return gamma - (2 * t - c) / (1 - c)
