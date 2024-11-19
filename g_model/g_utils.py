""" Helper functions for the G-model
    V. Ragulin - 11/19/2024
"""
import numpy as np


def parabolic(c: float, N: int) -> np.ndarray:
    """ Generate trades for a parabolic adversary
    :param c: parameter of the parabola
    :param N: number of trades
    :return: array of N trades
    """
    assert c != 1, "Error: parabolic strategy is not defined for c=1"
    t = np.linspace(0, 1, N)
    t_unscaled = (2 * t - c)
    t_scaled = t_unscaled / np.sum(t_unscaled)
    return t_scaled
