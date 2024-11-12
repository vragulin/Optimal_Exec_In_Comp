""" Trading intensity functions examples
    V. Ragulin, 11/09/2024
"""

import numpy as np


# Trading intensity functions examples, integrate to 1 over [0,1]
def linear(t: float, slope: float) -> float:
    return 1 + slope * (t - 0.5)


def eager(t: float, a: float) -> float:
    k = a / (1 - np.exp(-a))
    return k * np.exp(-a * t)


def quadratic(t: float) -> float:
    return 3 * t ** 2


def bucket(t: float, start: float, end: float, max_pos: float) -> float:
    if t < start:
        return max_pos / start
    elif t >= end:
        return (1 - max_pos) / (1 - end)
    else:
        return 0


def ow_approx(t: float, start: float, end: float, start_pos: float = 0.5) -> float:
    return bucket(t, start, end, max_pos=start_pos)


def sin_approx(t: float, amplitude: float, frequency: float) -> float:
    # This is the deriovative of t+a*sin(freq*pi*t)
    return 1 + amplitude * np.pi * frequency * np.cos(np.pi * frequency * t)