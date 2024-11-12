""" Caclulate the displacement price impact using Haar wavelets - Fast Version
V. Ragulin, 11/09/2024
"""

import numpy as np
from scipy.integrate import quad
import os
import sys
import matplotlib.pyplot as plt
from typing import Callable

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import haar_funcs_fast as hf

# Example usage
if __name__ == "__main__":

    rho = 1.0
    t = 0.5
    haar = np.array([0.2, 0, 0.5, 0, 0, -0.5, 0.25, -0.25], dtype=float)
    level = len(haar).bit_length() - 1
    lmum = hf.calc_lmum(level)
    result = hf.price_haar(t, haar, rho, lmum=lmum)

    # Not calculate using quad
    points = np.linspace(0, 1, 2**2+1).astype(float)

    def integrand(s):
        m = hf.reconstruct_from_haar(s, haar, lmum=lmum)
        return m * np.exp(-rho * (t - s))

    result_quad = quad(integrand, 0, t, points=points)[0]

    print(f"Integral result over [0, {t}]: haar: {result}, quad: {result_quad}")



