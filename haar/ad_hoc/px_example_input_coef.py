""" Caclulate the displacement price impact using Haar wavelets
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
import haar_funcs as hf

# Example usage
if __name__ == "__main__":

    rho = 1.0
    t = 0.5
    coeffs = [
        (1, 0, 0.5),
        (2, 1, -0.5),
        (2, 2, 0.25),
        (2, 3, -0.25)
    ]  # Example coefficients given as (level, k, value)
    c0 = 0.2  # Average of the function

    result = hf.price_haar(t, (c0, coeffs), rho)

    # Not calculate using quad
    points = np.linspace(0, 1, 2**2+1).astype(float)

    def integrand(s):
        m = hf.reconstruct_from_haar((c0, coeffs), s)
        return m * np.exp(-rho * (t - s))

    result_quad = quad(integrand, 0, t, points=points)[0]

    print(f"Integral result over [0, {t}]: haar: {result}, quad: {result_quad}")



