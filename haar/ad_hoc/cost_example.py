""" Price Displacement when trading intesity is given as a function \
    V. Ragulin, 11/09/2024
"""

import numpy as np
import warnings
from scipy.integrate import quad, IntegrationWarning
import os
import sys
import matplotlib.pyplot as plt
from typing import Callable
from codetiming import Timer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import haar_funcs as hf
from trading_intensity_funcs import linear, eager, quadratic, bucket, ow_approx, sin_approx

# Global Parameters
FUNC_TRADER = 'eager'
FUNC_OTHERS = 'constant'
LAMBDA_OTHERS = 1
RHO = 2.0
LEVEL = 5
RUN_EXACT2 = False
RUN_DBLQUAD = False


# Filter out integration warnings
warnings.filterwarnings("ignore", category=IntegrationWarning)

if __name__ == "__main__":
    func_dict = {'constant': linear,
                 'linear': linear,
                 'eager': eager,
                 'quadratic': quadratic,
                 'bucket': bucket,
                 'ow_approx': ow_approx,
                 'sin_1_term': sin_approx,
                 }

    func_args = {'constant': {'slope': 0},
                 'linear': {'slope': 1},
                 'eager': {'a': 10},
                 'quadratic': {},
                 'bucket': {'start': 0.1, 'end': 0.9, 'max_pos': 2},
                 'ow_approx': {'start': 0.05, 'end': 0.95, 'start_pos': 0.5},
                 'sin_1_term': {'amplitude': 1, 'frequency': 1},
                 }

    # Encode the trading intensity function for the trader and the others
    trader_func = func_dict[FUNC_TRADER]
    trader_args = tuple(func_args[FUNC_TRADER].values())
    trader_coeff = hf.haar_coeff(trader_func, level=LEVEL, func_args=trader_args)

    others_func = func_dict[FUNC_OTHERS]
    others_args = tuple(func_args[FUNC_OTHERS].values())
    others_coeff = hf.haar_coeff(others_func, level=LEVEL, func_args=others_args)

    # Combine trader and others coeffs to get market coeffs
    market_coeff = hf.add_haar([trader_coeff, others_coeff], [1, LAMBDA_OTHERS])

    # Calculate the cost
    points = []
    for f in [FUNC_TRADER, FUNC_OTHERS]:
        if f in {'bucket', 'ow_approx'}:
            points += [func_args[f]['start'], func_args[f]['end']]

    with Timer(text="Cost calculation time: {seconds:.2f} seconds"):
        cost = hf.cost_quad(trader_coeff, market_coeff, rho=RHO, points=points)


    # Now do the exact calculation
    def mkt_func(t: float) -> float:
        f_trader = trader_func(t, *trader_args)
        f_others = others_func(t, *others_args)
        return f_trader + f_others * LAMBDA_OTHERS


    def integrand_exact(t):
        price = quad(lambda s: np.exp(-RHO * (t - s)) * mkt_func(s), 0, t)[0]
        a_prime = trader_func(t, *trader_args)
        return a_prime * price


    with Timer(text="Exact cost calculation time: {seconds:.2f} seconds"):
        cost_exact = quad(integrand_exact, 0, 1)[0]

    # # Test agains the exact calculation in the class
    if RUN_EXACT2:
        with Timer(text="Exact2 cost calculation time: {seconds:.2f} seconds"):
            cost_exact2 = hf.cost_quad2(trader_coeff, market_coeff, rho=RHO, points=points)
        print(f"Cost: approx: {cost}, exact: {cost_exact}, exact2: {cost_exact2}")

    # Test agains the double-quad calculation in the class
    elif RUN_DBLQUAD:
        with Timer(text="Double-quad cost calculation time: {seconds:.2f} seconds"):
            cost_dblquad = hf.cost_dblquad(trader_coeff, market_coeff, rho=RHO)
        print(f"Cost: approx: {cost}, exact: {cost_exact}, dblquad: {cost_dblquad}")

    else:
        print(f"Cost: approx: {cost}, exact: {cost_exact}")