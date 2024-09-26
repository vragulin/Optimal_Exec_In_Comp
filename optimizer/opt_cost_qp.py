"""
Miminize cost function given b
A version of opt_cost.py that uses a specialized QP solver.
"""
# ToDo: This is work in progress.  The code is not complete.

import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'cost_function')))
import fourier as fr
import trading_funcs as tf
from cost_function_approx import cost_fn_a_approx
import time

# Global Parameters
N = 10  # number of Fourier terms
KAPPA = 1  # permanent impact
LAMBD = 6  # temporary impact
SIGMA = 0  # volatility of the stock -- not sure if it works with the approximate cost function
XI_A = 0  # risk aversion of a -- not sure if it works with the approximate cost function
N_PLOT_POINTS = 100  # number of points for plotting

# Global variables
b_coeffs = None  # It will be estimated when needed


def b_func(t: float, params: dict) -> float:
	""" Adversary trading schedule """
	return tf.b_func_risk_averse(t, params)


def cost_function(a_coeffs, gamma=1):
	global b_coeffs
	if b_coeffs is None:
		b_coeffs = fr.sin_coeff(lambda t: b_func(t, kappa, lambd) - t, N)

	return cost_fn_a_approx(a_coeffs, b_coeffs, kappa, lambd)


if __name__ == "__main__":
	# Initial guess for a_coeffs
	initial_guess = np.zeros(N)
	initial_cost = cost_function(initial_guess)
	print(f"Initial a_coeffs = {np.round(initial_guess, 3)}")
	print(f"Initial guess cost = {initial_cost:.4f}\n")

	# Minimize the cost function
	start = time.time()
	result = minimize(cost_function_approx, initial_guess)
	print(f"optimization time = {(time.time() - start):.4f}s")

	# Optimized coefficients
	optimized_a_coeffs = result.x

	# Compute the cost with optimized coefficients
	optimized_cost = cost_function(optimized_a_coeffs)

	print(f"Optimized a_coeffs = {np.round(optimized_a_coeffs, 3)}")
	print(f"Optimized cost = {optimized_cost:.4f}")

