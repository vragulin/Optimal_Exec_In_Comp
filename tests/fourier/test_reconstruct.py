import pytest as pt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../../cost_function"))
import fourier as fr


@pt.mark.parametrize("n, expected", [
	(1, 0),
	(2, -0.25),
	(3, -0.375),
	(4, -0.4375)
])
def test_reconstruct_power(n, expected):
	def func(t):
		return t ** n - t

	# Define the coefficients for the polynomial x^n - x
	coeffs = fr.sin_coeff(func, 10)  # Example coefficients for x^2 - x
	t = 0.5  # Example point to evaluate the function

	# Call the function
	result = fr.reconstruct_from_sin(t, coeffs)

	# Assert the result is close to the expected value
	assert np.isclose(result, expected, rtol=1e-3), f"Expected {expected}, but got {result}"


@pt.mark.parametrize("n, expected", [
	(1, 0),
	(2, -0.25),
	(3, -0.375),
	(4, -0.4375)
])
def test_reconstruct_power_small_n(n, expected):
	def func(t):
		return t ** n - t

	# Define the coefficients for the polynomial x^n - x
	coeffs = fr.sin_coeff(func, 15)  # Example coefficients for x^2 - x
	t = 0.5  # Example point to evaluate the function

	# Call the function
	result = fr.reconstruct_from_sin(t, coeffs, 10)

	# Assert the result is close to the expected value
	assert np.isclose(result, expected, rtol=1e-3), f"Expected {expected}, but got {result}"


@pt.mark.parametrize("x_exp, deriv_order, expected", [
	(1, 1, 0),
	(1, 2, 0),
	(2, 1, 0),
	(3, 1, -0.25),
	(4, 1, -0.5),
	(2, 2, 2),
	(3, 2, 3)
])
def test_reconstruct_deriv_power(x_exp, deriv_order, expected):
	def func(t):
		return t ** x_exp - t

	# Define the coefficients for the polynomial x^n - x
	coeffs = fr.sin_coeff(func, 35)  # Example coefficients for x^2 - x
	t = 0.5  # Example point to evaluate the function

	# Call the function
	result = fr.reconstruct_deriv_from_sin(t, coeffs, 35, deriv_order)

	# Assert the result is close to the expected value
	rtol = 5e-2 if deriv_order == 2 else 5e-3
	assert np.isclose(result, expected, rtol=rtol), f"Expected {expected}, but got {result}"


@pt.mark.parametrize("x_exp, deriv_order, expected", [
	(1, 1, 0),
	(1, 2, 0),
	(2, 1, 0),
	(3, 1, -0.25),
	(4, 1, -0.5),
	(2, 2, 2),
	(3, 2, 3)

])
def test_reconstruct_deriv_power_small_n(x_exp, deriv_order, expected):
	def func(t):
		return t ** x_exp - t

	# Define the coefficients for the polynomial x^n - x
	coeffs = fr.sin_coeff(func, 50)  # Example coefficients for x^2 - x
	t = 0.5  # Example point to evaluate the function

	# Call the function
	result = fr.reconstruct_deriv_from_sin(t, coeffs, 35, deriv_order)

	# Assert the result is close to the expected value
	rtol = 5e-2 if deriv_order == 2 else 5e-3
	assert np.isclose(result, expected, rtol=rtol), f"Expected {expected}, but got {result}"


