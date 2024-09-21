import pytest as pt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../../cost_function"))
import fourier as fr


def test_sin_coeff_zero():
	def func(t):
		return 0

	coeffs = fr.sin_coeff(func, 10)
	exp_coeffs = np.zeros(10)
	np.testing.assert_allclose(coeffs, exp_coeffs, atol=1e-10)


def test_sin_coeff_x_squared():
	def func(t):
		return t * t - t

	coeffs = fr.sin_coeff(func, 10)
	exp_coeffs = np.array([-0.258012, 0., -0.00955601, 0., -0.0020641,
	                       0., -0.000752222, 0., -0.000353926, 0.])
	np.testing.assert_allclose(coeffs, exp_coeffs, atol=1e-6)


def test_sin_coeff_x_sqrt():
	def func(t):
		return t ** 0.5 - t

	coeffs = fr.sin_coeff(func, 10)
	exp_coeffs = np.array([0.238085, 0.0777079, 0.0438919, 0.0278884, 0.0202588,
	                       0.0152408, 0.0121998, 0.00991587, 0.00835828, 0.00710156])
	np.testing.assert_allclose(coeffs, exp_coeffs, atol=1e-6)


def test_find_fourier_coeffs_x_xsquared():
	def a_func(t: float, kappa: float, lambda_: float) -> float:
		return t

	def b_func(t: float, kappa: float, lambda_: float) -> float:
		return t * t

	functions = [a_func, b_func]
	kappa: float = 1.0
	lambda_: float = 1.0
	N: int = 10

	coeffs_list = fr.find_fourier_coefficients(functions, kappa, lambda_, N)
	coeffs = np.vstack(coeffs_list)
	exp_coeffs = np.vstack(
		[np.zeros(10),
		 np.array([-0.258012, 0., -0.00955601, 0., -0.0020641,
		           0., -0.000752222, 0., -0.000353926, 0.])]
	)
	np.testing.assert_allclose(coeffs, exp_coeffs, atol=1e-6)
