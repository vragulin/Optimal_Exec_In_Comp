import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def a_function(t, kappa, lambda_, gamma=1):
	return gamma * t  # Example: a simple linear function


def b_function(t, kappa, lambda_, gamma=1):
	return gamma * np.sinh(lambda_ * t) / np.sinh(lambda_)  # Example: a quadratic function


def a_dot_fun(t, kappa, lambda_, gamma=1):
	return gamma * 1  # Derivative of t is 1


def b_dot_fun(t, kappa, lambda_, gamma=1):
	return gamma * lambda_ * np.cosh(lambda_ * t) / np.cosh(lambda_)  # Derivative of t^2 is 2t


# 2. Function to find Fourier coefficients for user-defined functions a and b (without the linear term)
def find_fourier_coefficients(a_func, b_func, kappa, lambda_, N, gamma=1):
	a_coeffs = np.zeros(N)
	b_coeffs = np.zeros(N)

	# Compute Fourier coefficients for both functions
	for n in range(1, N + 1):
		# Fourier coefficient for a(t) without the linear term (subtract t)
		a_coeffs[n - 1], _ = quad(
			lambda t: (a_func(t, kappa, lambda_) - t * gamma
			           ) * np.sin(n * np.pi * t),
			0, 1)

		# Fourier coefficient for b(t) without the quadratic term (subtract t^2)
		b_coeffs[n - 1], _ = quad(
			lambda t: (b_func(t, kappa, lambda_) - t * gamma
			           ) * np.sin(n * np.pi * t), 0,
			1)

	return 2 * a_coeffs, 2 * b_coeffs


def reconstruct_function(t, coeffs, N, gamma=1):
	reconstruction = gamma * t  # Add the linear term t separately
	for n in range(1, N + 1):
		reconstruction += coeffs[n - 1] * np.sin(n * np.pi * t)
	return reconstruction

