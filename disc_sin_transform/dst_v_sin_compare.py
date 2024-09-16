import numpy as np
import scipy.fftpack
from scipy.integrate import quad
import matplotlib.pyplot as plt


# Define the function
def f(x):
	return x ** 0.05 - x


# Number of sample points
N_COEFFS = [10]
N_TIMES = [5, 10, 50, 100, 500, 1000]
DECIMALS = 5
# Interval [0, 1]
a, b = 0, 1


def fourier_sine_coefficients(n, f, a=0, b=1):
	a_coeffs = np.zeros(n)

	# Compute Fourier coefficients for both functions
	for n in range(1, n + 1):
		a_coeffs[n - 1], _ = quad(f, a, b, weight='sin', wvar=(n * np.pi), )

	return 2 * a_coeffs


if __name__ == "__main__":
	for n in N_TIMES:
		# Sample points
		x = np.linspace(a, b, n)
		y = f(x)

		# Compute DST-II coefficients
		dst_coefficients = scipy.fftpack.dst(y, type=2)
		# Normalize DST coefficients to match the Fourier sine series coefficients
		dst_coefficients = dst_coefficients / n  # (n + 1)

		# Compute Fourier sine series coefficients
		sine_series_coefficients = fourier_sine_coefficients(n, f, a, b)

		print(f"\nn_times={n}:")
		print("sin coeffs:\t", np.round(sine_series_coefficients[:5], DECIMALS))
		print("dst coeffs:\t", np.round(dst_coefficients[:5], DECIMALS))
		max_diff = max(abs(sine_series_coefficients - dst_coefficients))
		max_diff_rel = max(
			(abs(sine_series_coefficients - dst_coefficients
			     ) / (abs(sine_series_coefficients) + abs(dst_coefficients)) * 2
			 )[:5]
		)
		print("max diff:\t", np.round(max_diff, DECIMALS), ",\trelative:", np.round(max_diff_rel * 100, DECIMALS - 2),
		      "%")
