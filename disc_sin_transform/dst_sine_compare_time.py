import numpy as np
import scipy.fftpack as sp
from scipy.integrate import quad
import time


# Define the function
def f(x):
	return x ** 2 - x


# Number of sample points
n_mesh = 50
n_coeffs = 7
n_iters = 1000

# Sample points
x = np.linspace(0, 1, n_mesh)
y = f(x)

# Measure time for DST
start_time = time.time()
for _ in range(n_iters):
	dst_coefficients = sp.dst(y, type=2)
dst_time = time.time() - start_time

# Normalize DST coefficients to match the Fourier sine series coefficients
dst_coefficients = dst_coefficients / n_mesh

# Reconstruct function values from DST coefficients
start_time = time.time()
for _ in range(n_iters):
	reconstructed_dst = sp.idst(dst_coefficients, type=2) / (2 * n_mesh)
reconstruction_dst_time = time.time() - start_time


# Measure time for Fourier sine series coefficients
def fourier_sine_coefficients(n, f, a, b):
	coefficients = []
	for k in range(1, n + 1):
		integral = (2 / (b - a)) * np.trapz(f(np.linspace(a, b, 1000)) * np.sin(k * np.pi * np.linspace(a, b, 1000)),
		                                    np.linspace(a, b, 1000))
		coefficients.append(integral)
	return np.array(coefficients)


def nc_fourier_sine_coefficients(n, f, a=0, b=1):
	a_coeffs = np.zeros(n)

	# Compute Fourier coefficients for both functions
	for n in range(1, n + 1):
		# Fourier coefficient for a(t) without the linear term (subtract t)
		a_coeffs[n - 1], _ = quad(lambda t: f(t) * np.sin(n * np.pi * t), a, b)

	return a_coeffs


start_time = time.time()
for _ in range(n_iters):
	sine_series_coefficients = nc_fourier_sine_coefficients(n_coeffs, f, 0, 1)
sine_series_time = time.time() - start_time

# Reconstruct function values from Fourier sine series coefficients
start_time = time.time()
for _ in range(n_iters):
	reconstructed_sine_series = np.zeros_like(x)
	for k in range(1, n_coeffs + 1):
		reconstructed_sine_series += sine_series_coefficients[k - 1] * np.sin(k * np.pi * x)
reconstruction_sine_series_time = time.time() - start_time

print(f"DST computation time: {dst_time:.6f} seconds")
print(f"Reconstruction from DST time: {reconstruction_dst_time:.6f} seconds")
print(f"Fourier sine series computation time: {sine_series_time:.6f} seconds")
print(f"Reconstruction from Fourier sine series time: {reconstruction_sine_series_time:.6f} seconds")
