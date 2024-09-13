import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt


# Define the function
def f(x):
	return x ** 2 - x


# Number of sample points
n_values = [10, 50, 100, 500, 1000, 10000]

# Interval [0, 1]
a, b = 0, 1


# Function to compute Fourier sine series coefficients
def fourier_sine_coefficients(n, f, a, b):
	coefficients = []
	for k in range(1, n + 1):
		integral = (2 / (b - a)) * np.trapz(f(np.linspace(a, b, 1000)) * np.sin(k * np.pi * np.linspace(a, b, 1000)),
		                                    np.linspace(a, b, 1000))
		coefficients.append(integral)
	return np.array(coefficients)


# Plotting the convergence
plt.figure(figsize=(12, 8))

for n in n_values:
	# Sample points
	x = np.linspace(a, b, n)
	y = f(x)

	# Compute DST-II coefficients
	dst_coefficients = scipy.fftpack.dst(y, type=2)

	# Compute Fourier sine series coefficients
	sine_series_coefficients = fourier_sine_coefficients(n, f, a, b)

	# Normalize DST coefficients to match the Fourier sine series coefficients
	#dst_coefficients = dst_coefficients / np.sqrt(2 * (n + 1))
	dst_coefficients = dst_coefficients / n #(n + 1)

	# Plot the coefficients
	# plt.plot(dst_coefficients, label=f'DST Coefficients (n={n})')
	# plt.plot(sine_series_coefficients, label=f'Sine Series Coefficients (n={n})', linestyle='dashed')

	print(f"\nn={n}:")
	print("sine coeffs: ", sine_series_coefficients[:5])
	print("dst coeffs: ", dst_coefficients[:5])
	print("max diff:", max(abs(sine_series_coefficients[:5] - dst_coefficients[:5])))

# plt.xlabel('Coefficient Index')
# plt.ylabel('Coefficient Value')
# plt.title('Convergence of DST and Fourier Sine Series Coefficients')
# plt.legend()
# plt.grid(True)
# plt.show()
