# Direct Sine Transform example, based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst
import time

# Parameters
DST_TYPE = 2  # See https://en.wikipedia.org/wiki/Discrete_sine_transform
NOISE_STD = 0.25  # Standard Dev of the noise term
N_TERMS = 3  # Number of terms to keep


# Step 1: Define the function on [0, 1]
def original_function(x):
	# return np.sin(np.pi * x) + 0.5 * np.sin(4 * np.pi * x) + NOISE_STD * np.random.normal(size=x.size)
	return x ** 4 - x


# Step 2: Perform the sine transform and approximate with the first N terms
def sine_transform_approximation(y, N):
	y_dst = dst(y, type=DST_TYPE)
	y_dst_cropped = np.copy(y_dst)
	y_dst_cropped[N:] = 0  # Zero out all but the first N terms
	y_approx = idst(y_dst, type=DST_TYPE)
	return y_approx, y_dst


# Step 3: Calculate the difference between the fitted and original function
def calculate_difference(original, fitted):
	return np.abs(original - fitted)


# Step 4: Plot both functions
def plot_functions(x, original, fitted, difference):
	plt.figure(figsize=(12, 6))
	plt.plot(x, original, label='Original Function')
	plt.plot(x, fitted, label='Fitted Function', linestyle='--')
	plt.plot(x, difference, label='Difference', linestyle=':')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Original vs Fitted Function')
	plt.legend()
	plt.show()


if __name__ == "__main__":
	# Step 5: Measure the time taken
	start_time = time.time()

	# Define the x values
	x = np.linspace(0, 1, 100)

	# Compute the original function values
	original_values = original_function(x)

	# Perform the sine transform approximation
	fitted_values, dst_coeffs = sine_transform_approximation(original_values, N_TERMS)

	# Calculate the difference
	difference_values = calculate_difference(original_values, fitted_values)

	# Plot the functions
	plot_functions(x, original_values, fitted_values, difference_values)

	end_time = time.time()
	print("DST Coeffs: ", dst_coeffs)
	plt.plot(dst_coeffs)
	plt.show()
	print(f"Time taken: {end_time - start_time:.4f} seconds")
