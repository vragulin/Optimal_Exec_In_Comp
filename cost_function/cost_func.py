import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from fourier import find_fourier_coefficients


# 1. Functions a(t) and b(t) are now user-defined
def a_function(t, kappa, lambda_, gamma=1):
	return gamma * t ** 100  # Example: a simple linear function


def b_function(t, kappa, lambda_, gamma=1):
	# return gamma * np.sinh(lambda_ * t) / np.sinh(lambda_)  # Example: a quadratic function
	return gamma * t ** 0.1


def a_dot_fun(t, kappa, lambda_, gamma=1):
	return gamma * 100 * t ** 99  # Derivative of t is 1


def b_dot_fun(t, kappa, lambda_, gamma=1):
	# return gamma * lambda_ * np.cosh(lambda_ * t) / np.sinh(lambda_)  # np.cosh(lambda_)  # Derivative of t^2 is 2t
	return gamma * 0.1 * t**(-0.9)


# 2. Function to find Fourier coefficients for user-defined functions a and b (without the linear term)
def find_fourier_coefficients_old(a_func, b_func, kappa, lambda_, N, gamma=1):
	a_coeffs = np.zeros(N)
	b_coeffs = np.zeros(N)

	# Compute Fourier coefficients for both functions
	for n in range(1, N + 1):
		# Fourier coefficient for a(t) without the linear term (subtract t)
		a_coeffs[n - 1], _ = quad(lambda t: (a_func(t, kappa, lambda_) - t * gamma) * np.sin(n * np.pi * t), 0, 1)

		# Fourier coefficient for b(t) without the quadratic term (subtract t^2)
		b_coeffs[n - 1], _ = quad(lambda t: (b_func(t, kappa, lambda_) - t * gamma) * np.sin(n * np.pi * t), 0, 1)

	return 2 * a_coeffs, 2 * b_coeffs


def reconstruct_function(t, coeffs, N, gamma=1):
	reconstruction = gamma * t  # Add the linear term t separately
	for n in range(1, N + 1):
		reconstruction += coeffs[n - 1] * np.sin(n * np.pi * t)
	return reconstruction


# 4. Exact cost computation function
def compute_exact_cost(a_func, b_func, kappa, lambda_, verbose=False):
	def integrand_temp(t):
		a_dot = a_dot_fun(t, kappa, lambda_)
		b_dot = b_dot_fun(t, kappa, lambda_)
		return (a_dot + lambda_ * b_dot) * a_dot

	def integrand_perm(t):
		a_dot = a_dot_fun(t, kappa, lambda_)
		return kappa * (a_func(t, kappa, lambda_) + lambda_ * b_func(t, kappa, lambda_)) * a_dot

	temp_cost, _ = quad(integrand_temp, 0, 1)
	perm_cost, _ = quad(integrand_perm, 0, 1)

	if verbose:
		print("Exact temp_cost: ", temp_cost)
		print("Exact perm_cost: ", perm_cost)

	return temp_cost + perm_cost


# 5. Approximate cost computation function
def compute_approximate_cost(a_coeffs, b_coeffs, kappa, lambda_, N, verbose=False):
	#  \int_0^1 \kappa(a + \lambda b)\dot a =
	#  \kappa ( \frac{1 + \lambda}{2} - \frac{2}{\pi} \sum_{k=1}^{\lfloor N/2 \rfloor} \frac{(1 + \lambda) a_{2k}}{k}
	#   + \lambda \sum_{k=1}^{\lfloor N/2 \rfloor} \frac{2 b_{2k}}{k \pi} )

	temp_cost = (1 + lambda_) + (np.pi ** 2 / 2) * sum(
		(a_n ** 2 + lambda_ * b_n * a_n) * n ** 2 for n, (a_n, b_n) in enumerate(zip(a_coeffs, b_coeffs), start=1))

	# perm_cost = kappa * ((1 + lambda_) / 2 - (2 / np.pi) * sum((1 + lambda_) * a_coeffs[2*k+1] / (2*k+2) for k in
	# range(1, N//2 - 1)) \ + lambda_ * sum(2 * b_coeffs[2*k+1] / ((2*k+2) * np.pi) for k in range(1, N//2 - 1)))

	# Correcting the definition of even_indices and other parts
	# even_indices = [i for i in range((N // 2) * 2 + 2) if i % 2 == 0]
	even_indices = [i for i in range(N) if i % 2 == 0]

	pcost_1 = (1 + lambda_) / 2

	# Fixing the sum loop for pcost_2
	pcost_2 = sum((2 * (1 + lambda_) * a_coeffs[k]) / ((k + 1) ** 2 * np.pi ** 2) for k in even_indices)

	# Fixing the sum loop for pcost_3
	pcost_3 = lambda_ * sum(2 * b_coeffs[k] / ((k + 1) * np.pi) for k in even_indices)

	perm_cost = kappa * (pcost_1 - pcost_2 + pcost_3)

	if verbose:
		print("Approx temp part", temp_cost)
		print("Approx perm part", perm_cost)
		print("pcost_1:", pcost_1)
		print("pcost_2:", pcost_2)
		print("pcost_3:", pcost_3)

	return temp_cost + perm_cost


# 6. Function to compare the exact and approximate costs
def compare_costs(a_func, b_func, kappa, lambda_, N, verbose=False):
	a_coeffs, b_coeffs = tuple(
		find_fourier_coefficients([a_func, b_func], kappa, lambda_, N)
	)

	if verbose:
		print("A coefs:", a_coeffs)
		print("B coefs:", b_coeffs, "\n")

	exact_cost = compute_exact_cost(a_func, b_func, kappa, lambda_, verbose=verbose)
	approx_cost = compute_approximate_cost(a_coeffs, b_coeffs, kappa, lambda_, N, verbose=verbose)
	difference = exact_cost - approx_cost
	return {"Exact Cost": exact_cost, "Approximate Cost": approx_cost, "Difference": difference}, a_coeffs, b_coeffs


# 7. Function to compute L2 distance and plot results
def check_fourier_approximation(a_func, b_func, a_coeffs, b_coeffs, kappa, lambda_, N):
	t_values = np.linspace(0, 1, 100)

	# Calculate original and reconstructed values
	a_original = np.array([a_func(t, kappa, lambda_) for t in t_values])
	b_original = np.array([b_func(t, kappa, lambda_) for t in t_values])

	# Reconstruct by adding back the linear term
	a_reconstructed = np.array([reconstruct_function(t, a_coeffs, N) for t in t_values])
	b_reconstructed = np.array([reconstruct_function(t, b_coeffs, N) for t in t_values])

	# Compute L2 distances
	a_l2_dist = np.sqrt(np.trapz((a_original - a_reconstructed) ** 2, t_values))
	b_l2_dist = np.sqrt(np.trapz((b_original - b_reconstructed) ** 2, t_values))

	print("\nL2 distance for a(t):", a_l2_dist)
	print("L2 distance for b(t):", b_l2_dist)

	# Plot original and reconstructed functions
	plt.figure(figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.plot(t_values, a_original, label='Original a(t)', color='blue')
	plt.plot(t_values, a_reconstructed, '--', label='Reconstructed a(t)', color='red')
	plt.title(f'Comparison of a(t), N={N}')
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(t_values, b_original, label='Original b(t)', color='blue')
	plt.plot(t_values, b_reconstructed, '--', label='Reconstructed b(t)', color='red')
	plt.title(f'Comparison of b(t), N={N}')
	plt.legend()

	plt.show()

	return a_l2_dist, b_l2_dist


# Example usage
if __name__ == "__main__":
	kappa = 1
	lambda_ = 6
	# N = [5, 10, 20, 50, 100, 200, 500]

	N = [5, 10, 20, 50, 100, 200, 500]

	# Define user-specific functions a_func and b_func
	a_func = a_function
	b_func = b_function

	approx_costs = []
	for n in N:
		# Compare the exact and approximate costs
		result, a_coeffs, b_coeffs = compare_costs(a_func, b_func, kappa, lambda_, n)
		approx_costs.append(result['Approximate Cost'])
		print(n, result)

		# Check the Fourier series approximations and plot
		check_fourier_approximation(a_func, b_func, a_coeffs, b_coeffs, kappa, lambda_, n)

		# even_indices = [i for i in range((n // 2) * 2 + 1) if i % 2 == 0]
		# print(even_indices)

	# Plot approximate cost as a function of Fourier terms
	plt.plot(N, approx_costs, label = "Approx Costs")
	plt.plot(N, [result['Exact Cost']] * len(N), label = "Exact Costs")
	plt.title("Exact vs. Approximate Costs vs. Number of Sine Terms (N)")
	plt.legend()
	plt.grid()
	plt.show()

	cost_error = [x - result['Exact Cost'] for x in approx_costs]
	plt.plot(N[3:], cost_error[3:], label="Approx Error")
	plt.title("Approximation Error vs. Number of Sine Terms (N)")
	plt.legend()
	plt.grid()
	plt.show()
