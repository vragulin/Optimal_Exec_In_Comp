"""
Miminize cost function given b
Subject to a contraint L(t) <= a(t) <= U(t)
"""
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import fourier as fr
import time

# Global Parameters
N = 20  # number of Fourier terms
kappa = 1  # permanent impact
lambda_ = 6  # temporary impact
xi_a = 0  # risk aversion of a
sigma = 0  # volatility of the stock

N_PLOT_POINTS = 100  # number of points for plotting
N_SAMPLE_POINTS = 50  # number of points to sample constraints

# Parameters for the b(t) function
B_CONSERVATIVE = False
B_EAGER_CONST = 1.2

if B_CONSERVATIVE:
	def b_func(t, kappa, lambda_, gamma=1):
		return gamma * np.sinh(lambda_ * t) / np.sinh(lambda_)


	def b_dot_func(t, kappa, lambda_, gamma=1):
		return gamma * lambda_ * np.cosh(lambda_ * t) / np.sinh(lambda_)


	def b_dbl_dot_func(t, kappa, lambda_, gamma=1):
		return gamma * lambda_ * lambda_ * np.sinh(lambda_ * t) / np.sinh(lambda_)
else:
	def b_func(t, kappa, lambda_, gamma=1):
		return gamma * t + B_EAGER_CONST * (0.25 - (t - 0.5) ** 2)


	def b_dot_func(t, kappa, lambda_, gamma=1):
		return gamma - 2 * B_EAGER_CONST * (t - 0.5)


	def b_dbl_dot_func(t, kappa, lambda_, gamma=1):
		return -2 * B_EAGER_CONST


def b_vector(t):
	return (
		b_func(t, kappa, lambda_),
		b_dot_func(t, kappa, lambda_),
		b_dbl_dot_func(t, kappa, lambda_)
	)


def compute_exact_cost(a_func, a_dot_func, kappa, lambda_, verbose=False):
	def integrand_temp(t):
		_a_dot = a_dot_func(t, kappa, lambda_)
		_b_dot = b_dot_func(t, kappa, lambda_)
		return (_a_dot + lambda_ * _b_dot) * _a_dot

	def integrand_perm(t):
		_a = a_func(t, kappa, lambda_)
		_b = b_func(t, kappa, lambda_)
		_a_dot = a_dot_func(t, kappa, lambda_)
		return kappa * (_a + lambda_ * _b) * _a_dot

	temp_cost = quad(integrand_temp, 0, 1)[0]
	perm_cost = quad(integrand_perm, 0, 1)[0]

	if verbose:
		print("Exact temp_cost: ", temp_cost)
		print("Exact perm_cost: ", perm_cost)

	return temp_cost + perm_cost


def plot_curves(init_guess, opt_coeffs, exact_solution, gamma=1) -> dict:
	""" Plot curves and and calc stats """
	t_values = np.linspace(0, 1, N_PLOT_POINTS)

	init_curve = [fr.reconstruct_from_sin(t, init_guess) + gamma * t for t in t_values]
	opt_curve = [fr.reconstruct_from_sin(t, opt_coeffs) + gamma * t for t in t_values]
	b_curve = [b_func(t, kappa, lambda_) for t in t_values]
	l_curve = [L_func(t) for t in t_values]
	u_curve = [U_func(t) for t in t_values]
	# Plot initial guess and optimized functions
	plt.figure(figsize=(10, 5))

	# plt.plot(t_values, init_curve, label='Initial guess', color='blue')
	plt.plot(t_values, opt_curve, label='Constrained optimal a(t)', color='red', linewidth=2)
	plt.plot(t_values, exact_solution, label="Unconstrained optimal a(t)", color="green")
	plt.plot(t_values, b_curve, label="Passive adversary b(t)", color="blue", linestyle="dashed")
	plt.plot(t_values, l_curve, label="Lower bound", color="grey", linestyle="dotted")
	plt.plot(t_values, u_curve, label="Upper bound", color="grey", linestyle="dotted")

	# Shade the area between l_curve and u_curve
	plt.fill_between(t_values, l_curve, u_curve, color='grey', alpha=0.3)

	plt.suptitle(f'Best Response to a Passive Adversary')
	plt.title(f'Adversary trading λ={lambda_} units, Permanent Impact κ={kappa}', fontsize=11)
	plt.legend()
	plt.xlabel('t')
	plt.ylabel('a(t), b(t)')
	plt.grid()
	plt.tight_layout()
	plt.show()

	# Calculate stats
	diff_approx = opt_curve - exact_solution
	max_diff = np.max(np.abs(diff_approx))
	l2_diff = norm(diff_approx)

	return {'max': max_diff, 'L2': l2_diff}


# Define the system of differential equations with exogenous b(t)
def equations(t, y, xi_a):
	a, a_prime = y
	b, b_prime, b_dbl_prime = b_vector(t)
	a_double_prime = -(lambda_ / 2) * (b_dbl_prime + kappa * b_prime) + xi_a * sigma ** 2 * a
	return np.vstack((a_prime, a_double_prime))


# Boundary conditionss
def boundary_conditions(ya, yb):
	return np.array([ya[0], yb[0] - 1])


if __name__ == "__main__":
	# Minimize the cost function

	def cost_function(a_coeffs):
		def a_func(t, kappa, lambda_, gamma=1):
			return fr.reconstruct_from_sin(t, a_coeffs) + gamma * t

		def a_dot_func(t, kappa, lambda_, gamma=1):
			return fr.reconstruct_deriv_from_sin(t, a_coeffs) + gamma

		return compute_exact_cost(a_func, a_dot_func, kappa, lambda_)


	def L_func(t):
		return 0.5 * t  # Example lower bound function


	def U_func(t):
		return min(1 + t, 1.7)  # Example upper bound function


	def constraint_function(a_coeffs):
		# Sample points
		t_points = np.linspace(0, 1, N_SAMPLE_POINTS)

		def a_func(t, kappa, lambda_, gamma=1):
			return fr.reconstruct_from_sin(t, a_coeffs) + gamma * t

		a_values = np.array([a_func(t, kappa, lambda_) for t in t_points])
		L_values = np.array([L_func(t) for t in t_points])
		U_values = np.array([U_func(t) for t in t_points])

		return np.concatenate([U_values - a_values, a_values - L_values])


	# Define the constraint
	constraints = ({
		'type': 'ineq',
		'fun': lambda a_coeffs: constraint_function(a_coeffs)
	})

	# Initial guess for a_coeffs
	initial_guess = np.zeros(N)
	initial_cost = cost_function(initial_guess)
	print(f"Initial a_coeffs = {np.round(initial_guess, 3)}")
	print(f"Initial guess cost = {initial_cost:.4f}\n")

	start = time.time()
	result = minimize(cost_function, initial_guess, method='SLSQP', constraints=constraints)
	print(f"optimization time = {(time.time() - start):.4f}s")

	# Optimized coefficients
	optimized_a_coeffs = result.x

	# Compute the cost with optimized coefficients
	optimized_cost = cost_function(optimized_a_coeffs)

	print(f"Optimized a_coeffs = {np.round(optimized_a_coeffs, 3)}")
	print(f"Optimized cost = {optimized_cost:.4f}")

	# Find the exact solution
	t = np.linspace(0, 1, N_PLOT_POINTS)
	y_init = np.zeros((2, t.size))
	sol = solve_bvp(lambda _t, y: equations(_t, y, xi_a), boundary_conditions, t, y_init)

	# Plot curves
	stats = plot_curves(initial_guess, optimized_a_coeffs, sol.y[0])
	print(f"Approx - Exact Distance: L2 = {stats['L2']:.4f}, max = {stats['max']:.4f}")
