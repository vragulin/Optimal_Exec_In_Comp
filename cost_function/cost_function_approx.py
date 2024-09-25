"""
Approximation of the cost function without integrals by using Fourier approximations for a(t), b(t)
"""
import numpy as np
import fourier as fr
from scipy.integrate import quad
from functools import reduce
from itertools import product


# integral of cos(n pi t) sin(n pi t) in closed form
def int_cos_sin_old(n, m):
	pi = np.pi
	if n != m:
		int_chk = (m - m * np.cos(m * pi) * np.cos(n * pi) - m * np.sin(m * pi) * np.sin(n * pi)) / (
				pi * (m ** 2 - n ** 2))
	else:
		int_chk = 0

	return int_chk


def int_cos_sin(n, m):
	pi = np.pi
	if n == m:
		ans = 0
	else:
		ans = (m - m * (-1) ** (m + n)) / (pi * (m ** 2 - n ** 2))

	return ans


def cost_fn_a_approx_old(a_n: np.ndarray, b_n: np.ndarray,
                     kappa: float, lambd: float, verbose=False):
	"""
	This computes the cost function of trader A from the formula in the
	real-world constraints paper.
	This approach avoids computing integrals.

	:param a_n: Coefficients a_n for n from 1 to N
	:param b_n: Coefficients b_n for n from 1 to N
	:param kappa: Constant κ (kappa) - permanent market impact
	:param lambd: Constant λ (lambda) - temporary market impact

	:return: The computed value of the expression I
	"""
	# Ensure input sequences are numpy arrays
	a_n = np.array(a_n, dtype=np.float64)
	b_n = np.array(b_n, dtype=np.float64)
	N = len(a_n)

	if len(b_n) != N:
		raise ValueError("Sequences a_n and b_n must be of the same length.")

	n = np.arange(1, N + 1)  # n from 1 to N

	# Compute the third term: kappa * [ (1 + lambd) / 2 + sum_{n odd} (2 * (b_n - a_n)) / (n * π) ]
	odd_indices = np.where(n % 2 == 1)[0]  # Indices where n is odd
	n_odd = n[odd_indices]
	a_n_odd = a_n[odd_indices]
	b_n_odd = b_n[odd_indices]
	pi = np.pi

	# I: \int_0^1 \dot{a} \dot{a} dt
	int_I = 1 + 0.5 * np.sum(n ** 2 * pi ** 2 * a_n ** 2)

	# II: \int_0^1 lambda \dot{b} \dot{a} dt
	int_II = lambd + (lambd / 2) * np.sum(n ** 2 * pi ** 2 * a_n * b_n)

	# III: \int_0^1 kappa \dot{a} a dt
	int_III = kappa / 2

	# Replaced the double loop with a faster version below
	# ans = 0
	# for n, a_co in enumerate(a_n, start=1):
	# 	for m, b_co in enumerate(b_n, start=1):
	# 		ans += a_co * b_co * n * pi * int_cos_sin(n, m)

	ans = reduce(
		lambda acc, x: acc + x[0] * x[1] * x[2] * pi * int_cos_sin(x[2], x[3]),
		((a_co, b_co, n, m) for n, a_co in enumerate(a_n, start=1) for m, b_co in enumerate(b_n, start=1)),
		0
	)

	iv4 = kappa * lambd * ans

	int_IV_1 = kappa * lambd * 0.5
	int_IV_2 = ((2 * lambd * kappa) / pi) * np.sum(b_n_odd / n_odd)
	int_IV_3 = -((2 * lambd * kappa) / pi) * np.sum(a_n_odd / n_odd)
	int_IV_4 = iv4

	int_IV = int_IV_1 + int_IV_2 + int_IV_3 + int_IV_4
	integral = int_I + int_II + int_III + int_IV

	if verbose:
		print("APPROX TOTAL COST FUNCTION FROM APPROX FORMULA:")
		print("int_I: ", int_I)
		print("int_II: ", int_II)
		print("int_III: ", int_III)
		print("int_IV: ", int_IV)
		print("\tint_0^1 lambda kappa t dt: ", int_IV_1)
		print("\tint_0^1 lambda kappa (b(t)-t) dt: ", int_IV_2)
		print("\tint_0^1 lambda kappa t * (a'(t)-1) dt: ", int_IV_3)
		print("\tint_0^1 lambda kappa (b(t) - 1)*(a'(t) - 1) dt: ", int_IV_4)
		print("\tTotal of components: ", int_IV_1 + int_IV_2 + int_IV_3 + int_IV_4)

		print("Loss function approximation formula: ", integral)

	return int_I + int_II + int_III + int_IV


def cost_fn_a_approx(a_n, b_n, kappa, lambd, verbose=False):
	"""
		This computes the cost function of trader A from the formula in the
		real-world constraints paper.
		This approach avoids computing integrals.

		:param a_n: Coefficients a_n for n from 1 to N
		:param b_n: Coefficients b_n for n from 1 to N
		:param kappa: Constant κ (kappa) - permanent market impact
		:param lambd: Constant λ (lambda) - temporary market impact

		:return: The computed value of the expression I
		"""
	# Ensure input sequences are numpy arrays
	a_n = np.array(a_n, dtype=np.float64)
	b_n = np.array(b_n, dtype=np.float64)
	n_coeffs = len(a_n)

	if len(b_n) != n_coeffs:
		raise ValueError("Sequences a_n and b_n must be of the same length.")

	pi = np.pi
	n = np.arange(1, n_coeffs + 1)

	# Calculate individual terms
	t1 = (2 + kappa) * (1 + lambd) / 2

	t2 = pi ** 2 / 2 * sum(i ** 2 * (a_n[i - 1] ** 2 + lambd * a_n[i - 1] * b_n[i - 1]) for i in n)

	t3 = 2 * kappa * lambd / pi * sum((b_n[i - 1] - a_n[i - 1]) / i for i in n if i % 2 == 1)

	t4 = 2 * kappa * sum((a_n[i - 1] + lambd * b_n[i - 1]) * a_n[j - 1] * i * j / (i * i - j * j)
	                     for i in n for j in n if (i + j) % 2 == 1)

	total_loss  = t1 + t2 + t3 + t4
	if verbose:
		print("APPROX TOTAL COST FUNCTION FROM APPROX FORMULA:")
		print("int_I: ", t1)
		print("int_II: ", t2)
		print("int_III: ", t3)
		print("int_IV: ", t4)

		print("Loss function approximation formula: ", total_loss)

	return total_loss


def cost_fn_a_approx_simplified(a_n, b_n, kappa, lambd, verbose=False):
	"""
		This computes the cost function of trader A from the formula in the
		real-world constraints paper.
		This approach avoids computing integrals.

		:param a_n: Coefficients a_n for n from 1 to N
		:param b_n: Coefficients b_n for n from 1 to N
		:param kappa: Constant κ (kappa) - permanent market impact
		:param lambd: Constant λ (lambda) - temporary market impact

		:return: The computed value of the expression I
		"""
	# Ensure input sequences are numpy arrays
	a_n = np.array(a_n, dtype=np.float64)
	b_n = np.array(b_n, dtype=np.float64)
	n_coeffs = len(a_n)

	if len(b_n) != n_coeffs:
		raise ValueError("Sequences a_n and b_n must be of the same length.")

	pi = np.pi
	n = np.arange(1, n_coeffs + 1)

	# Calculate individual terms
	t1 = (2 + kappa) * (1 + lambd) / 2

	t2 = pi ** 2 / 2 * sum(i ** 2 * (a_n[i - 1] ** 2 + lambd * a_n[i - 1] * b_n[i - 1]) for i in n)

	t3 = 2 * kappa * lambd / pi * sum((b_n[i - 1] - a_n[i - 1]) / i for i in n if i % 2 == 1)

	t4 = 2 * kappa * sum((lambd * b_n[i - 1]) * a_n[j - 1] * i * j / (i * i - j * j)
	                     for i in n for j in n if (i + j) % 2 == 1)

	total_loss = t1 + t2 + t3 + t4
	if verbose:
		print("APPROX TOTAL COST FUNCTION FROM APPROX FORMULA:")
		print("int_I: ", t1)
		print("int_II: ", t2)
		print("int_III: ", t3)
		print("int_IV: ", t4)

		print("Loss function approximation formula: ", total_loss)

	return total_loss


def cost_fn_b_approx(a_n, b_n, kappa, lambd, verbose=False):
	"""
		This computes the cost function of trader B from the formula in the
		real-world constraints paper.
		This approach avoids computing integrals.

		:param a_n: Coefficients a_n for n from 1 to N
		:param b_n: Coefficients b_n for n from 1 to N
		:param kappa: Constant κ (kappa) - permanent market impact
		:param lambd: Constant λ (lambda) - temporary market impact

		:return: The computed value of the expression I
		"""
	# Ensure input sequences are numpy arrays
	a_n = np.array(a_n, dtype=np.float64)
	b_n = np.array(b_n, dtype=np.float64)
	n_coeffs = len(a_n)

	if len(b_n) != n_coeffs:
		raise ValueError("Sequences a_n and b_n must be of the same length.")

	pi = np.pi
	n = np.arange(1, n_coeffs + 1)

	# Calculate individual terms
	t1 = (2 + kappa) * (1 + lambd) / 2

	t2 = pi ** 2 / 2 * sum(i ** 2 * (a_n[i - 1] * b_n[i - 1] + lambd * b_n[i - 1] ** 2) for i in n)

	t3 = 2 * kappa / pi * sum((a_n[i - 1] - b_n[i - 1]) / i for i in n if i % 2 == 1)

	t4 = 2 * kappa * sum((a_n[i - 1] + lambd * b_n[i - 1]) * b_n[j - 1] * i * j / (i * i - j * j)
	                     for i in n for j in n if (i + j) % 2 == 1)

	total_loss = lambd * (t1 + t2 + t3 + t4)
	if verbose:
		print("APPROX TOTAL COST FUNCTION FROM APPROX FORMULA:")
		print("int_I: ", t1)
		print("int_II: ", t2)
		print("int_III: ", t3)
		print("int_IV: ", t4)

		print("Loss function approximation formula: ", total_loss)

	return total_loss


# Exact cost computation function
def compute_integral_loss_function_direct(a_func, b_func, a_func_dot, b_func_dot, kappa, lambd):
	"""
	This computes the intgral of the loss function directly by integrating the
	functions of the trading strategies and their derivatives.

	WARNING: the derivatives of the functions are not che'cked for correctness.|
	"""

	# I: \int_0^1 \dot{a} \dot{a} dt
	def integrand_I(t):
		a_dot = a_func_dot(t, kappa, lambd)
		return a_dot * a_dot

	# II: \int_0^1 lambda \dot{b} \dot{a} dt
	def integrand_II(t):
		a_dot = a_func_dot(t, kappa, lambd)
		b_dot = b_func_dot(t, kappa, lambd)
		return lambd * b_dot * a_dot

	# III: \int_0^1 kappa a \dot{a} dt
	def integrand_III(t):
		a_dot = a_func_dot(t, kappa, lambd)
		return kappa * a_func(t, kappa, lambd) * a_dot

	# IV: \int_0^1 kappa lambda b \dot{a} dt
	def integrand_IV(t):
		a_dot = a_func_dot(t, kappa, lambd)
		return kappa * lambd * b_func(t, kappa, lambd) * a_dot

	int_I, _ = quad(integrand_I, 0, 1)
	int_II, _ = quad(integrand_II, 0, 1)
	int_III, _ = quad(integrand_III, 0, 1)
	int_IV, _ = quad(integrand_IV, 0, 1)

	integral = int_I + int_II + int_III + int_IV

	# Dissection of integral IV into four components

	def t_fn(t):
		return lambd * kappa * t

	def int_IV_component2(t):
		b_t = b_func(t, kappa, lambd)
		return lambd * kappa * (b_t - t)

	def int_IV_component3(t):
		a_dot_t = a_func_dot(t, kappa, lambd)
		return lambd * kappa * t * (a_dot_t - 1)

	#  return lambd * kappa * (a_dot_t - 1) * (b_t - 1)
	def int_IV_component4(t):
		a_dot_t = a_func_dot(t, kappa, lambd)
		b_t = b_func(t, kappa, lambd)
		return lambd * kappa * (b_t - t) * (a_dot_t - 1)

	int_IV_1, _ = quad(t_fn, 0, 1, limit=100)
	int_IV_2, _ = quad(int_IV_component2, 0, 1, limit=100)
	int_IV_3, _ = quad(int_IV_component3, 0, 1, limit=100)
	int_IV_4, _ = quad(int_IV_component4, 0, 1, limit=100)

	print("INTEGRATION OF LOSS FUNCTION FROM ORIGINAL FUNCTIONS")
	print("int_I: ", int_I)
	print("int_II: ", int_II)
	print("int_III: ", int_III)
	print("int_IV: ", int_IV)
	print("\tint_0^1 lambda kappa t dt: ", int_IV_1)
	print("\tint_0^1 lambda kappa (b(t)-t) dt: ", int_IV_2)
	print("\tint_0^1 lambda kappa t * (a'(t)-1) dt: ", int_IV_3)
	print("\tint_0^1 lambda kappa (b(t) - 1)*(a'(t) - 1) dt: ", int_IV_4)
	print("\tTotal of components: ", int_IV_1 + int_IV_2 + int_IV_3 + int_IV_4)

	print("Exact integral of loss function: ", integral)

	return integral


def fourier_integral_cost_fn(a_coeffs, b_coeffs, kappa, lambd):
	"""
	Compute the integral I = ∫₀¹ [ (a_dot + λ b_dot) a_dot + kappa (a + λ b) a_dot ] dt

	where each of a, b, a_dot, b_dot is formed using the coefficients passed in:

	a(t) = t + sum a_n sin(n pi t)
	b(t) = t + sum b_n sin(n pi t)
	a_dot(t) = 1 + sum a_n n pi cos(n pi t)
	b_dot(t) = 1 + sum b_n n pi cos(n pi t)

	Integral components:

	Int_IV: \\$int_0^1$

	Purpose:

	Allows us to check if integrating the loss function formed from the Fourier approximations of
	the strategies and their derivatives yields the same result as integrating the loss function
	itself.

	Parameters:
	a_coeffs (list or array): Coefficients aₙ for n = 1 to N
	b_coeffs (list or array): Coefficients bₙ for n = 1 to N
	lambd (float): Constant λ
	kappa (float): Constant κ (not used in this integral but included as per the request)

	Returns:
	float: The computed value of the integral I
	"""
	# Ensure coefficients are numpy arrays
	a_coeffs = np.array(a_coeffs, dtype=np.float64)
	b_coeffs = np.array(b_coeffs, dtype=np.float64)
	N = len(a_coeffs)

	if len(b_coeffs) != N:
		raise ValueError("Sequences a_coeffs and b_coeffs must be of the same length.")

	pi = np.pi

	# Define the functions a(t), b(t), a_dot(t), b_dot(t)
	def a(t):
		n = np.arange(1, N + 1)  # 1-based indexing
		terms = a_coeffs * np.sin(n * pi * t)
		return t + np.sum(terms)

	def b(t):
		n = np.arange(1, N + 1)
		terms = b_coeffs * np.sin(n * pi * t)
		return t + np.sum(terms)

	def a_dot(t):
		n = np.arange(1, N + 1)
		terms = a_coeffs * n * pi * np.cos(n * pi * t)
		return 1 + np.sum(terms)

	def b_dot(t):
		n = np.arange(1, N + 1)
		terms = b_coeffs * n * pi * np.cos(n * pi * t)
		return 1 + np.sum(terms)

	def a_b_dot(t):
		n = np.arange(1, N + 1)
		m = np.arange(1, N + 1)
		terms_b = b_coeffs * np.sin(m * pi * t)
		terms_a_dot = a_coeffs * n * pi * np.cos(n * pi * t)
		return terms_b * terms_a_dot

	# Components of the loss function

	# a'(t) * a'(t) (temporary impact)
	def int_adot_adot(t):
		a_t = a(t)
		b_t = b(t)
		a_dot_t = a_dot(t)
		b_dot_t = b_dot(t)
		return a_dot_t * a_dot_t

	# lambda b'(t) * a'(t) (temporary impact)
	def int_adot_lam_bdot(t):
		a_dot_t = a_dot(t)
		b_dot_t = b_dot(t)
		return lambd * b_dot_t * a_dot_t

	def int_kappa_a_a_dot(t):
		a_t = a(t)
		b_t = b(t)
		a_dot_t = a_dot(t)
		b_dot_t = b_dot(t)
		return kappa * a_t * a_dot_t

	# kappa * lambda * b(t) * a'(t)
	def int_kappa_lambd_b_a_dot(t):
		b_t = b(t)
		a_dot_t = a_dot(t)
		return kappa * lambd * b_t * a_dot_t

	int_I, error = quad(int_adot_adot, 0, 1, limit=100)
	int_II, error = quad(int_adot_lam_bdot, 0, 1, limit=100)
	int_III, error = quad(int_kappa_a_a_dot, 0, 1, limit=100)
	int_IV, error = quad(int_kappa_lambd_b_a_dot, 0, 1, limit=100)

	# Dissection of integral IV into four components

	def t_fn(t):
		return lambd * kappa * t

	def int_IV_component2(t):
		b_t = b(t)
		return lambd * kappa * (b_t - t)

	int_IV_2, _ = quad(int_IV_component2, 0, 1, limit=100)

	def int_IV_component3(t):
		a_dot_t = a_dot(t)
		return lambd * kappa * t * (a_dot_t - 1)

	def int_IV_component4(t):
		a_dot_t = a_dot(t)
		b_t = b(t)
		return lambd * kappa * (a_dot_t - 1) * (b_t - t)

	int_IV_1, _ = quad(t_fn, 0, 1, limit=100)
	int_IV_2, _ = quad(int_IV_component2, 0, 1, limit=100)
	int_IV_3, _ = quad(int_IV_component3, 0, 1, limit=100)
	int_IV_4, _ = quad(int_IV_component4, 0, 1, limit=100)

	integral = int_I + int_II + int_III + int_IV

	print("FOURIER INTEGRAL VALUES")
	print("int_I (int_0^1 a'(t) * a'(t) dt): ", int_I)
	print("int_II: (int_0^1 lambda a'(t) * b'(t) dt): ", int_II)
	print("int_III: (int_0^1 kappa a(t) * a'(t) dt): ", int_III)
	print("int_IV: (int_0^1 kappa * lambda * b(t) * a'(t) dt): ", int_IV)
	print("\tint_0^1 lambda kappa t dt: ", int_IV_1)
	print("\tint_0^1 lambda kappa b(t) dt: ", int_IV_2)
	print("\tint_0^1 lambda kappa t * a'(t) dt: ", int_IV_3)
	print("\tint_0^1 lambda kappa (a'(t)-1)(b(t)-t) dt: ", int_IV_4)
	print("\tTotal of components: ", int_IV_1 + int_IV_2 + int_IV_3 + int_IV_4)
	print("Integral of the loss function computed from Fourier approximations:", integral)

	return integral


def compute_sine_series_for_functions(a_func, b_func, kappa, lambd, N):
	"""
	Compute the sine series coefficients a_n and b_n for functions a_func and b_func.

	Parameters:
	a_func: function(t, lambd, kappa, N)
		The function a_func(t, lambd, kappa, N) to be expanded.
	b_func: function(t, lambd, kappa, N)
		The function b_func(t, lambd, kappa, N) to be expanded.
	lambd: float
		Parameter lambda.
	kappa: float
		Parameter kappa.
	N: int
		Number of terms in the series.

	Returns:
	a_coeffs: numpy array
		The sine series coefficients for a_func.
	b_coeffs: numpy array
		The sine series coefficients for b_func.
	"""
	pi = np.pi
	a_coeffs = np.zeros(N)
	b_coeffs = np.zeros(N)

	for n in range(1, N + 1):
		# Compute coefficients for a_func
		def integrand_a(t):
			return 2 * (a_func(t, kappa, lambd) - t) * np.sin(n * pi * t)

		coeff_a, _ = quad(integrand_a, 0, 1)
		a_coeffs[n - 1] = coeff_a

		# Compute coefficients for b_func
		def integrand_b(t):
			return 2 * (b_func(t, kappa, lambd) - t) * np.sin(n * pi * t)

		coeff_b, _ = quad(integrand_b, 0, 1)
		b_coeffs[n - 1] = coeff_b

	return a_coeffs, b_coeffs


# Example functions that satisfy f(0) = 0 and f(1) = 1
def a_func(t, kappa, lambd):
	# Example function depending on lambda and kappa
	# Satisfies boundary conditions f(0) = 0, f(1) = 1
	return t ** 2 + lambd * np.sin(np.pi * t)


# Derivative of a_func (WARNING: there is no check to see if this is it!)
def a_func_dot(t, kappa, lambd):
	return 2 * t + np.pi * lambd * np.cos(np.pi * t)


def b_func(t, kappa, lambd):
	# Another example function depending on lambda and kappa
	# Satisfies boundary conditions f(0) = 0, f(1) = 1
	return t ** 3 + kappa * np.sin(2 * np.pi * t)


# Derivative of b_func (WARNING: there is no check to see if this is it!)
def b_func_dot(t, kappa, lambd):
	# Another example function depending on lambda and kappa
	# Satisfies boundary conditions f(0) = 0, f(1) = 1
	return 3 * t ** 2 + 2 * np.pi * kappa * np.cos(2 * np.pi * t)


# TESTING THE FUNCTIONS
if __name__ == "__main__":
	# Example coefficients
	a_coeffs = [1.0, 0.5]  # Replace with your coefficients
	b_coeffs = [-1.0, -5.0]  # Replace with your coefficients
	lambd = 1  # Replace with your value
	kappa = 1  # Replace with your value (not used in this integral)
	N = 2  # Number of terms in the series

	# Compute the sine series coefficients
	# a_coeffs, b_coeffs = compute_sine_series_for_functions(a_func, b_func, kappa, lambd, N)

	I = fourier_integral_cost_fn(a_coeffs, b_coeffs, kappa, lambd)
	print("---> The value of the Fourier actual cost fn is:", I)
	print()

	I = compute_integral_loss_function_direct(a_func, b_func, a_func_dot, b_func_dot, kappa, lambd)
	print("---> The value of the exact integrated actual cost fn is:", I)
	print()

	I = cost_fn_a_approx(a_coeffs, b_coeffs, kappa, lambd, verbose=True)

	print("---> The value of the totally new approximate cost fn is:", I)
	print()

	I = cost_fn_a_approx(a_coeffs, b_coeffs, kappa, lambd, verbose=True)

	print("---> The value of the formula the 9/16 paper is:", I)
	print()
