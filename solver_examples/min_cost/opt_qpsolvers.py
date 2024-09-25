""" Minimize the cost function using the qpsolvers package.
    V. Ragulin, 24-Sep-2024
"""
import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import minimize
from typing import List
import time

# *** Global Parameters ***
ABS_TOL = 1e-6
N_TERMS = 2
DECAY = 0.95
DO_QP = True
DO_SCI = True


def cost_fn_a_approx(a_n: List[float], b_n: List[float], kappa: float, lambd: float, verbose: bool = False) -> float:
    a_n = np.array(a_n, dtype=np.float64)
    b_n = np.array(b_n, dtype=np.float64)
    n_coeffs = len(a_n)

    if len(b_n) != n_coeffs:
        raise ValueError("Sequences a_n and b_n must be of the same length.")

    pi = np.pi
    n = np.arange(1, n_coeffs + 1)

    t1 = (2 + kappa) * (1 + lambd) / 2
    t2 = pi ** 2 / 2 * sum(i ** 2 * (a_n[i - 1] ** 2 + lambd * a_n[i - 1] * b_n[i - 1]) for i in n)
    t3 = 2 * kappa * lambd / pi * sum((b_n[i - 1] - a_n[i - 1]) / i for i in n if i % 2 == 1)
    t4 = 2 * kappa * sum((a_n[i - 1] + lambd * b_n[i - 1]) * a_n[j - 1] * i * j / (i * i - j * j)
                         for i in n for j in n if (i + j) % 2 == 1)

    total_loss = t1 + t2 + t3 + t4

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


def minimize_qpsolvers(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                       abs_tol: float = 1e-6) -> np.ndarray:
    n_coeffs = len(a_n)
    H = np.zeros((n_coeffs, n_coeffs))
    f = np.zeros(n_coeffs)

    for i in range(n_coeffs):
        for j in range(n_coeffs):
            if i == j:
                H[i, j] = (2 + kappa) * (1 + lambd) / 2 + (np.pi ** 2 / 2) * (i + 1) ** 2
            else:
                H[i, j] = 2 * kappa * (i + 1) * (j + 1) / ((i + 1) ** 2 - (j + 1) ** 2)

        f[i] = 2 * kappa * lambd / np.pi * (b_n[i] - a_n[i]) / (i + 1)

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=SOLVER)

    return a_n_opt


def minimize_qpsolvers_1(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                         abs_tol: float = 1e-6) -> np.ndarray:
    n_coeffs = len(a_n)
    H = np.zeros((n_coeffs, n_coeffs))
    f = np.zeros(n_coeffs)

    pi = np.pi

    H[0, 0] = pi * pi
    f[0] = lambd * (pi * pi * b_n[0] / 2 - 2 * kappa / pi)

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=SOLVER)

    return a_n_opt


def minimize_qpsolvers_2(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                         abs_tol: float = 1e-6) -> np.ndarray:
    n_coeffs = len(a_n)
    H = np.zeros((n_coeffs, n_coeffs))
    f = np.zeros(n_coeffs)

    pi = np.pi
    n = np.arange(1, n_coeffs + 1)

    # The Hessian matrix
    H[0, 0] = pi * pi
    H[1, 1] = 4 * pi * pi

    # Build f as the sum of 4 terms.
    # The impact of the first term is zero
    # The impact of the second term:
    n_sq = (n ** 2)
    f2 = pi * pi / 2 * lambd * n_sq * b_n

    # The impact of the third term:
    n_odd_idx = (n % 2)
    f3 = - 2 * kappa * lambd / pi * n_odd_idx / n

    # The impact of the fourth term:
    m_p_n_odd = (n[:, None] + n[None , :]) % 2
    mn = n[:, None] @ n[None, :]
    msq_nsq = n_sq[:, None] - n_sq[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(m_p_n_odd, mn / msq_nsq, 0)

    f4 = 2 * kappa * lambd * M @ b_n

    # Add all terms to f
    f = f2 + f3 + f4

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=SOLVER)

    return a_n_opt


def minimize_qpsolvers(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                         abs_tol: float = 1e-6) -> np.ndarray:
    n_coeffs = len(a_n)
    H = np.zeros((n_coeffs, n_coeffs))
    f = np.zeros(n_coeffs)

    pi = np.pi
    n = np.arange(1, n_coeffs + 1)

    # The Hessian matrix
    H[0, 0] = pi * pi
    H[1, 1] = 4 * pi * pi

    # Build f as the sum of 4 terms.
    # The impact of the first term is zero
    # The impact of the second term:
    n_sq = (n ** 2)
    f2 = pi * pi / 2 * lambd * n_sq * b_n

    # The impact of the third term:
    n_odd_idx = (n % 2)
    f3 = - 2 * kappa * lambd / pi * n_odd_idx / n

    # The impact of the fourth term:
    m_p_n_odd = (n[:, None] + n[None , :]) % 2
    mn = n[:, None] @ n[None, :]
    msq_nsq = n_sq[:, None] - n_sq[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(m_p_n_odd, mn / msq_nsq, 0)

    f4 = 2 * kappa * lambd * M @ b_n

    # Add all terms to f
    f = f2 + f3 + f4

    a_n_opt = solve_qp(H, f, None, None, None, None, solver=SOLVER)

    return a_n_opt


def minimize_scipy(a_n: List[float], b_n: List[float], kappa: float, lambd: float,
                   abs_tol: float = 1e-6) -> np.ndarray:
    a_n = np.array(a_n, dtype=np.float64)
    b_n = np.array(b_n, dtype=np.float64)

    # Define the objective function for minimize
    def objective(a_n: np.ndarray) -> float:
        return cost_fn_a_approx_simplified(a_n, b_n, kappa, lambd)

    # Initial guess for a_n
    initial_guess = np.zeros_like(a_n)

    # Call the minimize function
    result = minimize(objective, initial_guess, method='BFGS', tol=abs_tol)

    return result.x


# Example usage
#
a_n = np.zeros(N_TERMS)
b_n = np.zeros(N_TERMS)  # b_n = [np.cos(2*i) * np.exp(-DECAY*i) for i in range(N_TERMS)]
kappa = 10
lambd = 20
SOLVER = 'daqp'  # 'quadprog'

print(f"Initial a_n: {a_n}")
print(f"Initial b_n: {b_n}")

# Solve using qpsolvers
if DO_QP:
    start_time = time.time()
    match N_TERMS:
        case 1:
            a_n_opt_qp = minimize_qpsolvers_1(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        case 2:
            a_n_opt_qp = minimize_qpsolvers_2(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
        case _:
            a_n_opt_qp = minimize_qpsolvers_2(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
    end_time = time.time()
    print(f"Optimal a_n using qpsolvers: {a_n_opt_qp}")
    print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")
    print(f"Objective function value (qpsolvers): {cost_fn_a_approx_simplified(a_n_opt_qp, b_n, kappa, lambd):.4f}")
    print(f"qpsolvers time taken: {end_time - start_time:.4f} seconds")

# Solve using scipy.optimize.minimize
if DO_SCI:
    start_time = time.time()
    a_n_opt_sci = minimize_scipy(a_n, b_n, kappa, lambd, abs_tol=ABS_TOL)
    end_time = time.time()
    print("Optimal a_n using scipy:", a_n_opt_sci)
    print(f"scipy time taken: {end_time - start_time:.4f} seconds")
    print(f"Objective function value (scipy): {cost_fn_a_approx_simplified(a_n_opt_sci, b_n, kappa, lambd):.4f}")
    print(f"scipy time taken: {end_time - start_time:.4f} seconds")
