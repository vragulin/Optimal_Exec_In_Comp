"""  Scripts to convert the cost function to a quadratic program
    V. Ragulin - 10/05/2024
"""
import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'matrix_utils')))
from propagator import cost_fn_prop_a_matrix
from est_quad_coeffs import find_coefficients


def cost_to_QP_numerical(a_n: np.ndarray, b_n: np.ndarray, lambd: float, rho: float
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """ Convert the cost function to a quadratic program
    :param a_n: Fourier coefficients of the trading rate a(t)
    :param b_n: Fourier coefficients of the trading rate b(t)
    :param lambd: size of trader B
    :param rho: decay of the propagator
    :return: Quadratic coefficients H, linear coefficients f, constant term C
    """
    # Find the quadratic coefficients
    n = len(a_n)
    S = lambda x: cost_fn_prop_a_matrix(x, b_n, lambd, rho)
    H, f, C = find_coefficients(S, n)
    return H, f, C


def cost_QP_params(b_n: np.ndarray, lambd: float, rho: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """ Calculte the quadratic coefficients of the cost function using an analytical formula
    :param a_n: Fourier coefficients of the trading rate a(t)
    :param b_n: Fourier coefficients of the trading rate b(t)
    :param lambd: size of trader B
    :param rho: decay of the propagator
    :return: Quadratic coefficients H, linear coefficients f
    """

    pi, exp = np.pi, np.exp

    # Constants (later can be precomputed)
    n = np.arange(1, len(b_n) + 1).reshape([-1, 1])
    N = np.diag(n.reshape(-1)).astype(float)
    n_sq = n ** 2
    i = np.ones(n.shape)
    i_odd = n % 2
    i_mp = i - 2 * i_odd

    m_p_n_odd = (n + n.T) % 2
    msq_nsq = n_sq - n_sq.T
    with np.errstate(divide='ignore', invalid='ignore'):
        M = 2 * np.where(m_p_n_odd, n_sq / msq_nsq, 0)

    D = np.diag((pi * n / (rho ** 2 + (n * pi) ** 2)).reshape(-1))
    h = D @ (i - exp(-rho) * i_mp)

    # Calluate the quadratic coefficients (Hessian)
    H = -rho ** 2 * (D @ i @ h.T + h @ i.T @ D) + pi * rho * N @ D + pi * (N @ M.T @ D + D @ M @ N)

    # Calculate the linear coefficients (gradient)
    f = D @ (exp(-rho) * i - i_mp) - (1 + lambd + rho ** 2 * lambd * i.T @ D @ b_n).item() * h \
        + pi * lambd * (0.5 * rho * N @ D + N @ M.T @ D) @ b_n.reshape([-1, 1])

    return H, f.reshape(-1)


if __name__ == '__main__':
    # Example
    a_n = np.array([1, 2, 3, 4])
    b_n = np.array([2, 3, 7, 8])
    lambd = 1
    rho = 10

    H, f, C = cost_to_QP_numerical(b_n, lambd, rho)
    print(H)
    print(f)
    print(C)

    # Test that the cost function is correctly represented as a quadratic program
    x = a_n
    cost_from_func = cost_fn_prop_a_matrix(x, b_n, lambd, rho)
    cost_from_QP = 0.5 * x.T @ H @ x + f.T @ x + C
    assert np.isclose(cost_from_func, cost_from_QP), "Cost function is not correctly represented as a QP"

    print("Done")
