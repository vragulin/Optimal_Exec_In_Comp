import numpy as np
from scipy.integrate import quad
from typing import Callable, List, Tuple


def sin_coeff(func: Callable[[float], float], n: int,
              a: float = 0, b: float = 1) -> np.ndarray:
	""" Calculate coefficients of a sine series """
	coeffs = np.zeros(n)
	for i in range(1, n + 1):
		coeffs[i - 1] = quad(func, a, b, weight='sin', wvar=(i * np.pi))[0]
	return 2 * coeffs


def find_fourier_coefficients(functions: List[Callable[[float, float, float], float]],
                              kappa: float, lambda_: float, n: int, gamma: float = 1) -> List[np.ndarray]:
	all_coeffs = []
	for func in functions:
		coeffs = sin_coeff(lambda t: (func(t, kappa, lambda_) - t * gamma), n)
		all_coeffs.append(coeffs)
	return all_coeffs


def reconstruct_from_sin_with_loop(t: float, coeffs: np.ndarray, n: [int | None] = None) -> float:
	"""
    Reconstructs the function value at a given point `t` from a vector of sine coefficients.
	This is a slow version that uses loops.  It's replaced with a faster version

    :param t: The point at which to evaluate the reconstructed function.
    :param coeffs: An array of sine coefficients.
    :param n: The number of sine terms to use in the reconstruction. If None or greater than the length of `coeffs`,
              all coefficients are used.
    :return: The reconstructed function value at the point `t`.

    Notes:
    ------
    The function reconstructs the value by summing the sine series up to the `n`-th term.
    If `n` is not specified or exceeds the length of `coeffs`, it defaults to using all available coefficients.
    """
	reconstruction = 0  # Add the linear term t separately
	if n is None or n > len(coeffs):
		n = len(coeffs)
	for i in range(1, n + 1):
		reconstruction += coeffs[i - 1] * np.sin(i * np.pi * t)
	return reconstruction


def reconstruct_deriv_from_sin_with_loop(t: float, coeffs: np.ndarray, n: [int | None] = None, order: int = 1) -> float:
	"""
    Reconstructs the nth derivative of the function value at a given point `t` from a vector of sine coefficients.
	This is a slow version that uses loops.  It's replaced with a faster version

    :param t: The point at which to evaluate the reconstructed function.
    :param coeffs: An array of sine coefficients.
    :param n: The number of sine terms to use in the reconstruction. If None or greater than the length of `coeffs`,
              all coefficients are used.
    :param order: The order of the derivative to reconstruct.
    :return: The reconstructed nth derivative of the function value at the point `t`.

    Notes:
    ------
    The function reconstructs the nth derivative by summing the sine series up to the `n`-th term and applying the
    appropriate derivative formula for sine functions.
    If `n` is not specified or exceeds the length of `coeffs`, it defaults to using all available coefficients.
    """
	reconstruction = 0
	if n is None or n > len(coeffs):
		n = len(coeffs)
	for i in range(1, n + 1):
		reconstruction += coeffs[i - 1] * (i * np.pi) ** order * np.sin(i * np.pi * t + order * np.pi / 2)
	return reconstruction


def reconstruct_from_sin(t: float, coeffs: np.ndarray, n: [int | None] = None,
                         sin_values: [np.ndarray | None] = None) -> float:
	"""
    Reconstructs the function value at a given point `t` from a vector of sine coefficients.
	This is a slow version that uses loops.  It's replaced with a faster version

    :param t: The point at which to evaluate the reconstructed function.
    :param coeffs: An array of sine coefficients.
    :param n: The number of sine terms to use in the reconstruction. If None or greater than the length of `coeffs`,
              all coefficients are used.
    :param sin_values: Pre-computed values of sin(n * pi * t) as an array of the same size as coeffs for speedup
    :return: The reconstructed function value at the point `t`.

    Notes:
    ------
    The function reconstructs the value by summing the sine series up to the `n`-th term.
    If `n` is not specified or exceeds the length of `coeffs`, it defaults to using all available coefficients.
    """
	if n is None or n > len(coeffs):
		n = len(coeffs)
	i = np.arange(1, n + 1)
	if sin_values is None:
		sin_values = np.sin(i * np.pi * t)
	return coeffs[:n] @ sin_values


def reconstruct_deriv_from_sin(t: float, coeffs: np.ndarray, n: [int | None] = None, order: int = 1,
                               trig_values: [np.ndarray | None] = None) -> float:
	"""
    Reconstructs the nth derivative of the function value at a given point `t` from a vector of sine coefficients.
	This is a slow version that uses loops.  It's replaced with a faster version

    :param t: The point at which to evaluate the reconstructed function.
    :param coeffs: An array of sine coefficients.
    :param n: The number of sine terms to use in the reconstruction. If None or greater than the length of `coeffs`,
              all coefficients are used.
    :param order: The order of the derivative to reconstruct.
    :param trig_values: Pre-computed values of f(n * pi * t) as an array of the same size as coeffs for speedup
                        Where f = cos() for first deriv, -sin() for 2nd deriv etc.
                        It's the responsiblity of the user to ensure that the right values have been passed.
    :return: The reconstructed nth derivative of the function value at the point `t`.

    Notes:
    ------
    The function reconstructs the nth derivative by summing the sine series up to the `n`-th term and applying the
    appropriate derivative formula for sine functions.
    If `n` is not specified or exceeds the length of `coeffs`, it defaults to using all available coefficients.
    """
	if n is None or n > len(coeffs):
		n = len(coeffs)

	i = np.arange(1, n + 1)

	if trig_values is None:
		trig_values = np.sin(i * np.pi * t + order * np.pi / 2)
	return np.sum(coeffs[:n] * np.power(i * np.pi, order) * trig_values)
