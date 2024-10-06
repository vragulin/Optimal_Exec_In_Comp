""" Esimates coefficients of a quadratic function from a set of points.
    V. Ragulin - 10/03/2024
"""

import numpy as np
from typing import Callable


def evaluate_S(x):
    # Replace this function with the actual evaluation of S(x)
    # For example, S(x) = x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2 + 4*x[0] + 5*x[1] + 6
    return x[0] ** 2 + 2 * x[0] * x[1] + 3 * x[1] ** 2 + 4 * x[0] + 5 * x[1] + 6


def find_coefficients0(func: Callable, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    H = np.zeros((n, n))
    f = np.zeros(n)
    C = func(np.zeros(n))

    # Find quadratic coefficients H
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        H[i, i] = func(2 * e_i) - 2 * func(e_i) + C
        for j in range(i + 1, n):
            e_j = np.zeros(n)
            e_j[j] = 1
            H[i, j] = H[j, i] = (func(e_i + e_j) - func(e_i) - func(e_j) + C)

    # Find linear coefficients f
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        f[i] = func(e_i) - C - 0.5 * H[i, i]

    return H, f, C


def find_coefficients(func: Callable, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    H = np.zeros((n, n))
    f = np.zeros(n)
    C = func(np.zeros(n))

    # Precompute evaluations for standard basis vectors
    evaluations = np.zeros((n, 2))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        evaluations[i, 0] = func(e_i)
        evaluations[i, 1] = func(2 * e_i)

    # Find quadratic coefficients H
    for i in range(n):
        H[i, i] = evaluations[i, 1] - 2 * evaluations[i, 0] + C
        for j in range(i + 1, n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            H[i, j] = H[j, i] = func(e_i + e_j) - evaluations[i, 0] - evaluations[j, 0] + C

    # Find linear coefficients f
    for i in range(n):
        f[i] = evaluations[i, 0] - 0.5 * H[i, i] - C

    return H, f, C


# Example usage
if __name__ == "__main__":
    n = 2  # Dimension of the vector x
    H, f, C = find_coefficients(evaluate_S,2)
    print("H:", H)
    print("f:", f)
    print("C:", C)
