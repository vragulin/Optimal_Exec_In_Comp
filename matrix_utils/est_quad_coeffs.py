""" Esimates coefficients of a quadratic function from a set of points.
    V. Ragulin - 10/03/2024
"""

import numpy as np


def evaluate_S(x):
    # Replace this function with the actual evaluation of S(x)
    # For example, S(x) = x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2 + 4*x[0] + 5*x[1] + 6
    return x[0] ** 2 + 2 * x[0] * x[1] + 3 * x[1] ** 2 + 4 * x[0] + 5 * x[1] + 6


def find_coefficients(n):
    H = np.zeros((n, n))
    f = np.zeros(n)
    C = evaluate_S(np.zeros(n))

    # Find linear coefficients f
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        f[i] = evaluate_S(e_i) - C

    # Find quadratic coefficients H
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        H[i, i] = evaluate_S(2 * e_i) - 2 * evaluate_S(e_i) + C
        for j in range(i + 1, n):
            e_j = np.zeros(n)
            e_j[j] = 1
            H[i, j] = H[j, i] = (evaluate_S(e_i + e_j) - evaluate_S(e_i) - evaluate_S(e_j) + C) / 2

    return H, f, C


# Example usage
if __name__ == "__main__":
    n = 2  # Dimension of the vector x
    H, f, C = find_coefficients(n)
    print("H:", H)
    print("f:", f)
    print("C:", C)
