import numpy as np
import pytest
from matrix_utils.est_quad_coeffs import find_coefficients, find_coefficients0
from typing import Any
import time


def evaluate_quad(x: np.ndarray, H: np.ndarray, f: np.ndarray, C: float) -> Any:
    return 0.5 * x.T @ H @ x + f.T @ x + C


def test_find_coefficients():
    n = 2

    H = np.array([[1, 2], [2, 3]])
    f = np.array([4, 5])
    C = 6

    def S(x):
        return evaluate_quad(x, H, f, C)

    H1, f1, C1 = find_coefficients(S, n)

    # Expected values for the example polynomial
    assert np.allclose(H1, H), f"Expected H: {H}, but got: {H1}"
    assert np.allclose(f1, f), f"Expected f: {f}, but got: {f1}"
    assert np.isclose(C1, C), f"Expected C: {C}, but got: {C1}"


def test_find_coefficients_zero():
    def evaluate_S_zero(x):
        return 0

    n = 3
    H, f, C = find_coefficients(evaluate_S_zero, n)

    expected_H = np.zeros((n, n))
    expected_f = np.zeros(n)
    expected_C = 0

    assert np.allclose(H, expected_H), f"Expected H: {expected_H}, but got: {H}"
    assert np.allclose(f, expected_f), f"Expected f: {expected_f}, but got: {f}"
    assert np.isclose(C, expected_C), f"Expected C: {expected_C}, but got: {C}"


def test_find_coefficients_linear():
    def S(x):
        return 3 * x[0] + 4 * x[1] + 5

    n = 2
    H, f, C = find_coefficients(S, n)

    expected_H = np.zeros((n, n))
    expected_f = np.array([3, 4])
    expected_C = 5

    assert np.allclose(H, expected_H), f"Expected H: {expected_H}, but got: {H}"
    assert np.allclose(f, expected_f), f"Expected f: {expected_f}, but got: {f}"
    assert np.isclose(C, expected_C), f"Expected C: {expected_C}, but got: {C}"


def test_compare_time():
    n = 50
    n_repetitions = 1000

    np.random.seed(12)
    H = np.random.randn(n, n)
    f = np.random.randn(n)
    C = np.random.randn()

    def S(x):
        return evaluate_quad(x, H, f, C)

    # Measure execution time of f1
    start_time = time.perf_counter()
    for _ in range(n_repetitions):
        res0 = find_coefficients0(S, n)
    end_time = time.perf_counter()
    t0 = end_time - start_time
    print(f"Execution time of find_coeff0: {t0} seconds")

    # Measure execution time of f2
    start_time = time.perf_counter()
    for _ in range(n_repetitions):
        res = find_coefficients(S, n)
    end_time = time.perf_counter()
    t = end_time - start_time
    print(f"Execution time of find_coeff: {t} seconds")

    for x, x0 in zip(res, res0):
        assert np.allclose(x, x0), (f"Expected the results to be the same, "
                                    f"but got\nres={x} and\nres0={x0}")
    assert t < t0, (f"Expected the optimized version to be faster, "
                    f"but got t(t)={t} and t(f0)={t0}")
