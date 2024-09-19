import pytest as pt
from strategy_class import EncodedStrategy
import numpy as np


@pt.mark.parametrize("n", range(1, 5))
def test_encode_power(n):
    def func(t):
        return t ** n

    # Create a strategy
    strat = EncodedStrategy(n_terms=10)
    strat.encode_power(n)

    t = 0.5  # Example point to evaluate the function

    # Call the function
    result = strat.reconstruct(t)
    expected= func(t)

    # Assert the result is close to the expected value
    assert np.isclose(result, expected, atol=1e-3), f"Expected {expected}, but got {result}"


@pt.mark.parametrize("c", [0.1, 0.5, 0.9])
def test_encode_parabolic(c):
    def func(t):
        return t * (t - c) / (1 - c)

    # Create a strategy
    strat = EncodedStrategy(n_terms=15)
    strat.encode_parabolic(c)

    t = 0.5  # Example point to evaluate the function

    # Call the function
    result = strat.reconstruct(t)
    expected= func(t)

    # Assert the result is close to the expected value
    assert np.isclose(result, expected, atol=1e-3), f"Expected {expected}, but got {result}"


@pt.mark.parametrize("sigma", [0.1, 1.0, 10.0])
def test_encode_passive(sigma):
    def func(t):
        return np.sinh(sigma * t) / np.sinh(sigma)

    # Create a strategy
    strat = EncodedStrategy(n_terms=15)
    strat.encode_passive(sigma)

    t = 0.5  # Example point to evaluate the function

    # Call the function
    result = strat.reconstruct(t)
    expected= func(t)

    # Assert the result is close to the expected value
    assert np.isclose(result, expected, atol=1e-3), f"Expected {expected}, but got {result}"
