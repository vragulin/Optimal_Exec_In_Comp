""" Test execution cost
"""
import pytest as pt
from strategy_class import EncodedStrategy
import numpy as np


def test_cost_two_risk_neutral():
    st_a, st_b = EncodedStrategy(), EncodedStrategy(lambd=6)
    st_a.encode_power(1)
    st_b.encode_power(1)

    cost = EncodedStrategy.cost(st_a, st_b)
    expected = 7.0
    assert pt.approx(cost, abs=1e-2) == expected


def test_cost_power_passive():
    st_a, st_b = EncodedStrategy(), EncodedStrategy(lambd=6, kappa=1)
    st_a.encode_power(1)
    st_b.encode_passive(6)

    cost = EncodedStrategy.cost(st_a, st_b)
    expected = 8.49505
    assert pt.approx(cost, abs=1e-2) == expected

