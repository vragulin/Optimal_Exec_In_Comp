""" Test execution cost
"""
import pytest as pt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../../representation"))
sys.path.append(os.path.abspath("../../optimizer"))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../cost_function')))
import trading_funcs as tf
from strategy_class import EncodedStrategy


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
