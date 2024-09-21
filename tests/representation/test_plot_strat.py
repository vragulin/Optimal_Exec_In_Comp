import os
import sys
sys.path.append(os.path.abspath("../../representation"))
sys.path.append(os.path.abspath("../../cost_function"))
from strategy_class import EncodedStrategy


def test_plot_parabolic():
    strat = EncodedStrategy(n_terms=10)
    strat.encode_parabolic(0.5)

    res = strat.plot()
    assert len(res) == 2


def test_plot_passive():
    strat = EncodedStrategy(n_terms=10)
    strat.encode_passive(0.001)

    res = strat.plot()
    assert len(res) == 2