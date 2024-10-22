"""  A simple test script to plot a SinesBlocks strategy
    V. Ragulin, 10/19/2024
"""
import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..')))

import prop_blocks as pb

if __name__ == '__main__':
    N = 3

    # Initialize trader A
    a_s, a_e = 0.6, 0.2
    # a_n = np.random.rand(N) * 0.2
    a_n = np.array([-0.3, 0.3, 0.1])
    strat_a = pb.SinesBlocks(N, (a_s, a_e), a_n)
    print(strat_a)

    # Initialize trader B
    b_s, b_e = 0.1, 0.3
    # a_n = np.random.rand(N) * 0.2
    b_n = np.array([0.2, 0.2, 0])
    strat_b = pb.SinesBlocks(N, (b_s, b_e), b_n, lambd=10)
    print("\n"+str(strat_b))

    c = pb.CostModel([strat_a, strat_b], rho=0.1)
    print("\n"+str(c))

    t_values = np.linspace(0, 1, 100)
    a, b = c.strats[c.A], c.strats[c.B]
    a_values = [a.calc(t) for t in t_values]
    b_values = [b.calc(t) for t in t_values]
    m_values = [c.mkt.calc(t) for t in t_values]

    import matplotlib.pyplot as plt
    plt.plot(t_values, a_values)
    plt.plot(t_values, b_values)
    plt.plot(t_values, m_values)
    plt.title("Trader A and B and Marke Unit strategies")
    plt.legend([f"A({a.lambd})",  f"B({b.lambd})", f"Mkt({c.mkt.lambd})"])
    plt.show()
