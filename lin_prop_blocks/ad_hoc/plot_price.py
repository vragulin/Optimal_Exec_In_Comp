"""  A simple test script to plot a SinesBlocks strategy
    V. Ragulin, 10/19/2024
"""
import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))

import prop_blocks as pb
import propagator as pr

if __name__ == '__main__':
    N = 2

    # Initialize trader A
    # a_s, a_e = 0.6, 0.2
    a_s, a_e = 0, 0
    # a_n = np.random.rand(N) * 0.2
    # a_n = np.array([-0.3, 0.3, 0.1])
    a_n = np.zeros(N)
    strat_a = pb.SinesBlocks(N, (a_s, a_e), a_n)

    # Initialize trader B
    # b_s, b_e = 0.1, 0.3
    b_s, b_e = 0, 0
    # a_n = np.random.rand(N) * 0.2
    # b_n = np.array([0.2, 0.2, 0])
    b_n = np.zeros(N)
    strat_b = pb.SinesBlocks(N, (b_s, b_e), b_n, lambd=0)

    c = pb.CostFunction([strat_a, strat_b], rho=10)
    print("\n"+str(c))

    t_values = np.linspace(0, 1, 100)
    a, b = c.strats[c.A], c.strats[c.B]
    a_values = [a.calc(t) for t in t_values]
    b_values = [b.calc(t) for t in t_values]
    m_values = [c.mkt.calc(t) for t in t_values]

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2)
    plt.suptitle(f"OW Cost Model, " +
                 r"$\rho=$" + f"{c.rho:.4g}")

    ax = axs[0]
    ax.plot(t_values, a_values)
    ax.plot(t_values, b_values)
    ax.plot(t_values, m_values)
    ax.set_title("Trader A and B and Marke Unit strategies")
    ax.legend([f"A({a.lambd})",  f"B({b.lambd})", f"Mkt({c.mkt.lambd})"])
    ax.grid()

    ax = axs[1]
    price = [c.price(t) for t in t_values]
    ax.plot(t_values, price)
    ax.set_title("Price Impact")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid()

    plt.tight_layout()
    plt.show()

    # Test vs. the propagator price impact if all blocks are zero
    print("p(0.4) = ", c.price(0.4))
    if c.mkt.blocks == (0, 0):
        exp_p = pr.prop_price_impact_approx(0.4, c.strats[c.A].coeff,  c.strats[c.B].coeff,
                                            c.strats[c.B].lambd, c.rho)
        print("expected p(0.4) = ", exp_p)
