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
    blocks = [0.2, 0.3]
    coeff = np.array([-0.2, 0.3, -.3])
    sb = pb.SinesBlocks(N, blocks, coeff)
    print(sb)

    t_values = np.linspace(0, 1, 100)
    a_values = [sb.calc(t) for t in t_values]
    trend = [pb.SinesBlocks(N, blocks, None).calc(t) for t in t_values]

    import matplotlib.pyplot as plt
    plt.plot(t_values, a_values)
    plt.plot(t_values, trend)
    plt.title("Trader A strategy")
    plt.legend(["A", "Trend"])
    plt.show()