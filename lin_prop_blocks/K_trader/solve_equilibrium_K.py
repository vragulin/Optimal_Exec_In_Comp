"""  Solve for the equilirbium within the OW-with-Blocks model
    V. Ragulin - 10/22/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy

from cvxopt.misc import use_C

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '../..', 'cost_function'),
    os.path.join(CURRENT_DIR, '../..', 'optimizer_qp')
])

from prop_blocks import SinesBlocks
from cost_model_K import CostModelK, Group
import trading_funcs as tf

# Parameters
N = 50  # number of Fourier terms
RHO = 0.1  # propagator decay
NGROUPS = 2  # number of groups
LAMBD = [90, 1]  # sizes
NTRADERS = [1, 10]  # number of traders in each group

# Regularization parameters
REG_PARAMS = {'wiggle': 0, 'wiggle_exp': 4}  # Regularization parameters - dict or None
STRAT_TYPE = {'blocks': True, 'sines': True}  # Types of strategies to use

# Exponent for the wiggle penalty
RUN_TESTS = True  # Check that the solution satisfies the Nash Equilibrium conditions
abs_tol = 1e-6

# Presentation parameters
N_PLOT_POINTS = 100  # number of points to plot, works best with N//2
N_COEFF_TO_PRINT = 4  # Number of coefficients to print


# Constants
def trader_code(trader: int) -> str:
    return f"{trader}"


def trader_color(trader: int) -> str:
    colors = {0: 'red',
              1: 'blue',
              2: 'green',
              3: 'purple',
              4: 'orange'}
    return colors[trader % len(colors)]


def trader_func_name(trader: int) -> str:
    return fr'$a_{{{trader}}}(t)$'


def lambd_list():
    return ', '.join([str(l) for l in LAMBD])


def ntraders_list():
    return ', '.join([str(n) for n in NTRADERS])


def plot_spectral_density(a_coeff, b_coeff, lambd, rho, N):
    """ Plot the spectral density of the Fourier coefficients """
    n = np.arange(1, N + 1)
    a_spec = np.abs(a_coeff)
    b_spec = np.abs(b_coeff)

    plt.figure(figsize=(10, 5))
    plt.title(f'Spectral Density of Fourier Coefficients, λ={lambd}, ρ={rho}, N={N}', fontsize=12)

    plt.plot(n, a_spec, label=r"$|a_n|$", color='red', alpha=0.5)
    plt.plot(n, b_spec, label=r"$|b_n|$", color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('Spectral Density')
    # plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def plot_curves(c: CostModelK, **kwargs) -> None:
    """ Plot the trading strategies and the temporary price displacement """
    n_points = kwargs.get('n_points', 100)
    init_guess = kwargs.get('init_guess', SinesBlocks(N=c.N))

    # Plot initial guess and optimized functions
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plt.suptitle(f'Nash Equilibrium in the Linear Propagator Model\n'
                 f'N={c.N}, λ=[{lambd_list()}], ntraders=[{ntraders_list()}], ρ={c.rho}',
                 fontsize=14)

    # Top Chart plot curves
    ax = axs[0]
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [init_guess.calc(t) for t in t_values]
    ax.plot(t_values, init_curve, label='Initial guess ', color='grey', linestyle="dotted")
    for i, g in enumerate(c.groups):
        curve = [g.strat.calc(t) for t in t_values]
        ax.plot(t_values, curve, label=trader_func_name(i), color=trader_color(i))

    ax.set_title(f'Trading Strategies for Trader Groups', fontsize=11)
    ax.set_xlabel('t')
    ax.set_ylabel(r"$a_i(t)$")
    ax.legend()
    ax.grid()

    # Bottom chart
    ax = axs[1]
    dp = [c.price(t) for t in t_values]
    dp[0] = 0
    dp[-1] = dp[-1] + c.mkt.blocks[-1] * c.mkt.lambd

    ax.set_title(f'Temporary Price Displacement', fontsize=11)
    ax.plot(t_values, dp, label="dP(0,t)", color='green')
    ax.set_xlabel('t')
    ax.set_ylabel("dP(0,t)")
    ax.legend()
    ax.grid()

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def main():
    # Create the model.  The starting strategy are not important, just the structure
    strats = [SinesBlocks(N, lambd=LAMBD[i]) for i in range(NGROUPS)]
    groups = [Group(trader_code(i), strats[i], NTRADERS[i]) for i in range(NGROUPS)]
    c = CostModelK(groups, RHO)

    # Solve for the equilibrium
    start_time = time.time()
    _ = c.solve_equilibrium(strat_type=STRAT_TYPE)
    print(f"Time to solve: {time.time() - start_time:.2f} seconds")

    # Compute the cost with optimized coefficients
    opt_cost = {k: c.cost_trader(k) for k in range(NGROUPS)}

    # Print the results
    print("\nEquilibrium Strategies:")
    print(c)
    print("\nOptimized costs:")
    print({k: round(float(v), 4) for k, v in opt_cost.items()})
    # print(f"Optimized wiggle penalties = , {c.wiggle_penalty(A):.4f}, {c.wiggle_penalty(B):.4f}\n")

    # Plot the curves
    plot_curves(c, n_points=N_PLOT_POINTS)

    # Run tests that the solution is, in fact, a Nash Equilibrium
    if RUN_TESTS:
        print("\nChecking that the strategy of each trader is the best response to the others")
        init_cost, coeffs_opt, coeff_diff = {}, {}, {}
        for i, g in enumerate(c.groups):

            def extract_coeffs(model: CostModelK, group: int, strat_type: dict | None = None) -> np.array:
                if strat_type is None:
                    use_blocks, use_sines = True, True
                else:
                    use_blocks, use_sines = strat_type['blocks'], strat_type['sines']
                encode_length = 2 * use_blocks + model.N * use_sines
                x = np.zeros(encode_length)
                if use_blocks:
                    x[:2] = model.groups[group].strat.blocks
                if use_sines:
                    x[2:] = model.groups[group].strat.coeff
                return x

            init_guess = extract_coeffs(c, i, STRAT_TYPE)
            init_cost[i] = c.cost_trader(group=i)
            print(f"Initial {i} coeffs = {np.round(init_guess[:N_COEFF_TO_PRINT + 2], 3)}")
            print(f"Initial {i} guess cost = {init_cost[i]:.4f}")

            # Minimize the cost function
            opt_strat, res = c.solve_min_cost(group=i, strat_type=STRAT_TYPE, abs_tol=abs_tol)

            # Save the optimal solution and implementation cost
            opt_cost[i] = res['scipy_output'].fun
            coeffs_opt[i] = extract_coeffs(res['var_model'], res['var_trader_idx'], STRAT_TYPE)
            print(f"Optimized {i} coeffs = {np.round(coeffs_opt[i][:N_COEFF_TO_PRINT + 2], 3)}")
            print(f"Optimized cost = {opt_cost[i]:.4f}")

            coeff_diff[i] = np.linalg.norm(coeffs_opt[i] - init_guess) / np.sqrt(N)
            print(f"Norm Diff {i}: ", coeff_diff[i])

            if init_cost[i] - opt_cost[i] > abs_tol or coeff_diff[i] > abs_tol:
                print(f"\nTest failed for group {i}!  Analytical solution is not optimal")
                break

        print("\nSuccess! Analytical solution is optimal for all trader groups")


if __name__ == "__main__":
    main()
