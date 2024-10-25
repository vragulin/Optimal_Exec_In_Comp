"""  Solve for the equilirbium within the OW-with-Blocks model
    V. Ragulin - 10/22/2024
"""
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(CURRENT_DIR, '..', 'cost_function'),
    os.path.join(CURRENT_DIR, '..', 'optimizer_qp')
])

from prop_blocks import SinesBlocks, CostModel
from cost_model_qp import CostModelQP
import trading_funcs as tf

# Parameters
N = 20  # number of Fourier terms
RHO = 1  # propagator decay
LAMBD = 20  # size of trader B

# Regularization parameters
REG_PARAMS = {'wiggle': 0, 'wiggle_exp': 4}  # Regularization parameters - dict or None
ONLY_BLOCKS = False  # If True, only optimize the block coefficients

# Exponent for the wiggle penalty
RUN_TESTS = True  # Check that the solution satisfies the Nash Equilibrium conditions
abs_tol = 1e-6

#
# Presentation parameters
N_PLOT_POINTS = 100  # number of points to plot, works best with N//2
N_COEFF_TO_PRINT = 4  # Number of coefficients to print

# Constants
A, B = range(2)


def other(trader: int) -> int:
    return 1 - trader


def trader_code(trader: int) -> str:
    return 'A' if trader == A else 'B'


def trader_color(trader: int) -> str:
    return 'red' if trader == A else 'blue'


def trader_func_name(trader: int) -> str:
    return 'a(t)' if trader == A else r'$b_\lambda(t)$'


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


def plot_curves(c: CostModelQP, **kwargs) -> None:
    n_points = kwargs.get('n_points', 100)
    init_guess = kwargs.get('init_guess', SinesBlocks(N=c.N))

    """ Plot curves and and calc stats """
    t_values = np.linspace(0, 1, n_points + 1)
    init_curve = [init_guess.calc(t) for t in t_values]
    a_curve = [c.strats[A].calc(t) for t in t_values]
    b_curve = [c.strats[B].calc(t) for t in t_values]

    # Plot initial guess and optimized functions
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plt.suptitle(f'Nash Equilibrium in the Linear Propagator Model\n'
                 f'N={c.N}, λ=[{c.strats[A].lambd, c.strats[B].lambd}], ρ={c.rho}',
                 fontsize=14)

    # Top chart
    ax = axs[0]
    ax.plot(t_values, init_curve, label='Initial guess ', color='grey', linestyle="dotted")
    ax.plot(t_values, a_curve, label=trader_func_name(A), color=trader_color(A), linewidth=2)
    ax.plot(t_values, b_curve, label=trader_func_name(B), color=trader_color(B), linestyle="dashed")
    ax.set_title(f'Trading Strategies for A and B', fontsize=11)
    ax.set_xlabel('t')
    ax.set_ylabel(f'{trader_func_name(A)}, {trader_func_name(B)}')
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
    # Solve for the equilibrium
    if ONLY_BLOCKS:
        strats, res = CostModelQP.solve_equilibrium_blocks(N=N, lambd=LAMBD, rho=RHO, reg_params=REG_PARAMS)
    else:
        strats, res = CostModelQP.solve_equilibrium(N=N, lambd=LAMBD, rho=RHO, reg_params=REG_PARAMS)

    # Compute the cost with optimized coefficients
    c = CostModelQP(strats, rho=RHO, reg_params=REG_PARAMS)
    opt_cost = {A: c.cost_trader_matrix(), B: c.cost_trader_matrix(B)}

    # Print the results
    print("Equilibrium Strategies:")
    print(c)
    print(f"Optimized costs = , {opt_cost[A]:.4f}, {opt_cost[B]:.4f}")
    print(f"Optimized wiggle penalties = , {c.wiggle_penalty(A):.4f}, {c.wiggle_penalty(B):.4f}\n")

    # Find the exact solution and plot curves
    plot_curves(c, n_points=N_PLOT_POINTS)

    # Run tests
    if RUN_TESTS:
        # Plot spectral densities of a and b
        plot_spectral_density(c.strats[A].coeff, c.strats[B].coeff, LAMBD, RHO, N)

        print("\nChecking that the strategy of each trader is the best response to the other")
        init_cost, coeffs_opt, coeff_diff = {}, {}, {}
        for trader in range(2):
            tcode = trader_code(trader)

            def extract_coeffs(model: CostModel, trader: int) -> np.array:
                return np.concatenate([model.strats[trader].blocks, model.strats[trader].coeff])

            init_guess = extract_coeffs(c, trader)
            init_cost[trader] = c.cost_trader_matrix(trader)
            print(f"Initial {tcode} coeffs = {np.round(init_guess[:N_COEFF_TO_PRINT + 2], 3)}")
            print(f"Initial {tcode} guess cost = {init_cost[trader]:.4f}")
            print(f"Initial wiggle penalty = {c.wiggle_penalty(trader):.4f}\n")

            # Minimize the cost function
            opt_strat, res = c.solve_min_cost_qp(trader)

            new_strats = deepcopy(c.strats)
            new_strats[trader] = opt_strat
            new_c = CostModelQP(new_strats, rho=RHO, reg_params=REG_PARAMS)

            # Compute the cost with optimized coefficients
            opt_cost[trader] = new_c.cost_trader_matrix(trader)
            coeffs_opt[trader] = extract_coeffs(new_c, trader)
            print(f"Optimized {tcode} coeffs = {np.round(coeffs_opt[trader][:N_COEFF_TO_PRINT + 2], 3)}")
            print(f"Optimized cost = {opt_cost[trader]:.4f}")
            print(f"Optimized wiggle penalty = {new_c.wiggle_penalty(trader):.4f}\n")

            coeff_diff[trader] = np.linalg.norm(coeffs_opt[trader] - init_guess) / np.sqrt(N)
            print(f"Norm Diff {trader}: ", coeff_diff[trader])

        if max(coeff_diff.values()) > abs_tol \
                or abs(opt_cost[A] - init_cost[A]) > abs_tol \
                or abs(opt_cost[B] - init_cost[B]) > abs_tol:
            print("\nTest failed!  Analytical solution is not optimal")
        else:
            print("\nSuccess! Analytical solution is optimal for both traders")


if __name__ == "__main__":
    main()
