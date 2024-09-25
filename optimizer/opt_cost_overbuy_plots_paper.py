"""
Grid plots for the paper for the Overbuying Constraint
Miminize cost function given b
Subject to a contraint L(t) <= a(t) <= U(t)
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Any
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'cost_function')))
import fourier as fr
from cost_function_approx import cost_fn_a_approx
from sampling import sample_sine_wave
import trading_funcs as tf

# Global Parameters
N = 10  # number of Fourier terms
kappa_list = [0.1, 0.5, 10, 50]  # permanent impact
lambd_list = [1, 5, 10, 20]  # temporary impact
rho_list = [None, 4, 1, 0]  # Overbuying cap
xi_a = 0  # risk aversion of a
sigma = 3  # volatility of the stock

T_SAMPLE_PER_SEMI_WAVE = 3  # number of points to sample constraints
N_PLOT_POINTS = 100  # number of points for plotting
RED_COLORS = ["darkred", "firebrick", "crimson", "indianred", "tomato"]
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
INCLUDE_SUPTITLE = False

# Parameters for the b(t) function
RISK_NEUTRAL, RISK_AVERSE, EAGER, PARABOLIC = range(4)
B_INV_TYPE = RISK_AVERSE
INV_LABELS = ["Risk-Neutral", "Risk-Averse", "Parabolic"]
B_PARABOLIC_CONST = 1.2
TRADING_FUNCS = {
    RISK_NEUTRAL: [tf.b_func_risk_neutral, tf.b_dot_func_risk_neutral],
    RISK_AVERSE: [tf.b_func_risk_averse, tf.b_dot_func_risk_averse],
    EAGER: [tf.b_func_eager, tf.b_dot_func_eager],
    PARABOLIC: [tf.b_func_parabolic, tf.b_dot_func_parabolic]
    }

# Global variables

# Constraint type
OVERBUYING, NO_SHORT = range(2)
CONSTRAINT_TYPE = OVERBUYING

b_coeffs = None  # It will be estimated when needed
t_sample = None  # Points at whcih we sample the inequalities

b_func, b_dot_func = TRADING_FUNCS[B_INV_TYPE]


def unpack_params(p: dict) -> tuple:
    return p['lambd'], p['kappa'], p['sigma'], p['gamma']


def plot_one_cell(opt_coeffs_frame: np.ndarray, ax: Any, params: dict) -> None:
    """ Plot curves and and calc stats """
    _, _, sig, gamma = unpack_params(params)
    t_values = np.linspace(0, 1, N_PLOT_POINTS)
    b_curve = [b_func(t, params) for t in t_values]

    for u, rho in enumerate(rho_list):
        opt_coeffs = opt_coeffs_frame[u]
        opt_curve = [fr.reconstruct_from_sin(t, opt_coeffs) + gamma * t
                     for t in t_values]

        # plt.plot(t_values, init_curve, label='Initial guess', color='blue')
        label = r"$a(t)$" + (r', $\rho$=' + f'{rho}' if rho is not None else "")
        ax.plot(t_values, opt_curve, label=label, color=RED_COLORS[u],
                linestyle=LINE_STYLES[u])

    if B_INV_TYPE != RISK_NEUTRAL:
        ax.plot(t_values, t_values, color="black", linestyle="dashed")
    ax.plot(t_values, b_curve, label=r"$b_{\lambda}(t)$", color="blue")

    ax.set_title(f'λ={lambd}, κ={kappa}')
    ax.legend()  # prop={'size': 6})
    ax.set_xlabel('t')
    ax.set_ylabel(r'$a(t), b_\lambda(t)$')
    ax.grid()


def make_plot_suptitle():
    constr_code = {OVERBUYING: "an Overbuying", NO_SHORT: "a No Shorting"}
    l = (f'Best Response, with {constr_code[CONSTRAINT_TYPE]} Constraint, '
         f'to a Passive {INV_LABELS[B_INV_TYPE]} Adversary\n'
         + r'who is trading the $\lambda$-scaled strategy $b_{\lambda}(t)$ with permanent impact $\kappa$')
    if B_INV_TYPE == RISK_AVERSE and sigma != 0:
        l += r', risk-aversion $\sigma$=' + f"{sigma}"
    return l


if __name__ == "__main__":
    # Minimize the cost function

    fig, axs = plt.subplots(len(kappa_list), len(lambd_list), figsize=(16, 16))
    if INCLUDE_SUPTITLE:
        plt.suptitle(make_plot_suptitle(), fontsize=16)

    # Set up an array to keep answers for different value of rho
    a_coeffs_frame = np.zeros((len(rho_list), N))

    for i, kappa in enumerate(kappa_list):  # ([kappa_list[0]]):
        for j, lambd in enumerate(lambd_list):  # ([lambd_list[0]]):

            params = {'lambd': lambd, 'kappa': kappa, 'sigma': sigma, 'gamma': 1}


            def cost_function(a_coeffs):
                global b_coeffs
                _, _, sig, gamma = unpack_params(params)
                if b_coeffs is None:
                    b_coeffs = fr.sin_coeff(lambda t: b_func(t, params) - gamma * t, N)

                return cost_fn_a_approx(a_coeffs, b_coeffs, kappa, lambd)


            for u, rho in enumerate(rho_list):

                def U_func(t):
                    return (1 + rho) if rho is not None else np.finfo(np.float64).max


                def constraint_function(a_coeffs):
                    # Sample points
                    global t_sample
                    if t_sample is None:
                        t_sample = sample_sine_wave(list(range(1, len(a_coeffs) + 1)), T_SAMPLE_PER_SEMI_WAVE)

                    def a_func(t: float) -> float:
                        return fr.reconstruct_from_sin(t, a_coeffs) + params['gamma'] * t

                    a_values = np.array([a_func(t) for t in t_sample])
                    U_values = np.array([U_func(t) for t in t_sample])

                    # return np.concatenate([U_values - a_values, a_values - L_values])
                    return U_values - a_values


                # Define the constraint
                constraints = ({
                    'type': 'ineq',
                    'fun': lambda a_coeffs: constraint_function(a_coeffs)
                }) if rho is not None else None

                # Initial guess for a_coeffs
                initial_guess = np.zeros(N)
                initial_cost = cost_function(initial_guess)
                # print(f"Initial a_coeffs = {np.round(initial_guess, 3)}")
                print(f"Initial guess cost = {initial_cost:.4f}\n")

                start = time.time()
                result = minimize(cost_function, initial_guess, method='SLSQP', constraints=constraints)
                print(f"optimization time = {(time.time() - start):.4f}s")

                # Optimized coefficients
                optimized_a_coeffs = result.x

                # Compute the cost with optimized coefficients
                optimized_cost = cost_function(optimized_a_coeffs)

                print(f"Optimized a_coeffs = {np.round(optimized_a_coeffs, 3)}")
                print(f"Optimized cost = {optimized_cost:.4f}")

                a_coeffs_frame[u] = optimized_a_coeffs

            # Plot curves
            plot_one_cell(a_coeffs_frame, axs[i, j], params)

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()
