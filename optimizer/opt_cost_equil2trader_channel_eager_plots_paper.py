"""
Grid plots for the paper
Trader B is trading an equilibrium strategy
Trader A trades best reponse with a channel constraint
Miminize cost function given b
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
lambd_list = [1, 2, 5, 10]  # temporary impact
kappa_list = [0.5, 2, 10, 20]  # Time by which we have to be "in the box"
sigma_list = [None, 1, 5, 10]  # percent that needs to be completed
xi_a = 0  # risk aversion of
gamma = 1  # Target position at t=1

T_SAMPLE_PER_SEMI_WAVE = 3  # number of points to sample constraints
N_PLOT_POINTS = 100  # number of points for plotting
RED_COLORS = ["darkred", "firebrick", "crimson", "indianred", "tomato"]
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
INCLUDE_SUPTITLE = False

# Parameters for the b(t) function
RISK_NEUTRAL, RISK_AVERSE, EAGER, EQ_2TRADER, PARABOLIC = range(5)
B_INV_TYPE = EQ_2TRADER
INV_LABELS = ["Risk-Neutral", "Risk-Averse", "Eager",
              "2-Trader Equilibrium", "Parabolic"]
B_PARABOLIC_CONST = 1.2
TRADER_A = True
TRADING_FUNCS = {
    RISK_NEUTRAL: [tf.b_func_risk_neutral, tf.b_dot_func_risk_neutral],
    RISK_AVERSE: [tf.b_func_risk_averse, tf.b_dot_func_risk_averse],
    EAGER: [tf.b_func_eager, tf.b_dot_func_eager],
    EQ_2TRADER:  [tf.equil_2trader, tf.equil_2trader_dot],
    PARABOLIC: [tf.b_func_parabolic, tf.b_dot_func_parabolic]
}

# Global variables

# Constraint type
OVERBUYING, NO_SHORT, COMPLETION, CHANNEL = range(4)
CONSTRAINT_TYPE = CHANNEL

b_coeffs = None  # It will be estimated when needed
t_sample = None  # Points at whcih we sample the inequalities

b_func, b_dot_func = TRADING_FUNCS[B_INV_TYPE]


def unpack(p: dict) -> tuple:
    return p['lambd'], p['kappa']


def plot_one_cell(opt_coeffs_frame: np.ndarray, ax: Any, params: dict) -> None:
    """ Plot curves for a chart in a single grid cell
    :param opt_coeffs_frame: 2d array with each row containing Fourier coeff for one scenario
    :param ax: a pointer to an plt Ax object representing the cell
    :param params: dictionary with function inputs
    """
    lambd, kappa = unpack(params)
    t_values = np.linspace(0, 1, N_PLOT_POINTS)
    b_params = params.copy()
    b_params['trader_a'] = False
    b_curve = [b_func(t, b_params) for t in t_values]

    for u, sigma in enumerate(sigma_list):
        opt_coeffs = opt_coeffs_frame[u]
        opt_curve = [fr.reconstruct_from_sin(t, opt_coeffs) + gamma * t
                     for t in t_values]

        label = r"$a(t)$" + (r', $\sigma$=' + f'{sigma}' if sigma is not None else "")
        ax.plot(t_values, opt_curve, label=label, color=RED_COLORS[u],
                linestyle=LINE_STYLES[u])

        if sigma is not None:
            lbound = [eager_bmk(t, sigma) for t in t_values]
            ax.fill_between(t_values, lbound, t_values,
                            color='grey', alpha=0.1 + 0.1 * (u + 1))

    if B_INV_TYPE != RISK_NEUTRAL:
        ax.plot(t_values, t_values, color="black", linestyle="dashed")
    ax.plot(t_values, b_curve, label=r"$b_\lambda(t)$", color="blue")

    ax.set_title(f'λ={lambd}, ' + r'$\kappa$' + f'={kappa}')
    ax.legend()  # prop={'size': 6})
    ax.set_xlabel('t')
    ax.set_ylabel(r'$a(t), b_\lambda(t)$')
    ax.grid()


def eager_bmk(t: float, sigma: float) -> float:
    return tf.b_func_eager(t, {'sigma': sigma, 'gamma': gamma})


def make_plot_suptitle():
    constr_code = {OVERBUYING: "an Overbuying",
                   NO_SHORT: "a No Shorting",
                   COMPLETION: "a Completion",
                   CHANNEL: "a Channel"}
    s = (f'Best Response, with {constr_code[CONSTRAINT_TYPE]} Constraint, '
         f'to a Passive {INV_LABELS[B_INV_TYPE]} Adversary\n'
         + r'who is trading the $\lambda$-scaled strategy $b(t; \lambda)$ '
           r'with permanent impact $\kappa$')
    s += ",\n" + (r"the Constraint requires that the position $a(t)$ is "
                  r"between t and the eager benchmark $c(t;\sigma)$")
    return s


if __name__ == "__main__":
    # Minimize the cost function

    fig, axs = plt.subplots(len(kappa_list), len(lambd_list), figsize=(16, 16))
    if INCLUDE_SUPTITLE:
        plt.suptitle(make_plot_suptitle(), fontsize=16)

    # Set up an array to keep answers and constraints for different value of sigma
    a_coeffs_frame = np.zeros((len(sigma_list), N))

    for i, kappa in enumerate(kappa_list):  # ([kappa_list[0]]):
        for j, lambd in enumerate(lambd_list):  # ([lambd_list[0]]):

            params = {'lambd': lambd, 'kappa': kappa, 'gamma': gamma, 'trader_a': True}

            def cost_function(a_coeffs):
                global b_coeffs
                if b_coeffs is None:
                    b_coeffs = fr.sin_coeff(lambda t: b_func(t, params) - gamma * t, N)

                return cost_fn_a_approx(a_coeffs, b_coeffs, kappa, lambd)


            for u, sigma in enumerate(sigma_list):

                def U_func(t):
                    return eager_bmk(t, sigma)

                def L_func(t):
                    return t

                def constraint_function(a_coeffs):
                    # Sample points
                    global t_sample
                    if t_sample is None:
                        t_sample = sample_sine_wave(list(range(1, len(a_coeffs) + 1)), T_SAMPLE_PER_SEMI_WAVE)

                    def a_func(t: float) -> float:
                        return fr.reconstruct_from_sin(t, a_coeffs) + params['gamma'] * t

                    a_values = np.array([a_func(t) for t in t_sample])
                    U_values = np.array([U_func(t) for t in t_sample])
                    L_values = np.array([L_func(t) for t in t_sample])
                    return np.concatenate([U_values - a_values, a_values - L_values])


                # Define the constraint
                constraints = ({
                    'type': 'ineq',
                    'fun': lambda a_coeffs: constraint_function(a_coeffs)
                }) if sigma is not None else None

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
