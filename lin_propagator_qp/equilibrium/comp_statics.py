""" Generate comparative statics plots for the equilibrium model
    V. Ragulin - 10/10/2024
"""

import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'matrix_utils')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..')))
from propagator import cost_fn_prop_a_approx, cost_fn_prop_b_approx
import prop_cost_to_QP as pcq
from analytic_solution import solve_equilibrium
import fourier as fr

# -------------------------------------------------------
# Script to run and test the analytic_solution.py functions
# -------------------------------------------------------
# Global parameters
N = 200  # Dimension of the vector x
LAMBD_LIST = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 100, 200]  # For the x-axis
RHO_LIST = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 100]  # For the x-axis
LAMBD_LIST_SHORT = [0.1, 1, 20, 100]  # For subplots, should be a subset of LAMBD_LIST
RHO_LIST_SHORT = [0.1, 1, 10, 100]  # For subplots, should be a subset of RHO_LIST

# Time points to plot
T_LIST = [0.25, 0.5, 0.75]
P25, P50, P75 = range(3)

# Presentation settings
N_COEFF_TO_PRINT = 4
N_SAMPLE_POINTS = 100  # Works beest when N is a multiple of 2 * N_PLOT_POINTS
DROP_LAST_SINES = None  # None or integer.  Use to reduce the 'wiggle'
STRAT_LABELS = ['a(t)', r'$b_\lambda(t)$']
INCLUDE_SUPTITLE = True

# Location of files to store data
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../', 'results/plots_OW/equilibrium'))
DATA_FILE = 'equil_data_r2.pkl'
RECALC_DATA = True  # If True, recalculate the data, otherwse load from file and plot


class Sim:

    def __init__(self, N, lambd_list, rho_list, **kwargs):
        self.N = N
        self.lambd_list = lambd_list
        self.rho_list = rho_list
        self.data = {}

        self.data_dir = kwargs.get('data_dir', DATA_DIR)
        self.data_file = kwargs.get('data_file', 'equilibrium_data.pkl')
        self.data_full_path = os.path.join(self.data_dir, self.data_file)
        self.n_sample = kwargs.get('n_sample', N_SAMPLE_POINTS)
        self.t_values = np.linspace(0, 1, self.n_sample + 1)
        self.drop_last_sines = kwargs.get('drop_last_sines', None)

    def get_data(self, recalc=True):
        if recalc or not os.path.exists(self.data_full_path):
            return self.gen_sim_data()
        else:
            return self.load_data()

    def gen_sim_data(self):
        for lambd in self.lambd_list:
            for rho in self.rho_list:
                stats = self.run_one_param_set(lambd, rho)
                self.data[(lambd, rho)] = stats

        self.save_data()

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with open(self.data_full_path, 'wb') as f:
            pickle.dump(self.data, f)

    def load_data(self):
        with open(self.data_full_path, 'rb') as f:
            self.data = pickle.load(f)
        return self.data

    def run_one_param_set(self, lambd, rho):
        eqil = solve_equilibrium(N, lambd, rho)
        a_coeffs, b_coeffs = eqil['a'], eqil['b']

        if self.drop_last_sines:
            _a_coeffs = a_coeffs[:-self.drop_last_sines]
            _b_coeffs = b_coeffs[:-self.drop_last_sines]
        else:
            _a_coeffs, _b_coeffs = a_coeffs, b_coeffs

        a_curve = [fr.reconstruct_from_sin(t, _a_coeffs) + t for t in self.t_values]
        b_curve = [fr.reconstruct_from_sin(t, _b_coeffs) + t for t in self.t_values]

        a_list = [fr.reconstruct_from_sin(t, a_coeffs) + t for t in T_LIST]
        b_list = [fr.reconstruct_from_sin(t, b_coeffs) + t for t in T_LIST]

        a_cost = cost_fn_prop_a_approx(_a_coeffs, _b_coeffs, lambd, rho)
        b_cost = cost_fn_prop_b_approx(_a_coeffs, _b_coeffs, lambd, rho) / lambd

        a_dict = {'coeffs': a_coeffs, 'max': max(a_curve), 'min': min(a_curve),
                  'cost': a_cost, 't_list_vals': a_list}
        b_dict = {'coeffs': b_coeffs, 'max': max(b_curve), 'min': min(b_curve),
                  'cost': b_cost, 't_list_vals': b_list}

        return {'a': a_dict, 'b': b_dict}

    def plot_grid_rho_list(self, rho_list: list | None = None):
        if rho_list is None:
            rho_list = self.rho_list

        fig, axs = plt.subplots(3, len(rho_list), figsize=(10, 10))
        if INCLUDE_SUPTITLE:
            fig.suptitle('Two Trader Equilibrium Strategies vs. Model Parameters\n'
                         'for different values of propagator decay, ρ', fontsize=16)

        axs = np.atleast_2d(axs)
        for col, rho in enumerate(rho_list):
            # Fetch data for the plot
            for i, (trader, color) in enumerate(zip(['a', 'b'], ['red', 'blue'])):
                vals_max = [self.data[(lambd, rho)][trader]['max'] for lambd in self.lambd_list]
                vals_min = [self.data[(lambd, rho)][trader]['min'] for lambd in self.lambd_list]
                vals_25p = [self.data[(lambd, rho)][trader]['t_list_vals'][P25] for lambd in self.lambd_list]
                vals_50p = [self.data[(lambd, rho)][trader]['t_list_vals'][P50] for lambd in self.lambd_list]
                vals_75p = [self.data[(lambd, rho)][trader]['t_list_vals'][P75] for lambd in self.lambd_list]
                cost_vals = [self.data[(lambd, rho)][trader]['cost'] for lambd in self.lambd_list]

                axs[i, col].plot(self.lambd_list, vals_max, label=f'max', color=color, linestyle=':')
                axs[i, col].plot(self.lambd_list, vals_min, label=f'min', color=color, linestyle=':')
                axs[i, col].plot(self.lambd_list, vals_25p, label=f't=0.25', color=color,
                                 linestyle='-')
                axs[i, col].plot(self.lambd_list, vals_50p, label=f't=0.5', color=color,
                                 linestyle='--')
                axs[i, col].plot(self.lambd_list, vals_75p, label=f't=0.75', color=color,
                                 linestyle='-', marker='.')
                if trader == 'a':
                    axs[2, col].scatter(self.lambd_list, cost_vals, s=10, label=f'cost({trader})', color=color)
                else:
                    axs[2, col].plot(self.lambd_list, cost_vals, label=f'cost({trader})', color=color)

                axs[i, col].set_xlabel('λ')
                axs[i, col].set_ylabel(STRAT_LABELS[i])
                axs[i, col].legend(loc='center', framealpha=0.5)
                axs[i, col].set_title(f'{trader}(t) vs. λ\nρ={rho}', fontsize=10)
                axs[i, col].set_xscale('log')

            axs[2, col].set_xlabel('λ')
            axs[2, col].set_ylabel('cost')
            axs[2, col].set_title(f'unit cost vs. λ\nρ={rho}', fontsize=10)
            axs[2, col].legend(framealpha=0.5)
            axs[2, col].set_xscale('log')

        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()

    def plot_grid_lambd_list(self, lambd_list: list | None = None):
        if lambd_list is None:
            lambd_list = self.lambd_list

        fig, axs = plt.subplots(3, len(lambd_list), figsize=(10, 10))
        if INCLUDE_SUPTITLE:
            fig.suptitle('Two Trader Equilibrium Strategies vs. Model Parameters\n'
                         'for different values of trader B size, λ'
                         '', fontsize=16)

        axs = np.atleast_2d(axs)
        for col, lambd in enumerate(lambd_list):
            # Fetch data for the plot
            for i, (trader, color) in enumerate(zip(['a', 'b'], ['red', 'blue'])):
                vals_max = [self.data[(lambd, rho)][trader]['max'] for rho in self.rho_list]
                vals_min = [self.data[(lambd, rho)][trader]['min'] for rho in self.rho_list]
                vals_25p = [self.data[(lambd, rho)][trader]['t_list_vals'][P25] for rho in self.rho_list]
                vals_50p = [self.data[(lambd, rho)][trader]['t_list_vals'][P50] for rho in self.rho_list]
                vals_75p = [self.data[(lambd, rho)][trader]['t_list_vals'][P75] for rho in self.rho_list]
                cost_vals = [self.data[(lambd, rho)][trader]['cost'] for rho in self.rho_list]

                axs[i, col].plot(self.rho_list, vals_max, label=f'max', color=color, linestyle=':')
                axs[i, col].plot(self.rho_list, vals_min, label=f'min', color=color, linestyle=':')
                axs[i, col].plot(self.rho_list, vals_25p, label=f't=0.25', color=color,
                                 linestyle='-')
                axs[i, col].plot(self.rho_list, vals_50p, label=f't=0.5', color=color,
                                 linestyle='--')
                axs[i, col].plot(self.rho_list, vals_75p, label=f't=0.75', color=color,
                                 linestyle='-', marker='.')
                if trader == 'a':
                    axs[2, col].scatter(self.rho_list, cost_vals, s=10, label=f'cost({trader})', color=color)
                else:
                    axs[2, col].plot(self.rho_list, cost_vals, label=f'cost({trader})', color=color)

                axs[i, col].set_xlabel('ρ')
                axs[i, col].set_ylabel(STRAT_LABELS[i])
                axs[i, col].legend(loc='center', framealpha=0.5)
                axs[i, col].set_title(f'{trader}(t) vs. ρ\nλ={lambd}', fontsize=10)
                axs[i, col].set_xscale('log')

            axs[2, col].set_xlabel('ρ')
            axs[2, col].set_ylabel('cost')
            axs[2, col].set_title(f'unit cost vs. ρ\nλ={lambd}', fontsize=10)
            axs[2, col].legend(framealpha=0.5)
            axs[2, col].set_xscale('log')

        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()


def main():
    sim = Sim(N, LAMBD_LIST, RHO_LIST)

    sim.get_data(recalc=RECALC_DATA)
    sim.plot_grid_rho_list(rho_list=RHO_LIST_SHORT)
    sim.plot_grid_lambd_list(lambd_list=LAMBD_LIST_SHORT)


if __name__ == "__main__":
    main()
