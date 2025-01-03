"""  Numerically solving for the two-trader equilibrium
     Implements 2 contstraints:  overbuying and short selling for both traders.
     Incorporate a regularization term to prenalize high second derivatge in the middle of the range.
     Implements QP solver for optimization.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from typing import Any, List, Dict
import os
import sys
import pickle
import config_prop_qp as cfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'optimizer_qp')))
import trading_funcs as tf
import cost_function_approx as ca
import fourier as fr
import propagator as pp
import qp_prop_solvers as qp
from sampling import sample_sine_wave

# Parameters and Constants
LAMBD = 100  # sixe of Trader B
RHO = 10  # propagator decay
N = 200  # number of Fourier Terms
GAMMA = 0.5  # Fraction of the way to move towards the new solution (float e.g. 1.0)
KAPPA = 10  # Parameter for the equilibrium strategy benchmark (permanent impact)
MAX_ITER = 20000  # Maximum number of iterations

TOL_COEFFS = 1e-6
TOL_COSTS = TOL_COEFFS
MAX_ABS_COST = 1e4
N_PLOT_POINTS = 100
N_ITER_LINES = 4

LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
INCLUDE_SUPTITLE = False
LABEL_OFFSET_MULT = 0.09
TO_HIGHLIGHT_FREQ = 50
N_COEFFS_TO_PRINT = 4

# Specify which solver to use for optimization
SCIPY, QP = range(2)
SOLVER = QP

# Regularization parameters
UNIFORM, SINE_WAVE = range(2)
# REG_PARAMS = {};
REG_PARAMS = {
    # '2nd_deriv': {
    #     'range': [0.05, 0.95],
    #     'factor': 0.00000,
    #     'factor_inc_rel': 0.000,
    #     'type': UNIFORM,
    #     'n_sample': N,
    #     'pts_per_semiwave': 3
    # },
    'iter_adjustment': {
        'max_other_cost_inc': 0.025,
        'max_iter_line_search': min(MAX_ITER // 10, 20)
    },
    'gamma_decay': 0.0007
    # Half-life = ln(2) / gamma_decay
}

# Initialize codes for the traders (like an enum)
TRADER_A, TRADER_B = range(2)

# Parameters to save simulation results
SAVE_RESULTS = True
DATA_FILE_SUFFIX = ""
ITER_SO_FAR = 0


class State:

    def __init__(self, **kwargs):
        # ToDo: write a docstring
        # Read model parameters
        self.rho = kwargs.get('rho', 0)
        self.lambd = kwargs.get('lambd', 1)
        self.gamma = kwargs.get('gamma', 0.8)
        self.kappa = kwargs.get('kappa', 1)

        assert self.rho > 0, "The propagator decay parameter rho must be positive"

        # Read initial strategy coefficients
        N = kwargs.get('N', 15)
        a_coeff = kwargs.get('a_coeff', None)
        b_coeff = kwargs.get('b_coeff', None)

        self.N = self._initialize_n(a_coeff, b_coeff, N)
        self.a_coeff = self._initialize_coeff(a_coeff, self.N)
        self.b_coeff = self._initialize_coeff(b_coeff, self.N)

        self._validate_coeff_lengths()

        # Update the costs for A and B if required
        if kwargs.get("calc_costs", True):
            self.a_cost = pp.cost_fn_prop_a_approx(self.a_coeff, self.b_coeff, self.lambd, self.rho)
            self.b_cost = pp.cost_fn_prop_b_approx(self.a_coeff, self.b_coeff, self.lambd, self.rho)
        else:
            self.a_cost = None
            self.b_cost = None

        self.reg_params = kwargs.get("reg_params", None)

        # Store precomputed variables to speed up the optimization
        self._precomp = {}
        self._t_sample = None
        self._sin_values = None
        self._sin_x_nsq = None
        self._reg_adj = 0
        self._n_iter = 0
        self._curr_gamma = self.gamma * np.exp(- REG_PARAMS['gamma_decay'] * ITER_SO_FAR)
        self._tol_costs = -1
        self._tol_coeffs = -1

    @staticmethod
    def _initialize_n(a_coeff: np.ndarray, b_coeff: np.ndarray, n: int) -> int:
        if a_coeff is None and b_coeff is None:
            return n
        return max(len(a_coeff) if a_coeff is not None else 0, len(b_coeff) if b_coeff is not None else 0)

    @staticmethod
    def _initialize_coeff(coeff: np.ndarray, n: int) -> np.ndarray:
        return coeff if coeff is not None else np.zeros(n)

    def _validate_coeff_lengths(self) -> None:
        if len(self.a_coeff) != len(self.b_coeff):
            raise ValueError(
                f"Fourier coeff array length mismatch: len(a)={len(self.a_coeff)}, len(b)={len(self.b_coeff)}")

    def copy(self, **kwargs):
        """
        Create a copy of the current state, with the option to overwrite select attributes.
        """
        # Create a new instance with the current attributes
        new_state = State(
            rho=kwargs.get('rho', self.rho),
            lambd=kwargs.get('lambd', self.lambd),
            gamma=kwargs.get('gamma', self.gamma),
            N=kwargs.get('N', self.N),
            a_coeff=kwargs.get('a_coeff', self.a_coeff),
            b_coeff=kwargs.get('b_coeff', self.b_coeff),
            calc_costs=kwargs.get('calc_costs', True),
            reg_params=kwargs.get('reg_params', self.reg_params)
        )

        # Overwrite any additional attributes provided in kwargs
        for key, value in kwargs.items():
            setattr(new_state, key, value)

        if ('copy_precomp' not in kwargs) or kwargs['copy_precomp']:
            new_state._precomp = self._precomp.copy()

        new_state._reg_adj = self._reg_adj
        new_state._t_sample = self._t_sample
        new_state._sin_values = self._sin_values
        new_state._sin_x_nsq = self._sin_x_nsq
        new_state._tol_costs = self._tol_costs
        new_state._tol_coeffs = self._tol_coeffs

        return new_state

    def within_tol(self, other: "State") -> bool:
        """ Check if two states are within iteration tolerance """
        if not isinstance(other, State):
            return False

        # Calculate L2 norm for a_coeff and b_coeff
        a_coeff_diff = np.linalg.norm(self.a_coeff - other.a_coeff) / np.sqrt(self.N)
        b_coeff_diff = np.linalg.norm(self.b_coeff - other.b_coeff) / np.sqrt(self.N)

        # Calculate max absolute difference for a_cost and b_cost
        a_cost_diff = abs(self.a_cost - other.a_cost) if self.a_cost is not None and other.a_cost is not None else 0
        b_cost_diff = abs(self.b_cost - other.b_cost) if self.b_cost is not None and other.b_cost is not None else 0

        self._tol_coeffs = np.maximum(a_coeff_diff, b_coeff_diff)
        self._tol_costs = np.maximum(a_cost_diff, b_cost_diff)

        # Check if differences are within tolerances
        return (a_coeff_diff < TOL_COEFFS and b_coeff_diff < TOL_COEFFS and
                a_cost_diff < TOL_COSTS and b_cost_diff < TOL_COSTS)

    def __str__(self):
        a_cost_str = f"{self.a_cost:.4f}" if self.a_cost is not None else "None"
        b_cost_str = f"{self.b_cost:.4f}" if self.b_cost is not None else "None"

        nc = N_COEFFS_TO_PRINT
        return (
            f"a_coeff = {self.a_coeff[:nc] if nc else self.a_coeff}\n"
            f"b_coeff = {self.b_coeff[:nc] if nc else self.a_coeff}\n"
            f"cost(a) = {a_cost_str},\t"
            f"cost(b) = {b_cost_str}\n"
            f"lambda = {self.lambd},\t"
            f"rho = {self.rho},\t"
            f"gamma = {self.gamma},\t"
            f"curr_gamma = {self._curr_gamma:.4f},\t"
            f"n = {self.N},\t"
            f"reg_adj = {self._reg_adj:.4f}\n"
            f"tol_coeffs = {self._tol_coeffs:.8f},\t"
            f"tol_costs = {self._tol_costs:.8f}"
        )

    def update(self, solve: int) -> "State":
        """ Solve for the best response of one trader with respect to the other """
        start_time = time.time()
        global ITER_SO_FAR

        if SOLVER == SCIPY:
            cost_function, init_guess = self._get_cost_function_and_guess(solve)
            args = (self.a_coeff, self.b_coeff)

            result = minimize(cost_function, init_guess, args=args, method='SLSQP')
            opt_coeff = result.x
        else:
            opt_coeff, result = self._minimize_cost_qp(solve)

        print(f"\nUpdated {'A' if solve == TRADER_A else 'B'}:, "
              f"Solve time = {(time.time() - start_time):.4f}s")
        if SOLVER == QP:
            print(f"Frob norms: N(P) = {np.linalg.norm(result['P']):.2f}, "
                  f"N(dP) = {np.linalg.norm(result['dP']):.2f}")

        # Generate a new updated state
        res_state = self._generate_new_state(solve, opt_coeff)
        res_state.a_cost = pp.cost_fn_prop_a_approx(res_state.a_coeff, res_state.b_coeff, self.lambd, self.rho)
        res_state.b_cost = pp.cost_fn_prop_b_approx(res_state.a_coeff, res_state.b_coeff, self.lambd, self.rho)

        if solve == TRADER_B:
            ITER_SO_FAR += 1
        return res_state

    def reg_term(self, x, x_prev, solve):
        n = np.arange(1, len(x) + 1)
        pi, f = np.pi, 0

        # Check if 2nd derivative regularization is enabled
        if self.reg_params.get('2nd_deriv') is not None:
            factor = self.curr_factor(ITER_SO_FAR)

            if factor > 0:
                # Calculate sample points if not already done
                if self._t_sample is None:
                    self._calc_t_sample()
                    self._sin_values = np.sin(pi * np.array(self._t_sample)[:, None] @ n[None, :])
                    self._sin_x_nsq = self._sin_values * n ** 2

                # Calculate second derivatives
                second_derivs = -pi * pi * self._sin_x_nsq @ x

                # Compute regularization term
                f = factor * (second_derivs.T @ second_derivs) / (len(self._t_sample) * len(x) ** 3)
                f *= (self.lambd ** 2 if solve == TRADER_B else 1)

        self._reg_adj = f  # Record the value of the regularization adjustment
        return f

    # def _calc_t_sample(self, force_recalc: bool = False) -> np.ndarray:
    #     if self._t_sample is None or force_recalc:
    #         cvx = self.reg_params.get('2nd_deriv')
    #         if cvx is not None:
    #             sample_type = cvx.get('type', UNIFORM)
    #             if sample_type == UNIFORM:
    #                 n_sample = cvx.get('n_sample', self.N)
    #                 full_sample = np.arange(0, 1, 1 / n_sample)
    #             else:
    #                 full_sample = sample_sine_wave(list(range(1, len(self.a_coeff) + 1)),
    #                                                cvx.get('pts_per_semiwave', 3))
    #             self._t_sample = np.array([t for t in full_sample if cvx['range'][0] <= t <= cvx['range'][1]])
    #     return self._t_sample

    def _calc_t_sample(self, force_recalc: bool = False) -> np.ndarray:
        # Recalculate sample points if needed
        if self._t_sample is None or force_recalc:
            cvx = self.reg_params.get('2nd_deriv')

            if cvx is not None:
                sample_type = cvx.get('type', UNIFORM)

                # Generate full sample based on sampling type
                if sample_type == UNIFORM:
                    n_sample = cvx.get('n_sample', self.N)
                    full_sample = np.arange(0, 1, 1 / n_sample)
                else:
                    full_sample = sample_sine_wave(
                        list(range(1, len(self.a_coeff) + 1)),
                        cvx.get('pts_per_semiwave', 3)
                    )

                # Filter sample points within the specified range
                self._t_sample = np.array([
                    t for t in full_sample if cvx['range'][0] <= t <= cvx['range'][1]
                ])

        return self._t_sample

    def _get_cost_function_and_guess(self, solve: int):
        if solve == TRADER_A:
            return (
                lambda x, a_coeff, b_coeff: pp.cost_fn_prop_a_approx(x, b_coeff, self.lambd, self.rho)
                                            + self.reg_term(x, a_coeff, solve),
                self.a_coeff
            )
        else:
            return (
                lambda x, a_coeff, b_coeff: pp.cost_fn_prop_b_approx(a_coeff, x, self.lambd, self.rho)
                                            + self.reg_term(x, b_coeff, solve),
                self.b_coeff
            )

    def _generate_new_state(self, solve: int, opt_coeff: np.ndarray) -> "State":
        n_iter = 0
        try:
            max_other_cost_inc = self.reg_params['iter_adjustment']['max_other_cost_inc']
        except KeyError:
            max_other_cost_inc = None

        try:
            max_iter_new_state = self.reg_params['iter_adjustment']['max_iter_line_search']
        except KeyError:
            max_iter_new_state = 0

        max_cost_increase = max_other_cost_inc * np.maximum(
            np.abs(self.b_cost), np.abs(self.a_cost) * self.lambd
        ) if max_other_cost_inc else None
        if solve == TRADER_A:
            a_coeff_new = opt_coeff * self._curr_gamma + self.a_coeff * (1 - self._curr_gamma)
            if max_other_cost_inc is None:
                return self.copy(a_coeff=a_coeff_new, b_coeff=self.b_coeff, calc_costs=False)
            else:
                while n_iter <= max_iter_new_state:
                    new_state = self.copy(a_coeff=a_coeff_new, b_coeff=self.b_coeff, calc_costs=False)
                    new_state.b_cost = pp.cost_fn_prop_b_approx(new_state.a_coeff, new_state.b_coeff, self.lambd,
                                                                self.rho)
                    if new_state.b_cost - self.b_cost <= max_cost_increase:
                        return new_state
                    else:
                        # a_coeff_new = self.a_coeff + (a_coeff_new - self.a_coeff) \
                        #               * np.abs(self.b_cost) * max_other_cost_inc / (new_state.b_cost - self.b_cost)
                        a_coeff_new = (self.a_coeff + a_coeff_new) / 2
                        n_iter += 1
        else:
            b_coeff_new = opt_coeff * self._curr_gamma + self.b_coeff * (1 - self._curr_gamma)
            if max_other_cost_inc is None:
                return self.copy(a_coeff=self.a_coeff, b_coeff=b_coeff_new, calc_costs=False)
            else:
                while n_iter <= max_iter_new_state:
                    new_state = self.copy(a_coeff=self.a_coeff, b_coeff=b_coeff_new, calc_costs=False)
                    new_state.a_cost = pp.cost_fn_prop_a_approx(new_state.a_coeff, new_state.b_coeff, self.lambd,
                                                                self.rho)
                    if new_state.a_cost - self.a_cost <= max_cost_increase / self.lambd:
                        return new_state
                    else:
                        # b_coeff_new = self.b_coeff + (b_coeff_new - self.b_coeff) \
                        #               * np.abs(self.a_cost) * max_other_cost_inc / (new_state.a_cost - self.a_cost)
                        b_coeff_new = (self.b_coeff + b_coeff_new) / 2
                        n_iter += 1

        return new_state

    def curr_factor(self, iter_so_far: int) -> float:
        if (self.reg_params is None) or (self.reg_params.get('2nd_deriv') is None):
            return 0
        factor_inc_rel = self.reg_params['2nd_deriv'].get('factor_inc_rel', 0)
        factor = self.reg_params['2nd_deriv']['factor']
        return factor * (1 + factor_inc_rel * iter_so_far)

    def _minimize_cost_qp(self, solve):
        # Precalculate matrices and vectors for the optimization
        _ = qp.precalc_obj_func_constants(self.N, self._precomp)
        self.reg_params.update({'t_sample': self._calc_t_sample()})
        self._precomp.update({'W': qp.reg_adjustment(self.b_coeff, self._precomp, self.reg_params)})

        if solve == TRADER_A:
            res = qp.min_cost_A_qp(self.b_coeff, self.rho, self.lambd, abs_tol=TOL_COEFFS,
                                   precomp=self._precomp, reg_params=self.reg_params,
                                   factor=self.curr_factor(ITER_SO_FAR))
        else:
            res = qp.min_cost_B_qp(self.a_coeff, self.rho, self.lambd, abs_tol=TOL_COEFFS,
                                   precomp=self._precomp, reg_params=self.reg_params,
                                   factor=self.curr_factor(ITER_SO_FAR))

        # Record the value of the regularization adjustment
        if self.curr_factor(ITER_SO_FAR) != 0:
            dP = res[1]['dP'] * (self.lambd ** 2 if solve == TRADER_B else 1)
            opt_coeff = res[0]
            self._reg_adj = 0.5 * opt_coeff.T @ dP @ opt_coeff
        return res

    def plot_solution_strategies(self, iter_hist: List["State"], ax: Any) -> Dict[str, Any]:
        """Check the solution against theoretical values.

        :param iter_hist: list of historical iteration values 
        :param ax: Matplotlib axis object for plotting
        :return: Statistics including function values and norms of differences
        """
        t_values = np.linspace(0, 1, N_PLOT_POINTS)

        a_approx = [fr.reconstruct_from_sin(t, self.a_coeff) + t for t in t_values]
        b_approx = [fr.reconstruct_from_sin(t, self.b_coeff) + t for t in t_values]

        if ax is not None:
            self._plot_values(ax, t_values, a_approx, b_approx, iter_hist)

        return {
            "a_approx": a_approx,
            "b_approx": b_approx,
        }

    def calculate_theoretical_values(self, t_values, trader_a) -> list:
        args_dict = {'kappa': self.kappa, 'lambd': self.lambd, 'trader_a': trader_a}
        return [tf.equil_2trader(t, **args_dict) for t in t_values]

    def _plot_values(self, ax, t_values, a_approx, b_approx, iter_hist: List["State"]) -> None:
        ax.plot(t_values, a_approx, label=r"$a^*(t)$", color="red", linestyle="-")
        ax.plot(t_values, b_approx, label=r"$b^*_{\lambda}(t)$", color="blue", linestyle="-")
        ax.plot(t_values, t_values, label="initial guess", color="grey", linestyle="--")
        n_iter = len(iter_hist) // 2
        ax.set_title("Equilibrium trading strategies for A and B\n" +
                     r"$\rho$" + f"={self.rho}, " + r"$\lambda$" + f"={self.lambd}, " +
                     f"N={self.N}, " + r'$\gamma=$' + f'{self.gamma:.3f} :: ' +
                     f'{n_iter} solver iterations', fontsize=12)
        ax.legend()
        ax.set_xlabel('t')
        ax.set_ylabel(r'$a(t), b_{\lambda}(t)$')
        ax.grid()

    @staticmethod
    def plot_state_space(iter_hist: List["State"], ax: Any) -> None:
        """ Plot evolution of costs for A and B over iterations
        """
        a_costs = [i.a_cost for i in iter_hist]
        b_costs = [i.b_cost for i in iter_hist]

        # Plot the points as circles
        ax.scatter(a_costs[1:-1], b_costs[1:-1], color='darkblue', s=20, label='(cost A, cost B)', alpha=0.4)

        # Connect the points with lines
        ax.plot(a_costs, b_costs, color='darkblue', linestyle='-', linewidth=1, alpha=0.6)

        # Highlight the starting and ending points
        ax.scatter(a_costs[0], b_costs[0], color='green', s=70, label='init guess')
        ax.scatter(a_costs[-1], b_costs[-1], color='red', s=70, label='solution')

        # Highlight every HIGHLIGHT_FREQ point in orange
        to_highlight = list(range(TO_HIGHLIGHT_FREQ, len(a_costs) - 1, TO_HIGHLIGHT_FREQ))
        ax.scatter([a_costs[i] for i in to_highlight], [b_costs[i] for i in to_highlight],
                   color='firebrick', s=70, marker='^',
                   label=f'every {TO_HIGHLIGHT_FREQ}th iter')

        # Label the starting and ending points
        x_offset = (a_costs[0] - a_costs[-1]) * LABEL_OFFSET_MULT
        ax.text(a_costs[0] - abs(x_offset), b_costs[0], 'init guess', fontsize=12, ha='right',
                color='black', weight='bold')
        ax.text(a_costs[-1] + x_offset, b_costs[-1], 'solution', fontsize=12,
                ha='left' if x_offset > 0 else 'right',
                color='black', weight='bold')

        ax.set_xlabel('Trader A cost')
        ax.set_ylabel('Trader B cost')

        ax.set_title(f'Trading Cost Convergence to Equilibrium\n' +
                     f'State Space Diagram: (x,y) = (cost A, cost B)')
        ax.legend()

    def plot_price_impact(self, ax: Any) -> None:
        """ Plot the price impact function for the final solution """
        ax.set_title('Price Impact of Combined A and B Trading Over Time')

        t_values = np.linspace(0, 1, N_PLOT_POINTS)
        dp_values = [pp.prop_price_impact_approx(t, self.a_coeff, self.b_coeff, self.lambd, self.rho) for t in t_values]
        ax.plot(t_values, dp_values, label='dP(0,t)', color='green')
        ax.set_xlabel('t')
        ax.set_ylabel('Price Impact')
        ax.legend()
        ax.grid()

    def plot_results(self, iter_hist: List["State"]) -> dict:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        if INCLUDE_SUPTITLE:
            plt.suptitle("Two-Trader Equilibrium Strategies, " +
                         r"$\rho$" + f"={self.rho}, " + r"$\lambda$" + f"={self.lambd}\n" +
                         f"{self.N} Fourier terms, " +
                         f"{len(iter_hist) // 2} solver iterations", fontsize=16)
        stats = self.plot_solution_strategies(iter_hist, axs[0])
        self.plot_price_impact(axs[1])
        self.plot_state_space(iter_hist, axs[2])
        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()
        return stats


def pickle_file_path(make_dirs: bool = False):
    # Define the directory and filename
    results_dir = os.path.join(CURRENT_DIR, cfg.SIM_RESULTS_DIR)
    # timestamp = datetime.now().strftime('%Y%m%d-%H%M')

    if 'iter_adjustment' in REG_PARAMS:
        other_lim = REG_PARAMS['iter_adjustment'].get('max_other_cost_inc', 0)
        other_lim_str = str(other_lim).replace('.', '_')
    else:
        other_lim_str = "NA"

    gamma_decay = REG_PARAMS.get('gamma_decay', 0)

    filename = cfg.SIM_FILE_NAME.format(
        LAMBD=LAMBD, RHO=RHO, OTHER_LIM=other_lim_str,
        GAMMA=GAMMA, GAMMA_DECAY=gamma_decay, N=N, SUFFIX=DATA_FILE_SUFFIX)

    file_path = os.path.join(results_dir, filename)

    # Create the directory if it does not exist
    if make_dirs:
        os.makedirs(results_dir, exist_ok=True)

    return file_path


def save_pickled_results(data: Any):
    # Define the directory and filename
    file_path = pickle_file_path(make_dirs=True)

    # Save the pickled serialization of sim_results
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_pickled_results() -> Any:
    # Define the directory and filename
    file_path = pickle_file_path(make_dirs=False)

    # Check if the file exists and load the pickled file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file {file_path} does not exist.')

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def main():
    # Initialize state
    state = State(N=N, rho=RHO, lambd=LAMBD, gamma=GAMMA, kappa=KAPPA, reg_params=REG_PARAMS)
    print("\nStarting Optimizatin, Solver = " +
          f"{'QP' if SOLVER == QP else 'SCIPY'}\n" +
          "Initial State: ")
    print(state)

    # Initialize the loop
    iter_hist = [state]
    converged_flag = False

    while True:
        print(f"\nStarting iteration: {len(iter_hist) // 2 + 1}")
        state_a = state.update(solve=TRADER_A)
        print("New state A:")
        print(str(state_a))
        state = state_a.update(solve=TRADER_B)
        print("New State :")
        print(str(state))
        iter_hist.extend([state_a, state])

        if state.within_tol(iter_hist[-3]):
            converged_flag = True
            break
        elif len(iter_hist) >= MAX_ITER * 2 + 1:
            print(f"\nMax Iteration {MAX_ITER} exceeded!")
            break
        elif max_abs_cost := max(abs(state.a_cost), abs(state.b_cost)) > MAX_ABS_COST:
            print(f"\nMax Abs Cost = {max_abs_cost} exceeded limit of {MAX_ABS_COST}!")
            break

    print(f"\n-------------------------------\n")
    if converged_flag:
        print(f"Converged after {len(iter_hist) // 2} iterations:")
    else:
        print(f"Did not converge.  Stopped after {len(iter_hist) // 2} iterations:")
    print("\nFinal State:")
    print(state)

    sim_stats = state.plot_results(iter_hist)

    # Save results
    if SAVE_RESULTS:
        save_pickled_results({'state': state, 'iter_hist': iter_hist, 'stats': sim_stats})


if __name__ == "__main__":
    main()
