"""  Numerically solving for the two-trader equilibrium
     Implements 2 contstraints:  overbuying and short selling for both traders.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from typing import Any, List, Dict
import os
import sys
import pickle
import config as cfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
import trading_funcs as tf
import cost_function_approx as ca
import fourier as fr
import qp_solvers as qp

# Parameters and Constants
LAMBD = 5
KAPPA = 1
N = 20
GAMMA = 0.8  # Fraction of the way to move towards the new solution (float e.g. 1.0)
# GAMMA = 0.8 if LAMBD > 5 else 0.2  # Fraction of the way to move towards the new solution (float e.g. 1.0)

TOL_COEFFS = 1e-4
TOL_COSTS = TOL_COEFFS
MAX_ITER = 100
MAX_ABS_COST = 1e10
N_PLOT_POINTS = 100
N_ITER_LINES = 4
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
INCLUDE_SUPTITLE = False
LABEL_OFFSET_MULT = 0.09
DEFAULT_N = 15

# Specify which solver to use for optimization
SCIPY, QP = range(2)
SOLVER = QP

# Initialize codes for the traders (like an enum)
TRADER_A, TRADER_B = range(2)

# Parameters to save simulation results
SAVE_RESULTS = True
DATA_FILE_SUFFIX = f"_g{GAMMA}"


class State:

    def __init__(self, **kwargs):
        # ToDo: write a docstring
        # Read model parameters
        self.kappa = kwargs.get('kappa', 0)
        self.lambd = kwargs.get('lambd', 1)
        self.gamma = kwargs.get('gamma', 0.8)

        # Read initial strategy coefficients
        N = kwargs.get('N', DEFAULT_N)
        a_coeff = kwargs.get('a_coeff', None)
        b_coeff = kwargs.get('b_coeff', None)

        self.N = self._initialize_n(a_coeff, b_coeff, N)
        self.a_coeff = self._initialize_coeff(a_coeff, self.N)
        self.b_coeff = self._initialize_coeff(b_coeff, self.N)

        self._validate_coeff_lengths()

        # Update the costs for A and B if required
        if kwargs.get("calc_costs", True):
            self.a_cost = ca.cost_fn_a_approx(self.a_coeff, self.b_coeff, self.kappa, self.lambd)
            self.b_cost = ca.cost_fn_b_approx(self.a_coeff, self.b_coeff, self.kappa, self.lambd)
        else:
            self.a_cost = None
            self.b_cost = None

        # Store precomputed variables to speed up the optimization
        self._precomp = {}

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
            kappa=kwargs.get('kappa', self.kappa),
            lambd=kwargs.get('lambd', self.lambd),
            gamma=kwargs.get('gamma', self.gamma),
            N=kwargs.get('N', self.N),
            a_coeff=kwargs.get('a_coeff', self.a_coeff),
            b_coeff=kwargs.get('b_coeff', self.b_coeff),
            calc_costs=kwargs.get('calc_costs', True)
        )

        # Overwrite any additional attributes provided in kwargs
        for key, value in kwargs.items():
            setattr(new_state, key, value)

        if ('copy_precomp' not in kwargs) or kwargs['copy_precomp']:
            new_state._precomp = self._precomp.copy()

        return new_state

    def within_tol(self, other: "State") -> bool:
        """ Check if two states are within iteration tolerance """
        if not isinstance(other, State):
            return False

        # Calculate L2 norm for a_coeff and b_coeff
        a_coeff_diff = np.linalg.norm(self.a_coeff - other.a_coeff)
        b_coeff_diff = np.linalg.norm(self.b_coeff - other.b_coeff)

        # Calculate max absolute difference for a_cost and b_cost
        a_cost_diff = abs(self.a_cost - other.a_cost) if self.a_cost is not None and other.a_cost is not None else 0
        b_cost_diff = abs(self.b_cost - other.b_cost) if self.b_cost is not None and other.b_cost is not None else 0

        # Check if differences are within tolerances
        return (a_coeff_diff < TOL_COEFFS and b_coeff_diff < TOL_COEFFS and
                a_cost_diff < TOL_COSTS and b_cost_diff < TOL_COSTS)

    def __str__(self):
        a_cost_str = f"{self.a_cost:.4f}" if self.a_cost is not None else "None"
        b_cost_str = f"{self.b_cost:.4f}" if self.b_cost is not None else "None"

        return (
            f"a_coeff = {self.a_coeff}\n"
            f"b_coeff = {self.b_coeff}\n"
            f"cost(a) = {a_cost_str},\t"
            f"cost(b) = {b_cost_str}\n"
            f"lambda = {self.lambd},\t"
            f"kappa = {self.kappa},\t"
            f"gamma = {self.gamma},\t"
            f"n = {self.N}"
        )

    def update(self, solve: int) -> "State":
        """ Solve for the best response of one trader with respect to the other """

        start_time = time.time()
        if SOLVER == SCIPY:
            cost_function, init_guess = self._get_cost_function_and_guess(solve)
            args = (self.a_coeff, self.b_coeff)

            result = minimize(cost_function, init_guess, args=args)
            opt_coeff = result.x
        else:
            opt_coeff, result = self._minimize_cost_qp(solve)

        print(f"Optimization time = {(time.time() - start_time):.4f}s")

        # Generate a new updated state
        res_state = self._generate_new_state(solve, opt_coeff)
        res_state.a_cost = ca.cost_fn_a_approx(self.a_coeff, self.b_coeff, self.kappa, self.lambd)
        res_state.b_cost = ca.cost_fn_b_approx(self.a_coeff, self.b_coeff, self.kappa, self.lambd)

        return res_state

    def _get_cost_function_and_guess(self, solve: int):
        if solve == TRADER_A:
            return (
                lambda x, a_coeff, b_coeff: ca.cost_fn_a_approx(x, b_coeff, self.kappa, self.lambd),
                self.a_coeff
            )
        else:
            return (
                lambda x, a_coeff, b_coeff: ca.cost_fn_b_approx(a_coeff, x, self.kappa, self.lambd),
                self.b_coeff
            )

    def _generate_new_state(self, solve: int, opt_coeff: np.ndarray) -> "State":
        if solve == TRADER_A:
            a_coeff_new = opt_coeff * self.gamma + self.a_coeff * (1 - self.gamma)
            return self.copy(a_coeff=a_coeff_new, b_coeff=self.b_coeff, calc_costs=False)
        else:
            b_coeff_new = opt_coeff * self.gamma + self.b_coeff * (1 - self.gamma)
            return self.copy(a_coeff=self.a_coeff, b_coeff=b_coeff_new, calc_costs=False)

    def _minimize_cost_qp(self, solve):
        # Precalculate matrices and vectors for the optimization
        _ = qp.precalc_obj_func_constants(self.N, self._precomp)

        if solve == TRADER_A:
            return qp.min_cost_A_qp(self.b_coeff, self.kappa, self.lambd, abs_tol=TOL_COEFFS,
                                    precomp=self._precomp)
        else:
            return qp.min_cost_B_qp(self.a_coeff, self.kappa, self.lambd, abs_tol=TOL_COEFFS,
                                    precomp=self._precomp)

    def check_v_theo(self, iter_hist: List["State"], ax: Any) -> Dict[str, Any]:
        """Check the solution against theoretical values.

        :param iter_hist: list of historical iteration values 
        :param ax: Matplotlib axis object for plotting
        :return: Statistics including function values and norms of differences
        """
        t_values = np.linspace(0, 1, N_PLOT_POINTS)

        a_theo = self.calculate_theoretical_values(t_values, trader_a=True)
        b_theo = self.calculate_theoretical_values(t_values, trader_a=False)

        a_approx = [fr.reconstruct_from_sin(t, self.a_coeff) + t for t in t_values]
        b_approx = [fr.reconstruct_from_sin(t, self.b_coeff) + t for t in t_values]

        a_diff = np.array(a_theo) - np.array(a_approx)
        b_diff = np.array(b_theo) - np.array(b_approx)

        l2_a = np.linalg.norm(a_diff, 2) / np.sqrt(len(a_diff))
        l2_b = np.linalg.norm(b_diff, 2) / np.sqrt(len(b_diff))

        if ax is not None:
            self._plot_values(ax, t_values, a_theo, b_theo, a_approx, b_approx,
                              l2_a, l2_b, iter_hist)

        return {
            "a_theo": a_theo,
            "b_theo": b_theo,
            "a_approx": a_approx,
            "b_approx": b_approx,
            "L2_a": l2_a,
            "L2_b": l2_b
        }

    def calculate_theoretical_values(self, t_values, trader_a) -> list:
        args_dict = {'kappa': self.kappa, 'lambd': self.lambd, 'trader_a': trader_a}
        return [tf.equil_2trader(t, **args_dict) for t in t_values]

    def _plot_values(self, ax, t_values, a_theo, b_theo, a_approx, b_approx, l2_a, l2_b,
                     iter_hist: List["State"]) -> None:
        ax.scatter(t_values, a_theo, s=10, label=r"$a_{eq}(t)$", color="red")
        ax.scatter(t_values, b_theo, s=10, label=r"$b_{{\lambda},eqf}(t)$", color="grey")
        ax.plot(t_values, a_approx, label=r"$a^*(t)$", color="green", linestyle="-")
        ax.plot(t_values, b_approx, label=r"$b^*_{\lambda}(t)$", color="blue", linestyle="-")
        n_iter = len(iter_hist) // 2
        ax.set_title("Theoretical and approximated trading strategies\n" +
                     r"$\kappa$" + f"={self.kappa}, " + r"$\lambda$" + f"={self.lambd}, " +
                     f"N={self.N}, " + r'$\gamma=$' + f'{self.gamma:.1f} :: ' +
                     f'{n_iter} solver iterations\n' +
                     r"$L_2(a_{diff})=$" + f"{l2_a:.4f}, " +
                     r"$L_2(b_{diff})=$" + f"{l2_b:.4f}",
                     fontsize=12)
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
        ax.plot(a_costs, b_costs, color='darkblue', linestyle='-', linewidth=1, alpha=0.4)

        # Highlight the starting and ending points
        ax.scatter(a_costs[0], b_costs[0], color='green', s=70, label='init guess')
        ax.scatter(a_costs[-1], b_costs[-1], color='red', s=70, label='solution')

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

    @staticmethod
    def plot_func_convergence(iter_hist: List["State"], res_coeffs: Dict[str, Any],
                              ax: Any, start_finish_lines: bool = True) -> None:
        """ Plot the difference between theoretical schedule and the approximations
        """
        t_values = np.linspace(0, 1, N_PLOT_POINTS)
        a_theo, b_theo = res_coeffs['a_theo'], res_coeffs['b_theo']

        if start_finish_lines:
            iter_idx = [0, len(iter_hist) - 1]
        else:
            iter_idx = State._generate_idx_of_iters_to_plot(len(iter_hist), N_ITER_LINES)

        for i_line, iter_idx in enumerate(iter_idx):
            a_diffs = State._diff_vs_theo(iter_hist, iter_idx, t_values, a_theo, is_a=True)
            b_diffs = State._diff_vs_theo(iter_hist, iter_idx, t_values, b_theo, is_a=False)

            linestyle = LINE_STYLES[i_line % len(LINE_STYLES)]
            if start_finish_lines:
                line_code = "init guess" if i_line == 0 else "solution"
                if iter_idx == 0:
                    label_a = r"$\Delta a^0(t)$, (init guess)"
                    label_b = r"$\Delta b^0_{\lambda}(t)$, (init guess)"
                else:
                    label_a = r"$\Delta a^{i_{max}}(t)$, (final)"
                    label_b = r"$\Delta b^{i_{max}}_{\lambda}(t)$, (final)"
            else:
                iter_code = (iter_idx + 1) // 2
                line_code = f"iter={iter_code}"
                label_a = r"$\Delta a(t)$, " + line_code
                label_b = r"$\Delta b_{\lambda}(t)$, " + line_code

            ax.plot(t_values, a_diffs, label=label_a,
                    color="red", linestyle=linestyle)
            ax.plot(t_values, b_diffs, label=label_b,
                    color="blue", linestyle=linestyle)

            ax.set_title("Solver Approximations vs. Equilibrium\n" +
                         "after i solver iterations.\n" +
                         r"$\Delta a^i = a_{eq} - a^i$, $\Delta b^i_{\lambda} = b_{{\lambda},eq} - b^i_{\lambda}$")
            ax.set_xlabel('t')
            ax.set_ylabel(r'$a(t), b_{\lambda}(t)$ residuals vs. equilibrium')
        ax.legend()
        ax.grid()

    @staticmethod
    def _diff_vs_theo(iter_hist: List["State"], i: int, t_values: np.ndarray,
                      theo_vals: List[float], is_a: bool) -> List[float]:
        i_state = iter_hist[i]
        coeff_approx = i_state.a_coeff if is_a else i_state.b_coeff
        val_approx = [fr.reconstruct_from_sin(t, coeff_approx) + t for t in t_values]
        return [appx - theo for appx, theo in zip(val_approx, theo_vals)]

    def _find_theo_fourier_coeffs(self):
        # Find theo Fourier Coeffs
        def a_func(t, kappa, lambd):
            arg_dict = {'kappa': kappa, 'lambd': lambd, 'trader_a': True}
            return tf.equil_2trader(t, **arg_dict)

        def b_func(t, kappa, lambd):
            arg_dict = {'kappa': kappa, 'lambd': lambd, 'trader_a': False}
            return tf.equil_2trader(t, **arg_dict)

        a_coeff_theo, b_coeff_theo = fr.find_fourier_coefficients((a_func, b_func), self.kappa, self.lambd, self.N)
        return a_coeff_theo, b_coeff_theo

    def plot_state_space_v_theo(self, iter_hist: List["State"], res_coeffs: Dict[str, Any],
                                ax: Any) -> None:
        """ Plot eveolution of costs for A and B vs. Teoretical over iterations
        """
        # Calculate theoretical equilibrium estimates
        a_coeff_theo, b_coeff_theo = self._find_theo_fourier_coeffs()
        a_cost_theo = ca.cost_fn_a_approx(a_coeff_theo, b_coeff_theo, self.kappa, self.lambd)
        b_cost_theo = ca.cost_fn_b_approx(a_coeff_theo, b_coeff_theo, self.kappa, self.lambd)

        # We are interested in abs differences between approximated and theo costs
        a_res = [abs(i.a_cost - a_cost_theo) for i in iter_hist]
        b_res = [abs(i.b_cost - b_cost_theo) for i in iter_hist]
        # Plot the points as circles
        t_values = np.linspace(0, 1, N_PLOT_POINTS)
        ax.scatter(a_res[1:-1], b_res[1:-1], color='darkblue', s=20, label=r'$|\Delta c^i_a|, |\Delta c^i_b|$',
                   alpha=0.4)

        # Connect the points with lines
        ax.plot(a_res, b_res, color='darkblue', linestyle='-', linewidth=1, alpha=0.4)

        # Highlight the starting and ending points
        ax.scatter(a_res[0], b_res[0], color='green', s=70, label='init guess')
        ax.scatter(a_res[-1], b_res[-1], color='red', s=70, label='solution')

        # Label the starting and ending points
        x_offset = abs(a_res[0] - a_res[-1]) * LABEL_OFFSET_MULT
        ax.text(a_res[0] - x_offset, b_res[0], 'init guess', fontsize=12, ha='right',
                color='black', weight='bold')
        ax.text(a_res[-1] + x_offset, b_res[-1], 'solution', fontsize=12, ha='left',
                color='black', weight='bold')

        ax.set_xlabel(r'$|c(a^i)-c(a_{eq})|$')
        ax.set_ylabel(r'$|c(b^i_{\lambda})-c(b_{{\lambda},eq})|$')
        ax.set_title('Absolute Diff. between Approx. and Equil. Costs\n'
                     r'$\Delta c^i_a =|c(a^i)-c(a_{eq})|$ vs. $\Delta c^i_b = |c(b^i_{\lambda})-c(b_{{\lambda},eq})|$')
        ax.legend()

    @staticmethod
    def _generate_idx_of_iters_to_plot(n_iter: int, n_lines: int) -> List[int]:
        idx_step = max(n_iter / (N_ITER_LINES - 1), 1);
        iter_idx = [round(i * idx_step) for i in range(n_lines) if i * idx_step < n_iter - 1]
        iter_idx.append(n_iter - 1)
        return iter_idx

    def plot_results(self, iter_hist: List["State"]) -> dict:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        if INCLUDE_SUPTITLE:
            plt.suptitle("Two-Trader Equilibrium Strategies, " +
                         r"$\kappa$" + f"={self.kappa}, " + r"$\lambda$" + f"={self.lambd}\n" +
                         f"{self.N} Fourier terms, " +
                         f"{len(iter_hist) // 2} solver iterations", fontsize=16)
        stats = self.check_v_theo(iter_hist, axs[0, 0])
        print(f"\nAccuracy: L2(a) = {stats['L2_a']:.5f}, L2(b) = {stats['L2_b']:.5f}")
        self.plot_state_space(iter_hist, axs[1, 0])
        self.plot_func_convergence(iter_hist, stats, axs[0, 1])
        self.plot_state_space_v_theo(iter_hist, stats, axs[1, 1])
        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()
        return stats


def pickle_file_path(make_dirs: bool = False):
    # Define the directory and filename
    results_dir = os.path.join(CURRENT_DIR, cfg.SIM_RESULTS_DIR)
    # timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    filename = cfg.SIM_FILE_NAME.format(
        LAMBD=LAMBD, KAPPA=KAPPA, N=N, SUFFIX=DATA_FILE_SUFFIX)
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
    state = State(N=N, kappa=KAPPA, lambd=LAMBD, gamma=GAMMA)
    print("\nStarting Optimizatin, Solver = " +
          f"{'QP' if SOLVER == 'QP' else 'SCIPY'}\n" +
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
        elif len(iter_hist) > MAX_ITER * 2 + 1:
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
