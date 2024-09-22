"""  Run optimization for different values of N and compare the quality fo the approximation
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from typing import Any, List, Dict, Tuple
import trading_funcs as tf
import os
import sys
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
import cost_function_approx as ca
import fourier as fr

# Parameters and Constants
LAMBD = 5
KAPPA = 20
N_LIST = [5, 7, 9, 12, 15, 20, 25, 30, 50]
IDX_TO_PLOT = [0, 1]  #, 4, 8]  # , 1, 2, 3, 4]

DEFAULT_N = 7
TOL_COEFFS = 1e-3
TOL_COSTS = TOL_COEFFS
FRACTION_MOVE = 0.2
MAX_ITER = 250
MAX_ABS_COST = 1e10
N_PLOT_POINTS = 100
N_ITER_LINES = 4
LINE_STYLES = ['-', '-.', ':', (0, (3, 1, 1, 1)), '--']
LINE_MARKERS = ['v', 'o', 'x', 's', '^']
GAMMA = 1  # Put it at the end since it never changes

# Which trader we are solving for
TRADER_A, TRADER_B = range(2)

# Simulaton paramters
REBUILD_SIM_DATA = True  # If false, load from data file
SIM_RESULTS_DIR = "../results/sim_results"
SIM_FILE_NAME = f"sim_data_l{LAMBD}_k{KAPPA}.pkl"
PLOT_TRADERS = [TRADER_A, TRADER_B]  # , TRADER_B]
TRADER_THEO_COLORS = ['darkred', 'royalblue']
TRADER_COLORS = {0: ['red', 'firebrick', 'indianred', 'lightcoral', 'salmon'],
                 1: ['blue', 'deepskyblue', 'dodgerblue', 'steelblue', 'lightblue']}


class State:

    def __init__(self, a_coeff: np.ndarray = None, b_coeff: np.ndarray = None, n: int = None,
                 calc_costs: bool = True):
        self.n = self._initialize_n(a_coeff, b_coeff, n)
        self.a_coeff = self._initialize_coeff(a_coeff, self.n)
        self.b_coeff = self._initialize_coeff(b_coeff, self.n)

        self._validate_coeff_lengths()

        if calc_costs:
            self.a_cost = ca.cost_fn_a_approx(self.a_coeff, self.b_coeff, KAPPA, LAMBD)
            self.b_cost = ca.cost_fn_b_approx(self.a_coeff, self.b_coeff, KAPPA, LAMBD)
        else:
            self.a_cost = None
            self.b_cost = None

    @staticmethod
    def _initialize_n(a_coeff: np.ndarray, b_coeff: np.ndarray, n: int) -> int:
        if a_coeff is None and b_coeff is None:
            return n if n is not None else DEFAULT_N
        return max(len(a_coeff) if a_coeff is not None else 0, len(b_coeff) if b_coeff is not None else 0)

    @staticmethod
    def _initialize_coeff(coeff: np.ndarray, n: int) -> np.ndarray:
        return coeff if coeff is not None else np.zeros(n)

    def _validate_coeff_lengths(self) -> None:
        if len(self.a_coeff) != len(self.b_coeff):
            raise ValueError(
                f"Fourier coeff array length mismatch: len(a)={len(self.a_coeff)}, len(b)={len(self.b_coeff)}")

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
            f"cost(b) = {b_cost_str}"
        )

    def update(self, solve: int) -> "State":
        """ Solve for the best response of one trader with respect to the other """

        cost_function, init_guess = self._get_cost_function_and_guess(solve)
        args = (self.a_coeff, self.b_coeff)

        # Minimize the cost function
        start_time = time.time()
        result = minimize(cost_function, init_guess, args=args)
        print(f"Optimization time = {(time.time() - start_time):.4f}s")

        # Generate a new updated state
        res_state = self._generate_new_state(solve, result.x)
        res_state.a_cost = ca.cost_fn_a_approx(self.a_coeff, self.b_coeff, KAPPA, LAMBD)
        res_state.b_cost = ca.cost_fn_b_approx(self.a_coeff, self.b_coeff, KAPPA, LAMBD)

        return res_state

    def _get_cost_function_and_guess(self, solve: int):
        if solve == TRADER_A:
            return (
                lambda x, a_coeff, b_coeff: ca.cost_fn_a_approx(x, b_coeff, KAPPA, LAMBD),
                self.a_coeff
            )
        else:
            return (
                lambda x, a_coeff, b_coeff: ca.cost_fn_b_approx(a_coeff, x, KAPPA, LAMBD),
                self.b_coeff
            )

    def _generate_new_state(self, solve: int, result_x):
        if solve == TRADER_A:
            a_coeff_new = result_x * FRACTION_MOVE + self.a_coeff * (1 - FRACTION_MOVE)
            return State(a_coeff_new, self.b_coeff, calc_costs=False)
        else:
            b_coeff_new = result_x * FRACTION_MOVE + self.b_coeff * (1 - FRACTION_MOVE)
            return State(self.a_coeff, b_coeff_new, calc_costs=False)

    def check_v_theo(self, lambd: float, kappa: float, ax: Any) -> Dict[str, Any]:
        """Check the solution against theoretical values.

        :param lambd: Scale of trader B
        :param kappa: Permanent impact
        :param ax: Matplotlib axis object for plotting
        :return: Statistics including function values and norms of differences
        """
        t_values = np.linspace(0, 1, N_PLOT_POINTS)

        a_theo = self.calculate_theoretical_values(t_values, kappa, lambd, trader_a=True)
        b_theo = self.calculate_theoretical_values(t_values, kappa, lambd, trader_a=False)

        a_approx = [fr.reconstruct_from_sin(t, self.a_coeff) + GAMMA * t for t in t_values]
        b_approx = [fr.reconstruct_from_sin(t, self.b_coeff) + GAMMA * t for t in t_values]

        if ax is not None:
            self._plot_values(ax, t_values, a_theo, b_theo, a_approx, b_approx)

        return {
            "a_theo": a_theo,
            "b_theo": b_theo,
            "a_approx": a_approx,
            "b_approx": b_approx
        }

    @staticmethod
    def calculate_theoretical_values(t_values, kappa, lambd, trader_a) -> list:
        params = {'kappa': kappa, 'lambd': lambd, 'trader_a': trader_a}
        return [tf.equil_2trader(t, params) for t in t_values]

    @staticmethod
    def _plot_values(ax, t_values, a_theo, b_theo, a_approx, b_approx):
        ax.scatter(t_values, a_theo, s=20, label="a theo", color="red")
        ax.scatter(t_values, b_theo, s=20, label="b theo", color="grey")
        ax.plot(t_values, a_approx, label="a approx", color="green", linestyle="-")
        ax.plot(t_values, b_approx, label="b approx", color="blue", linestyle="-")
        ax.set_title(r"Theoretical and Approximated Trading Schedules")
        ax.legend()
        ax.set_xlabel('t')
        ax.set_ylabel('a(t), b(t)')
        ax.grid()

    @staticmethod
    def plot_state_space(iter_hist: List["State"], ax: Any) -> None:
        """ Plot evolution of costs for A qnd B over iterations
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
        ax.set_title(f'Trading Cost Convergence to Equilibrium\n'
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
            else:
                iter_code = (iter_idx + 1) // 2
                line_code = f"iter={iter_code}"

            ax.plot(t_values, a_diffs, label=r"$\Delta a(t)$, " + line_code,
                    color="red", linestyle=linestyle)
            ax.plot(t_values, b_diffs, label=r"$\Delta b(t)$, " + line_code,
                    color="blue", linestyle=linestyle)
            ax.set_title("Convergence to Theoretical Value\n" +
                         f"Diffs between approx. and theo. a(t),b(t) curves")
            ax.set_xlabel('t')
            ax.set_ylabel('a(t), b(t) residuals vs. theoretical')
            ax.legend()
            ax.grid()

    @staticmethod
    def _diff_vs_theo(iter_hist: List["State"], i: int, t_values: np.ndarray,
                      theo_vals: List[float], is_a: bool) -> List[float]:
        i_state = iter_hist[i]
        coeff_approx = i_state.a_coeff if is_a else i_state.b_coeff
        val_approx = [fr.reconstruct_from_sin(t, coeff_approx) + GAMMA * t for t in t_values]
        return [appx - theo for appx, theo in zip(val_approx, theo_vals)]

    @staticmethod
    def _find_theo_fourier_coeffs(n_coeffs):
        # Find theo Fourier Coeffs
        def a_func(t, kappa, lambd):
            params = {'kappa': kappa, 'lambd': lambd, 'trader_a': True}
            return tf.equil_2trader(t, params)

        def b_func(t, kappa, lambd):
            params = {'kappa': kappa, 'lambd': lambd, 'trader_a': False}
            return tf.equil_2trader(t, params)

        a_coeff_theo, b_coeff_theo = fr.find_fourier_coefficients((a_func, b_func), KAPPA, LAMBD, n_coeffs, GAMMA)
        return a_coeff_theo, b_coeff_theo

    def plot_state_space_v_theo(self, iter_hist: List["State"], res_coeffs: Dict[str, Any],
                                ax: Any) -> None:
        """ Plot eveolution of costs for A and B vs. Teoretical over iterations
        """
        # Calculate theoretical equilibrium estimates
        a_coeff_theo, b_coeff_theo = State._find_theo_fourier_coeffs(self.n)
        a_cost_theo = ca.cost_fn_a_approx(a_coeff_theo, b_coeff_theo, KAPPA, LAMBD)
        b_cost_theo = ca.cost_fn_b_approx(a_coeff_theo, b_coeff_theo, KAPPA, LAMBD)

        # We are interested in abs differences between approximated and theo costs
        a_res = [abs(i.a_cost - a_cost_theo) for i in iter_hist]
        b_res = [abs(i.b_cost - b_cost_theo) for i in iter_hist]
        # Plot the points as circles
        t_values = np.linspace(0, 1, N_PLOT_POINTS)
        ax.scatter(a_res[1:-1], b_res[1:-1], color='darkblue', s=20, label=r'$|\Delta A|, |\Delta B|$', alpha=0.4)

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

        ax.set_xlabel(r'$|A_{appox}-A_{theo}|$')
        ax.set_ylabel(r'$|B_{appox}-B_{theo}|$')
        ax.set_title('Absolute Diff. between Approx. and Theo. Costs\n'
                     r'$|A_{appox}-A_{theo}|$ vs. $|B_{approx} - B_{theo}|$')
        ax.legend()

    @staticmethod
    def _generate_idx_of_iters_to_plot(n_iter: int, n_lines: int) -> List[int]:
        idx_step = max(n_iter / (N_ITER_LINES - 1), 1);
        iter_idx = [round(i * idx_step) for i in range(n_lines) if i * idx_step < n_iter - 1]
        iter_idx.append(n_iter - 1)
        return iter_idx

    def plot_results(self, iter_hist: List["State"]):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plt.suptitle("Two-Trader Equilibrium Strategies, " +
                     r"$\kappa$" + f"={KAPPA}, " + r"$\lambda$" + f"={LAMBD}\n" +
                     f"{self.n} Fourier terms, " +
                     f"{len(iter_hist) // 2} solver iterations", fontsize=16)
        res_coeffs = self.check_v_theo(LAMBD, KAPPA, axs[0, 0])
        self.plot_state_space(iter_hist, axs[1, 0])
        self.plot_func_convergence(iter_hist, res_coeffs, axs[0, 1])
        self.plot_state_space_v_theo(iter_hist, res_coeffs, axs[1, 1])
        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()


def plot_2x2(sims: List[State]):
    """ Generate a 2x2 graph plot"""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.suptitle("Approximation Quality vs. Number of Fourier Terms\n" +
                 r"Two-Trader Equilibrium: $\kappa$" + f"={KAPPA}, " + r"$\lambda$" + f"={LAMBD}")
    plot_schedules(sims, axs[0, 0])
    # self.plot_state_space(iter_hist, axs[1, 0])
    # self.plot_func_convergence(iter_hist, res_coeffs, axs[0, 1])
    # self.plot_state_space_v_theo(iter_hist, res_coeffs, axs[1, 1])
    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


def plot_schedules(sims: list, ax: Any) -> None:
    """
    :param sims:  list of simulation results
    :param ax: pointed to the subplot
    :return:
    """
    t_values = np.linspace(0, 1, N_PLOT_POINTS)
    trader_dict = {}  # data dictionary
    for trader in PLOT_TRADERS:
        theo = State.calculate_theoretical_values(t_values, KAPPA, LAMBD, trader_a=(trader == TRADER_A))
        trader_code = "A" if trader == TRADER_A else "B"
        color = TRADER_COLORS[trader]
        ax.plot(t_values, theo, label=trader_code + " theo", color=TRADER_THEO_COLORS[trader])

        for i, idx in enumerate(IDX_TO_PLOT):
            state, series_dict = sims[N_LIST[i]]
            approx_vals = series_dict[f'{trader_code.lower()}_approx']
            # linestyle = LINE_STYLES[(i + 1) % len(LINE_STYLES)]
            marker = LINE_MARKERS[i % len(LINE_MARKERS)]
            color = TRADER_COLORS[trader][i]
            ax.scatter(t_values, approx_vals, s=20, label=trader_code + f" approx, N={N_LIST[idx]}",
                       color=color, marker=marker, alpha=0.5)

    ax.set_title(r"Theoretical and Approximated Trading Schedules")
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('a(t), b(t)')
    ax.grid()


def _pickle_file_path(make_dirs: bool = False):
    # Define the directory and filename
    results_dir = os.path.join(CURRENT_DIR, SIM_RESULTS_DIR)
    # timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    filename = SIM_FILE_NAME
    file_path = os.path.join(results_dir, filename)

    # Create the directory if it does not exist
    if make_dirs:
        os.makedirs(results_dir, exist_ok=True)

    return file_path


def _save_pickled_results(data: Any):
    # Define the directory and filename
    file_path = _pickle_file_path(make_dirs=True)

    # Save the pickled serialization of sim_results
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def _load_pickled_results() -> Any:
    # Define the directory and filename
    file_path = _pickle_file_path(make_dirs=False)

    # Check if the file exists and load the pickled file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file {file_path} does not exist.')

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def run_one_n(n: int) -> Tuple[State, dict]:
    # Initialize state
    print(f"\nStart optimization for n={n} terms:")
    state = State(n=n)

    # Initialize the loop
    iter_hist = [state]
    converged_flag = False

    while True:
        print("\nStarting iteration:")
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

    # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    series_dict = state.check_v_theo(LAMBD, KAPPA, ax=None)
    # plt.show()
    return state, series_dict


def main():
    if REBUILD_SIM_DATA:
        sim_results = {}

        for n in N_LIST:
            sim_results[n] = run_one_n(n)

        _save_pickled_results(sim_results)

    else:
        sim_results = _load_pickled_results()

    plot_2x2(sim_results)


if __name__ == "__main__":
    main()
