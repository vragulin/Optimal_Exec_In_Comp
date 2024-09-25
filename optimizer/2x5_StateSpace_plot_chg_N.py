""" Generate a 2x5 plot using previously saved simulation results
    The program expects data for every scenario in a pickle file in the results directory.
    (locaton and file nae format specified in config.py)
    In this version we iterate over Gamam (which is the FRACTION_MOVE in the optimizer)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from collections import namedtuple
import sys
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
import cost_function_approx as ca
import fourier as fr
import trading_funcs as tf
import config as cfg

# Global Parameters
Params = namedtuple('Parameters', ['lambd', 'kappa', 'n', 'gamma'])
sim_runs_params = [
    # lambd, kappa, n, gamma
    Params(5, 20, 3, 0.5),
    Params(5, 20, 5, 0.5),
    Params(5, 20, 9, 0.5),
    Params(5, 20, 15, 0.5),
    Params(5, 20, 21, 0.5),
]
# sim_runs_params = [
#     Params(5, 1, 10),
#     Params(10, 1, 10),
#     Params(2, 20, 10),
#     Params(1, 20, 10),
#     Params(1.5, 30, 10),
# ]

# Plot parameters
LABEL_OFFSET_MULT = 0.09
INCLUDE_SUPTITLE = False


# Suffix to identify the relevant the data files
DATA_FILE_SUFFIX = "_g{FRACTION_MOVE}"  # "" for unconstrained, "_cons" for constrained


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

        a_diff = np.array(a_theo) - np.array(a_approx)
        b_diff = np.array(b_theo) - np.array(b_approx)

        l2_a = np.linalg.norm(a_diff, 2) / np.sqrt(len(a_diff))
        l2_b = np.linalg.norm(b_diff, 2) / np.sqrt(len(b_diff))

        # l2_a_chk = np.sqrt(a_diff @ a_diff)
        # l2_b_chk = np.sqrt(b_diff @ b_diff)
        #
        # print("\n Check manually-computed L2 norm vs. np.linalg.norm()")
        # print(f"L2(a) = {l2_a}, L2_chk(a) = {l2_a_chk}")
        # print(f"L2(b) = {l2_b}, L2_chk(b) = {l2_b_chk}")

        return {
            "a_theo": a_theo,
            "b_theo": b_theo,
            "a_approx": a_approx,
            "b_approx": b_approx,
            "L2_a": l2_a,
            "L2_b": l2_b
        }

    @staticmethod
    def calculate_theoretical_values(t_values, kappa, lambd, trader_a) -> list:
        params = {'kappa': kappa, 'lambd': lambd, 'trader_a': trader_a}
        return [tf.equil_2trader(t, params) for t in t_values]


def pickle_file_path(p: Params) -> str:
    # Define the directory and filename
    results_dir = os.path.join(CURRENT_DIR, cfg.SIM_RESULTS_DIR)

    # timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    filename = cfg.SIM_FILE_NAME.format(
        LAMBD=p.lambd, KAPPA=p.kappa, N=p.n,
        SUFFIX=DATA_FILE_SUFFIX.format(FRACTION_MOVE=p.gamma)
    )
    file_path = os.path.join(results_dir, filename)

    return file_path


def fetch_one_sim_file(p: Params) -> dict:
    """ Load simulation results for one file
    :param p: Parameters tuple
    """
    file_path = pickle_file_path(p)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        print(f"File {file_path} not found")

    return data


def plot_eq_and_approx_strats(results: dict, params: Params, ax: Any) -> dict:
    lambd, kappa, n, frac_move = params
    state = results['state']
    n_iter = len(results['iter_hist']) // 2
    t_values = np.linspace(0, 1, cfg.N_PLOT_POINTS)

    a_theo = State.calculate_theoretical_values(t_values, kappa, lambd, trader_a=True)
    b_theo = State.calculate_theoretical_values(t_values, kappa, lambd, trader_a=False)

    a_approx = [fr.reconstruct_from_sin(t, state.a_coeff) + cfg.GAMMA * t for t in t_values]
    b_approx = [fr.reconstruct_from_sin(t, state.b_coeff) + cfg.GAMMA * t for t in t_values]

    a_diff = np.array(a_theo) - np.array(a_approx)
    b_diff = np.array(b_theo) - np.array(b_approx)

    l2_a = np.linalg.norm(a_diff, 2) / np.sqrt(len(a_diff))
    l2_b = np.linalg.norm(b_diff, 2) / np.sqrt(len(b_diff))

    if ax is not None:
        ax.scatter(t_values, a_theo, s=10, label=r"$a_{eq}(t)$", color="red")
        ax.scatter(t_values, b_theo, s=10, label=r"$b_{eq,\lambda}(t)$", color="grey")
        ax.plot(t_values, a_approx, label=r"$a^*(t)$", color="green", linestyle="-")
        ax.plot(t_values, b_approx, label=r"$b^*_{\lambda}(t)$", color="blue", linestyle="-")
        ax.set_title(r"$\kappa$" + f"={kappa}, " + r"$\lambda$" + f"={lambd}, " +
                     f"N={state.n}, " + r"$\gamma$" + f"={frac_move}, " +
                     f"\n({n_iter} iterations)\n" +
                     # r"$L_2(a_{eq}-a^*)=$" + f"{l2_a:.4f}," +
                     # r"$L_2(b_{eq}-b^*)=$" + f"{l2_b:.4f}",
                     r"$L_2(a_{diff})=$" + f"{l2_a:.4f}, " +
                     r"$L_2(b_{diff})=$" + f"{l2_b:.4f}",
                     fontsize=12)
        ax.legend()
        ax.set_xlabel('t')
        ax.set_ylabel(r'$a(t), b_{\lambda}(t)$')
        ax.grid()

    return {
        "a_theo": a_theo,
        "b_theo": b_theo,
        "a_approx": a_approx,
        "b_approx": b_approx,
        "L2_a": l2_a,
        "L2_b": l2_b
    }


def plot_state_space(results: dict, params: Params, ax: Any) -> None:
    """ Plot evolution of costs for A and B over iterations
      """
    state, iter_hist = results['state'], results['iter_hist']

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
                 f'(x,y) = (cost A, cost B)')
    ax.legend()


def main():
    # Load simulation results
    sim_results = [fetch_one_sim_file(p) for p in sim_runs_params]

    # Plot results
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    if INCLUDE_SUPTITLE:
        plt.suptitle(r"Two-Trade Equilibrium Strategies -- Approximation Accuracy vs. Number of Fourier Terms (N)"
                     "\nTop: Analytic Solutions vs. Fourier Approximation\n "
                     "Bottom: State Space Diagram of Solver Convergence Path in Terms of A,B Trading Costs",
                     fontsize=16)

    for i, (params, results) in enumerate(zip(sim_runs_params, sim_results)):
        t = np.linspace(0, 1, 100)
        _ = plot_eq_and_approx_strats(results, params, axs[0, i])
        plot_state_space(results, params, axs[1, i])

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()


if __name__ == '__main__':
    main()
