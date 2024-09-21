"""  Numerically solving for the two-trader equilibrium
     Generally works ok, but does not always converge, e.g. lambd =1, kappa=20.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from typing import Any, List, Dict
import trading_funcs as tf
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'cost_function')))
import cost_function_approx as ca
import fourier as fr


# Parameters and Constants
LAMBD = 5
KAPPA = 1

DEFAULT_N = 10
TOL_COEFFS = 1e-2
TOL_COSTS = TOL_COEFFS
FRACTION_MOVE = 0.2
MAX_ITER = 250
MAX_ABS_COST = 1e10
N_PLOT_POINTS = 100
GAMMA = 1  # Put it at the end since it never changes

# Which trader we are solving for
TRADER_A, TRADER_B = range(2)


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

        a_theo = self._calculate_theoretical_values(t_values, kappa, lambd, trader_a=True)
        b_theo = self._calculate_theoretical_values(t_values, kappa, lambd, trader_a=False)

        a_approx = [fr.reconstruct_from_sin(t, self.a_coeff) + GAMMA * t for t in t_values]
        b_approx = [fr.reconstruct_from_sin(t, self.b_coeff) + GAMMA * t for t in t_values]

        self._plot_values(ax, t_values, a_theo, b_theo, a_approx, b_approx)

        return {
            "a_theo": a_theo,
            "b_theo": b_theo,
            "a_approx": a_approx,
            "b_approx": b_approx
        }

    @staticmethod
    def _calculate_theoretical_values(t_values, kappa, lambd, trader_a):
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
        ax.text(a_costs[0], b_costs[0], 'init guess', fontsize=12, ha='right',
                color='black', weight='bold')
        ax.text(a_costs[-1], b_costs[-1], 'solution', fontsize=12, ha='right',
                color='black', weight='bold')

        ax.set_xlabel('Trader A cost')
        ax.set_ylabel('Trader B cost')
        ax.set_title(f'Trading Cost Convergence to Equilibrium\n'
                     f'State Space Diagram: (x,y) = (cost A, cost B)')
        ax.legend()

    def plot_func_convergence(self, iter_hist: List["State"], res_coeffs: Dict[str, Any],
                               ax: Any) -> None:
        """ Plot the difference between theoretical schedule and the approximations
        """

        iter_step = 5


    def plot_state_space_v_theo(self, iter_hist: List["State"], res_coeffs: Dict[str, Any],
                                ax: Any) -> None:
        """ Plot eveolution of costs for A and B vs. Teoretical over iterations
        """
        ...

    def plot_results(self, iter_hist: List["State"]):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plt.suptitle("Two-Trader Equilibrium Strategies\n" +
                     r"$\kappa$" + f"={KAPPA}, " + r"$\lambda$" + f"={LAMBD}, " +
                     f"{len(self.a_coeff)} Fourier terms, "+
                     f"{len(iter_hist) // 2} iterations", fontsize=16)
        res_coeffs = self.check_v_theo(LAMBD, KAPPA, axs[0, 0])
        self.plot_state_space(iter_hist, axs[1, 0])
        # self.plot_func_convergence(iter_hist, res_coeffs, axs[0, 1])
        # self.plot_state_space_v_theo(iter_hist, res_coeffs, axs[1, 1])
        plt.tight_layout(rect=(0., 0.01, 1., 0.97))
        plt.show()


def main():
    # Initialize state
    state = State()
    print("Initial State")
    print(state)

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

    state.plot_results(iter_hist)


if __name__ == "__main__":
    main()
