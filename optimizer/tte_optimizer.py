"""  Numerically solving for the two-trader equilibrium
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from typing import Any, List
import cost_function_approx as ca
import trading_funcs as tf
import fourier as fr

lambd = 6
kappa = 1
gamma = 1

DEFAULT_N = 10
TOL_COEFFS = 1e-1
TOL_COSTS = TOL_COEFFS
N_PLOT_POINTS = 100

# Which trader we are solving for
TRADER_A, TRADER_B = range(2)


class State:

    def __init__(self, a_coeff: np.ndarray | None = None,
                 b_coeff: np.ndarray | None = None,
                 n: int | None = None, calc_costs: bool = True):
        if a_coeff is None and b_coeff is None:
            self.n = n if n is not None else DEFAULT_N
            self.a_coeff = np.zeros(self.n)
            self.b_coeff = np.zeros(self.n)
        else:
            n_a = len(a_coeff) if a_coeff is not None else 0
            n_b = len(b_coeff) if b_coeff is not None else 0
            self.n = max(n_a, n_b)

            self.a_coeff = a_coeff if a_coeff is not None else np.zeros(self.n)
            self.b_coeff = b_coeff if b_coeff is not None else np.zeros(self.n)

        assert len(self.a_coeff) == len(self.b_coeff), (
            f"Fourier coeff array length mismatch: len(a)={len(self.a_coeff)},",
            f"len(b)={len(self.b_coeff)}")

        if not calc_costs:
            self.a_cost = None
            self.b_cost = None
        else:
            self.a_cost = ca.cost_fn_a_approx(self.a_coeff, self.b_coeff, kappa, lambd)
            self.b_cost = ca.cost_fn_b_approx(self.a_coeff, self.b_coeff, kappa, lambd)

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
        s = ""
        s += f"a_coeff = {self.a_coeff}\n"
        s += f"b_coeff = {self.b_coeff}\n"
        if self.a_cost is not None:
            s += f"cost(a) = {self.a_cost:.4f},\t"
        else:
            s += f"cost(a) = None,\t"
        if self.b_cost is not None:
            s += f"cost(b) = {self.b_cost:.4f}"
        else:
            s += f"cost(b) = None"
        return s

    def update(self, solve: int) -> "State":
        """ Solve for the best response of one trader with respect to the other """

        if solve == TRADER_A:

            def cost_function(x, b_coeffs):
                return ca.cost_fn_a_approx(x, b_coeffs, kappa, lambd)

            init_guess = self.a_coeff
            init_cost = cost_function(init_guess) if self.a_cost is None else self.a_cost
            args = (self.b_coeff,)
        else:
            def cost_function(x, a_coeffs):
                return ca.cost_fn_b_approx(a_coeffs, x, kappa, lambd)

            init_guess = self.b_coeff
            init_cost = cost_function(init_guess) if self.b_cost is None else self.b_cost
            args = (self.a_coeff,)

        # Minimize the cost function
        start = time.time()
        result = minimize(cost_function, init_guess, args=(args))
        print(f"optimization time = {(time.time() - start):.4f}s")

        # Generate a new updated state
        if solve == TRADER_A:
            res_state = State(result.x, self.b_coeff, calc_costs=False)
            res_state.a_cost = cost_function(result.x, self.b_coeff)
            res_state.b_cost = self.b_cost
        else:
            res_state = State(self.a_coeff, result.x, calc_costs=False)
            res_state.a_cost = self.a_cost
            res_state.b_cost = cost_function(self.a_coeff, result.x)

        return res_state

    def check_v_theo(self, lambd: float, kappa: float, ax: Any) -> dict:
        """  Check the stolution against theoretical
        :param lambd:  scale of trader B
        :param kappa:  permanent impact
        :return: statistics: function values, norms of differences
        """

        # tf.equil_2trader() for t in not t_sample()..
        # calculate values using fr.reconstruct
        # compare and save

        t_values = np.linspace(0, 1, N_PLOT_POINTS)

        params = {'kappa': kappa, 'lambd': lambd, 'trader_a': True}
        a_theo = [tf.equil_2trader(t, params) for t in t_values]

        params['trader_a'] = False
        b_theo = [tf.equil_2trader(t, params) for t in t_values]

        a_approx = [fr.reconstruct_from_sin(t, self.a_coeff) + gamma * t for t in t_values]
        b_approx = [fr.reconstruct_from_sin(t, self.b_coeff) + gamma * t for t in t_values]

        # ax.plot(t_values, a_theo, s=20, label="a theo", color="green", linestyle="--", linewidth=3)
        # ax.plot(t_values, b_theo, s=20, label="b theo", color="skyblue", linestyle="--", linewidth=3)
        ax.scatter(t_values, a_theo, s=20, label="a theo", color="red")
        ax.scatter(t_values, b_theo, s=20, label="b theo", color="tomato")
        ax.plot(t_values, a_approx, label="a approx", color="green", linestyle="-")
        ax.plot(t_values, b_approx, label="b approx", color="blue", linestyle="-")
        ax.set_title(r"Theoretical and Approximated Trading Schedules")
        ax.legend()
        ax.set_xlabel('t')
        ax.set_ylabel('a(t), b(t)')
        ax.grid()

    @staticmethod
    def plot_convergence(iter_hist: List["State"], ax: Any) -> None:

        a_costs = [i.a_cost for i in iter_hist]
        b_costs = [i.b_cost for i in iter_hist]

        # Plot the points as circles
        ax.scatter(a_costs[1:-1], b_costs[1:-1], color='darkblue', s=20, label='(a,b) costs', alpha=0.4)

        # Connect the points with lines
        ax.plot(a_costs, b_costs, color='darkblue', linestyle='-', linewidth=1, alpha=0.4)
        # for i in range(len(a_costs) - 1):
        #     plt.arrow(a_costs[i], b_costs[i], a_costs[i + 1] - a_costs[i],
        #               b_costs[i + 1] - b_costs[i],
        #               head_width=0.03, head_length=0.04, fc='red', ec='red')

        # Highlight the starting and ending points
        plt.scatter(a_costs[0], b_costs[0], color='green', s=70, label='init guess')
        plt.scatter(a_costs[-1], b_costs[-1], color='red', s=70, label='result')

        # Label the starting and ending points
        plt.text(a_costs[0], b_costs[0], 'init guess', fontsize=12, ha='right',
                 color='black', weight='bold')
        plt.text(a_costs[-1], b_costs[-1], 'result', fontsize=12, ha='right',
                 color='black', weight='bold')

        ax.set_xlabel('Trader A cost')
        ax.set_ylabel('Trader B cost')
        ax.set_title('Trading Cost Convergence to Equilibrium - State Space Diagram')
        plt.legend()
        # ax.grid()


if __name__ == "__main__":
    # Initialize a,b via the state class
    state = State()
    print("Initial State")
    print(state)

    # Start with no constraints
    iter_hist = [state]

    while True:
        print("\nStarting iteration:")
        state_a = state.update(solve=TRADER_A)
        print("New state A:")
        print(str(state_a))
        state = state_a.update(solve=TRADER_B)
        print("New State :")
        print(str(state))
        iter_hist.append(state_a)
        iter_hist.append(state)
        if iter_hist[-1].within_tol(iter_hist[-3]):
            break

    print(f"\n-------------------------------\n"
          f"Converged after {len(iter_hist) //2 } iterations:")
    print("\nFinal State:")
    print(state)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle("Two-Trader Equilibrium Strategies\n" +
                 r"$\kappa$" + f"={kappa}, " + r"$\lambda$" + f"={lambd}", fontsize=16)
    state.check_v_theo(lambd, kappa, axs[0])
    state.plot_convergence(iter_hist, axs[1])

    plt.tight_layout(rect=(0., 0.01, 1., 0.97))
    plt.show()
