"""
Check that the iterations in the equilibrium solver are optimal
"""
import numpy as np
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..', 'optimizer_qp')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..', 'lin_propagator')))
import fourier as fr
from propagator import cost_fn_prop_a_approx, cost_fn_prop_b_approx, prop_price_impact_approx
from tte_optimizer_prop import State, TRADER_A, TRADER_B
from opt_cost_prop import CostFunction
import trading_funcs as tf

# Global Parameters
DATA_FILE = r"tte_sim_data_l5_r10_k10_n200_g0.8.pkl"
SIM_RESULTS_DIR = os.path.abspath(os.path.join(current_dir, '../..', 'results/sim_prop_results'))

# Test Parameters
TEST_ITER_NUM = 2
SOLVE = TRADER_A


def load_iter_data(data_file: str):
    full_path = os.path.join(SIM_RESULTS_DIR, data_file)
    with open(full_path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    data = load_iter_data(DATA_FILE)

    # Get the initial state
    s = data['iter_hist'][iter_idx := 2 * (TEST_ITER_NUM - 1) + SOLVE + 1]
    s_prev = data['iter_hist'][iter_idx-1]
    print(f"Loaded Iteration {TEST_ITER_NUM}, solving for for Trader A:\n", s)

    # Intialize the Cost Function class
    if SOLVE == TRADER_A:
        expected_opt_coeff = s.a_coeff
        c = CostFunction(rho=s.rho, lambd=s.lambd, N=s.N, b_coeffs=s.b_coeff)
        init_guess = s_prev.a_coeff
    else:
        expected_opt_coeff = s.b_coeff
        c = CostFunction(rho=s.rho, lambd=s.lambd, N=s.N, a_coeffs=s.a_coeff, b_coeffs=s.b_coeff,
                         use_trader_B_fn=True)
        init_guess = s_prev.b_coeff
    print("Cost function initialized:\n", c)

    # Solve for the optimal coefficients and check that they are consistent with what's in the data file
    opt_coeffs, res = c.solve_min_cost(init_guess=init_guess)
    opt_coeffs_adj = opt_coeffs * s.gamma + init_guess * (1 - s.gamma)

    print(f"Optimal Coefficients:\n{opt_coeffs_adj}")
    print("Optimization Result:\n", res)

    # Compare the optimal coefficients to the expected ones
    if np.allclose(opt_coeffs_adj, expected_opt_coeff, atol=1e-4):
        print("Optimal coefficients are close to expected.")
    else:
        print("Optimal coefficients are NOT close to expected")


if __name__ == "__main__":
    main()
