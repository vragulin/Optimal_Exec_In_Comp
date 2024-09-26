# Random benchmark using the SLSQP solver
import numpy as np
from scipy.optimize import minimize
from qp_benchmark_random import gen_random_inputs
from typing import Any, Dict
import time


def objective(x: np.ndarray, P: np.ndarray, q: np.ndarray) -> float:
	return 0.5 * np.dot(x.T, np.dot(P, x)) + np.dot(q, x)


def constraint_ineq(x: np.ndarray, G: np.ndarray, h: np.ndarray) -> np.ndarray:
	return h - np.dot(G, x)


def solve_random_qp(n: int, solver: str = 'SLSQP', inp: Dict[str, np.ndarray] = None) -> Any:
	if inp is None:
		inp = gen_random_inputs(n)
	P, q, G, h = inp['P'], inp['q'], inp['G'], inp['h']

	x0 = np.zeros(n)  # Initial guess
	constraints = {'type': 'ineq', 'fun': constraint_ineq, 'args': (G, h)}

	result = minimize(objective, x0, args=(P, q), method=solver, constraints=constraints)

	return result


# Example usage
if __name__ == "__main__":
	N = 500
	np.random.seed(123)
	SOLVER = "SLSQP"
	print(f"\nRunning {SOLVER}")
	start_time = time.time()
	result = solve_random_qp(N)
	end_time = time.time()
	print(f"{SOLVER} time taken: {end_time - start_time:.4f} seconds")
	print(f"Optimization {SOLVER}: ")
	print(result.x[:5])
