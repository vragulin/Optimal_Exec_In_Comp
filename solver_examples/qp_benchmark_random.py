# https://scaron.info/blog/quadratic-programming-in-python.html
import cvxopt
from qpsolvers import solve_qp
import numpy as np
import time
from scipy.linalg import toeplitz
from typing import Any


def gen_random_inputs(n: int) -> dict:
	M, b = np.random.random((n, n)), np.random.random(n)
	P, q = np.dot(M.T, M), np.dot(b, M).reshape((n,))
	G = toeplitz(
		[1., 0., 0.] + [0.] * (n - 3),
		[1., 2., 3.] + [0.] * (n - 3)
	)
	h = np.ones(n)
	return {'P': P, 'q': q, 'G': G, 'h': h}


def solve_random_qp(n: int, solver: str, inp: dict = None) -> Any:
	if dict is None:
		inp = gen_random_inputs(n)
	P, q, G, h = inp['P'], inp['q'], inp['G'], inp['h']
	return solve_qp(P, q, G, h, solver=solver)


if __name__ == "__main__":
	solvers = ['cvxopt', 'daqp', 'piqp', 'quadprog']  #ToDo add osqp later
	N = 500
	np.random.seed(123)
	inp = gen_random_inputs(N)
	for solver in solvers:
		print(f"\nRunning {solver}")
		start_time = time.time()
		res = solve_random_qp(N, solver, inp)
		end_time = time.time()
		print(f"{solver} time taken: {end_time - start_time:.4f} seconds")
		print(f"Optimization {solver}: ")
		print(res[:5])
