# QP Examples
# https://towardsdatascience.com/quadratic-optimization-with-constraints-in-python-using-cvxopt-fc924054a9fc
# https://stackoverflow.com/questions/32543475/how-python-cvxopt-solvers-qp-basically-works
# https://scaron.info/blog/quadratic-programming-in-python.html
# https://github.com/qpsolvers/qpsolvers
# https://github.com/qpsolvers/qpsolvers?tab=readme-ov-file
import cvxopt
import quadprog
import numpy as np
import time


def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
	P = .5 * (P + P.T)  # make sure P is symmetric
	args = [cvxopt.matrix(P), cvxopt.matrix(q)]
	args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
	if A is not None:
		args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
	sol = cvxopt.solvers.qp(*args)
	if 'optimal' not in sol['status']:
		return None
	return np.array(sol['x']).reshape((P.shape[1],))


def quadprog_solve_qp(P, q, G, h, A=None, b=None):
	qp_G = .5 * (P + P.T)  # make sure P is symmetric
	qp_a = -q
	if A is not None:
		qp_C = -np.vstack([A, G]).T
		qp_b = -np.hstack([b, h])
		meq = A.shape[0]
	else:  # no equality constraint
		qp_C = -G.T
		qp_b = -h
		meq = 0
	return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)
q = -np.dot(M.T, np.array([3., 2., 3.]))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.]).reshape((3,))

# Solve with cvxopt
start_time = time.time()
res_cvx = cvxopt_solve_qp(P, q, G, h)
end_time= time.time()
print(f"CvxOpt time taken: {end_time - start_time:.4f} seconds")
print("Optimization cvxopt: ")
print(res_cvx)

# Solve with quadrog
start_time = time.time()
res_qp = quadprog_solve_qp(P, q, G, h)
end_time= time.time()
print(f"QuadProg time taken: {end_time - start_time:.4f} seconds")
print("Optimization quadprog: ")
print(res_qp)