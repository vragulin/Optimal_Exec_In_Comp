# Example of Bratu's problem from SciPy docs
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


def fun(x, y):
	return np.vstack((y[1], -np.exp(y[0])))


def bc(ya, yb):
	return np.array([ya[0], yb[0]])


if __name__ == "__main__":
	# Define initial mesh
	x = np.linspace(0, 1, 5)

	# define two initial quesses
	y_a = np.zeros((2, x.size))
	y_b = np.zeros((2, x.size))
	y_b[0] = 3

	# Solve
	res_a = solve_bvp(fun, bc, x, y_a)
	res_b = solve_bvp(fun, bc, x, y_b)

	# Plot
	x_plot = np.linspace(0, 1, 100)
	y_plot_a = res_a.sol(x_plot)[0]
	y_plot_b = res_b.sol(x_plot)[0]

	plt.plot(x_plot, y_plot_a, label='y_a')
	plt.plot(x_plot, y_plot_b, label='y_b')
	plt.legend()
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()
