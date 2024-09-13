# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html#r25f8479e577a-2
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


def fun(x, y, p):
	k = p[0]
	return np.vstack((y[1], -k ** 2 * y[0]))


def bc(ya, yb, p):
	k = p[0]
	return np.array([ya[0], yb[0], ya[1] - k])


if __name__ == "__main__":
	x = np.linspace(0, 1, 5)
	y = np.zeros((2, x.size))
	y[0, 1] = 1
	y[0, 3] = -1

	sol = solve_bvp(fun, bc, x, y, p=[6])

	x_plot = np.linspace(0, 1, 100)
	y_plot = sol.sol(x_plot)[0]
	plt.plot(x_plot, y_plot)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()