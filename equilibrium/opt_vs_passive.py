import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Define constants
λ = 5; κ = 5; σ = 0.5  # Increased value for effect


# Define the exogenous function b(t)
def b_exogenous(t):
	""" Return a tuple: b, b', b''
		Function with conditions b(0) = 0 and b(1) = 1
	"""
	return t ** 4, 4*t**3, 12*t**2


# Define the system of differential equations with exogenous b(t)
def equations(t, y, ξa):
	a, a_prime = y
	b, b_prime, b_dbl_prime = b_exogenous(t)
	a_double_prime = -(λ / 2) * (b_dbl_prime + κ * b_prime) + ξa * σ ** 2 * a
	return np.vstack((a_prime, a_double_prime))


# Boundary conditionss
def boundary_conditions(ya, yb):
	return np.array([ya[0], yb[0] - 1])


# Define the grid of ξa values, multiplied by 3
xi_values = [1.5, 10, 50, 200]

# Generate the plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for idx, ξa in enumerate(xi_values):
	i, j = divmod(idx, 2)

	# Initial guess for the solution
	t = np.linspace(0, 1, 100)
	y_init = np.zeros((2, t.size))

	# Solve the boundary value problem
	sol = solve_bvp(lambda t, y: equations(t, y, ξa), boundary_conditions, t, y_init)

	# Plot the solutions
	axs[i, j].plot(sol.x, sol.y[0], 'b', label='a(t)')
	axs[i, j].plot(sol.x, b_exogenous(sol.x)[0], 'r', label='b(t) (exogenous)')
	axs[i, j].legend()
	axs[i, j].set_title(f'ξa={ξa}, σ={σ}')
	axs[i, j].set_xlabel('t')
	axs[i, j].set_ylabel('a(t), b(t)')
	axs[i, j].grid(True)

plt.tight_layout()
plt.show()