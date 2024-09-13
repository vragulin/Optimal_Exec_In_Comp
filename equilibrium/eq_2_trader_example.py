import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Define constants
λ = 5; κ = 5; σ = 0.5  # Increased value for effect


# Define the system of differential equations
def equations(t, y, ξa, ξb):
	a, a_prime, b, b_prime = y

	# Equations - equations from the paper modified to ensure that a'', b'' are only on the LHS
	a_double_prime = - (λ / 2) * ((κ / 3) * b_prime - (2 * ξa * σ ** 2) / (3 * λ) * a - (2 * κ) / (3 * λ) * a_prime + (
				4 * ξb * σ ** 2) / (3 * λ ** 2) * b + κ * b_prime) + ξa * σ ** 2 * a
	b_double_prime = (κ / 3) * b_prime - (2 * ξa * σ ** 2) / (3 * λ) * a - (2 * κ) / (3 * λ) * a_prime + (
				4 * ξb * σ ** 2) / (3 * λ ** 2) * b

	return np.vstack((a_prime, a_double_prime, b_prime, b_double_prime))


# Boundary conditions
def boundary_conditions(ya, yb):
	return np.array([ya[0], ya[2], yb[0] - 1, yb[2] - 1])


# Define the grid of ξa and ξb values, multiplied by 3
xi_values = [
	[(1.5, 1.5), (10, 10), (50, 50)],
	[(10, 1.5), (50, 1.5), (200, 1.5)],
	[(1.5, 10), (1.5, 50), (1.5, 200)]
]

# Generate the 3x3 grid of plots
nrows, ncols = len(xi_values), len(xi_values[0])
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

for i in range(nrows):
	for j in range(ncols):
		ξa, ξb = xi_values[i][j]

		# Initial guess for the solution
		t = np.linspace(0, 1, 500)
		y_init = np.zeros((4, t.size))
		y_init[[1,3]] = 1

		# Solve the boundary value problem
		sol = solve_bvp(lambda t, y: equations(t, y, ξa, ξb), boundary_conditions, t, y_init)

		# Plot the solutions
		axs[i,j].set_ylim(0,2.5)
		axs[i, j].plot(sol.x, sol.y[0], 'b', label='a(t)')
		axs[i, j].plot(sol.x, sol.y[2], 'r', label='b(t)')
		axs[i, j].legend()
		axs[i, j].set_title(f'ξa={ξa}, ξb={ξb}, σ={σ}')
		axs[i, j].set_xlabel('t')
		axs[i, j].set_ylabel('a(t), b(t)')
		axs[i, j].grid(True)

plt.tight_layout()
plt.show()
