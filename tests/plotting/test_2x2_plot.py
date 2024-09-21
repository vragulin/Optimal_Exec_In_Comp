import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.linspace(0, 1, 100)
y1 = x
y2 = x**2
y3 = x**2 - 0.5
y4 = -x**2 + 0.5
y5 = -x

# Create subplots
fig, axs = plt.subplots(4, 2, figsize=(10, 8))

# Subplot [0, 0]
axs[0:2, 0].plot(x, y1, 'r-')
axs[0:2, 0].set_title('y = x')

# Subplot [1, 0] - top half
axs[2:, 0].plot(x, y3, 'b-')
axs[2:, 0].set_title('y = x^2 - 0.5')

# Subplot [0, 1]
axs[0, 1].plot(x, y2, 'g-')
axs[0, 1].set_title('y = x^2')


# Subplot [2, 0] - bottom half
axs[1, 1].plot(x, y4, 'm-')
axs[1, 1].set_title('y = -x^2 + 0.5')

# Subplot [1, 1]
axs[2:, 1].plot(x, y5, 'c-')
axs[2:, 1].set_title('y = -x')

# Adjust layout
plt.tight_layout()
plt.show()