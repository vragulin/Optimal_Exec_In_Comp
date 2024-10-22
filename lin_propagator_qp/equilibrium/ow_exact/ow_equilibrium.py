"""  The exact cost function for the Obizhaeva-Wang (2013) model
    V. Ragulin - 12/08/2024
"""
import numpy as np
import matplotlib.pyplot as plt


class OW:
    N_POINTS_TO_PLOT = 100

    def __init__(self, rho: float, lambd: float, **kwargs):
        self.rho = rho
        self.lambd = lambd
        self.a = np.zeros(2) if 'a' not in kwargs else np.array(kwargs['a'])
        self.b = np.zeros(2) if 'b' not in kwargs else np.array(kwargs['b'])
        self.n_to_plot = 100 if 'n_to_plot' not in kwargs else kwargs['n_to_plot']
        self.c = np.concatenate((self.a, self.b))
        self._cost_a = None
        self._cost_b = None

        self._tv = self.a + self.lambd * self.b  # Total volume
        self._mu_a = 1 - np.sum(self.a)
        self._mu_b = 1 - np.sum(self.b)
        self._mu = self._mu_a + self.lambd * self._mu_b
        self._J = (1 - np.exp(-self.rho)) / self.rho
        self._K = (self.rho + np.exp(-self.rho) - 1) / self.rho ** 2

    def __str__(self):
        s = f"OW Model: rho={self.rho}, \t lambd={self.lambd}"
        s += f"a: {self.a}, \tb: {self.b}"
        s += f"\nCost A: {self.cost_a}, \t Cost B: {self.cost_b}"
        return s

    @property
    def cost_a(self) -> float:
        if self._cost_a is None:
            self._cost_a = self._cost_fn_a()
        return self._cost_a

    @property
    def cost_b(self) -> float:
        if self._cost_b is None:
            self._cost_b = self._cost_fn_b()
        return self._cost_b

    def _cost_fn_a(self) -> float:
        term1 = 0.5 * self._tv.T @ self.a
        term2 = self._mu_a * (self._tv[0] * self._J + self._mu * self._K)
        term3 = (self._tv[0] * np.exp(-self.rho) + self._mu * self._J) * self.a[1]
        return term1 + term2 + term3

    def _cost_fn_b(self) -> float:
        term1 = 0.5 * self._tv.T @ self.b
        term2 = self._mu_b * (self._tv[0] * self._J + self._mu * self._K)
        term3 = (self._tv[0] * np.exp(-self.rho) + self._mu * self._J) * self.b[1]
        return self.lambd * (term1 + term2 + term3)

    def grad_a_matrix(self) -> np.ndarray:
        """ Do the calculation in vector form for dCa/da0 and dCa/da1 """
        # Name LHS variables by variable in the stocked c-vector that
        # they correspond to

        # dCa/da0
        a0_0 = 1 + 2 * (self._K - self._J)
        a1_0 = np.exp(-self.rho) + 2 * (self._K - self._J)
        b0_0 = (0.5 + self._K - self._J) * self.lambd
        b1_0 = self._K * self.lambd
        k0 = -(1 + self.lambd) * self._K - (self._K - self._J)

        # dCa/da1
        a0_1 = np.exp(-self.rho) + 2 * (self._K - self._J)
        a1_1 = 1 + 2 * (self._K - self._J)
        b0_1 = (np.exp(-self.rho) + self._K - 2 * self._J) * self.lambd
        b1_1 = (0.5 + self._K - self._J) * self.lambd
        k1 = - (1 + self.lambd) * (self._K - self._J) - self._K

        return np.array([[a0_0, a0_1], [a1_0, a1_1],
                         [b0_0, b0_1], [b1_0, b1_1],
                         [k0, k1]])

    def grad_b_matrix(self) -> np.ndarray:
        """ Do the calculation in vector form for dCb/db0 and dCb/db1"""
        # Use the symmetry relationship that Ca(a, b, lambd) = lambd^2 * Cb(b, a, 1/lambd)

        sym_ow = OW(self.rho, 1 / self.lambd, a=self.b, b=self.a)
        vectors = self.lambd ** 2 * sym_ow.grad_a_matrix()
        return vectors[[2, 3, 0, 1, 4]]  # Switch back a and b coeffs

    def grad_a(self) -> np.ndarray:
        vectors = self.grad_a_matrix()
        return vectors.T @ np.append(self.c, 1)

    def grad_b(self) -> np.ndarray:
        vectors = self.grad_b_matrix()
        return vectors.T @ np.append(self.c, 1)

    def solve_nash(self) -> np.ndarray:
        grad_vec_a = self.grad_a_matrix()
        grad_vec_b = self.grad_b_matrix()
        G = np.concatenate((grad_vec_a[:-1], grad_vec_b[:-1]), axis=1)
        m = np.concatenate((grad_vec_a[-1], grad_vec_b[-1]))
        return np.linalg.solve(G.T, -m)

    def ow_paper(self, t):
        """" The optimal size according to the Obizhaeva-Wang (2013) model
        """
        block = 1 / (2 + self.rho)
        continous_speed = self.rho / (2 + self.rho)
        match t:
            case 0:
                return 0
            case 1:
                return 1
            case _:
                return block + continous_speed * t

    @staticmethod
    def ow_strat(t: float, p: tuple) -> np.ndarray:
        match t:
            case 0:
                return 0
            case 1:
                return 1
            case _:
                return p[0] + (1 - p[0] - p[1]) * t

    def plot(self, title: str | None = None):
        t_sample = np.linspace(0, 1, self.n_to_plot + 1)
        a_curve = np.array([self.ow_strat(t, self.a) for t in t_sample])
        b_curve = np.array([self.ow_strat(t, self.b) for t in t_sample])
        ow_curve = np.array([self.ow_paper(t) for t in t_sample])

        plot_title = title if title is not None \
            else f"Optimal Trading Strategies, OW Model\n"
        plt.plot(t_sample, a_curve, label='a(t)', color='red')
        plt.plot(t_sample, b_curve, label=r'$b_\lambda(t)$', color='blue', linestyle='dashed')
        plt.plot(t_sample, t_sample, label='t', color='grey', linestyle='dotted')
        plt.plot(t_sample, ow_curve, label='OW(t)', color='green', linestyle='dotted')
        plt.title(plot_title)
        plt.xlabel('t')
        plt.ylabel('amount traded')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    lambd = 10
    rho = 10
    m = OW(rho=rho, lambd=lambd)

    # Check: print derivatives at the equilibrium
    eq = m.solve_nash()
    print("Model Parameters: λ={}, ρ={}".format(lambd, rho))
    print("Equilibrium parameters: a0={:.4f}, a1={:.4f}, b0={:.4f}, b1={:.4f}".format(*eq))

    print("\nChecking the gradients at the equilibrium:")
    print('Gradient C_a', n.grad_a())
    print('Gradient C_b', n.grad_b())
    n.plot(title=f"Equilibrium Trading Strategies, OW Model"
                 f"\nλ={lambd}, ρ={rho}")
