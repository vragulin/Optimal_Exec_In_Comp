# Class to store and process strategy information
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'cost_function')))
import fourier as fr
from cost_function_approx import cost_fn_a_approx


class EncodedStrategy:

    def __init__(self, sin_coeffs: np.ndarray | None = None,
                 lambd: float = 1, kappa: float = 0, n_terms: int = 10,
                 gamma: float = 1):
        self.sin_coeffs = sin_coeffs if sin_coeffs is not None else np.zeros(n_terms)
        self.n_terms = n_terms
        self.lambd = lambd
        self.kappa = kappa
        self.gamma = gamma

    def encode_power(self, p: float = 0) -> None:
        """  Encode strategy x^p
        """
        self.sin_coeffs = fr.sin_coeff(lambda x: x ** p - self.gamma * x, self.n_terms)

    def encode_parabolic(self, c: float = 0):
        """  Encode strategy x^p
        """
        self.sin_coeffs = fr.sin_coeff(lambda x: x * (x - c) / (1 - c) - self.gamma * x
                                       , self.n_terms)

    def encode_passive(self, sigma: float = 0):
        """  Encode the optimal non-competitive strat from Almgren-Chriss (2004)
        """
        self.sin_coeffs = fr.sin_coeff(
            lambda x: np.sinh(sigma * x) / np.sinh(sigma) - self.gamma * x
            , self.n_terms)

    def reconstruct(self, t, *args, **kwargs):
        """ Get the approximate value of the trajectory at time t
        """
        return fr.reconstruct_from_sin(t, self.sin_coeffs, *args, **kwargs) + self.gamma * t

    def best_response(self):
        """ Best reponse to a strategy
                Returns fourier coefficient and execution cost
            """

        def cost_function(t):
            return cost_fn_a_approx(t, self.sin_coeffs, self.kappa, self.lambd)

        init_guess = np.zeros(self.n_terms)
        result = minimize(cost_function, init_guess)

        return result.x, result

    @staticmethod
    def cost(a_strat: "EncodedStrategy", b_strat: "EncodedStrategy") -> float:
        return cost_fn_a_approx(
            a_strat.sin_coeffs, b_strat.sin_coeffs,
            b_strat.kappa, b_strat.lambd
        )

    def plot(self, scaled: bool = False, n_ticks: int = 100) -> tuple:
        t_values = np.linspace(0, 1, n_ticks)
        y_values = [fr.reconstruct_from_sin(t, self.sin_coeffs) + self.gamma * t for t in t_values]

        plt.figure(figsize=(10, 5))
        plt.plot(t_values, y_values)
        plt.title("Strategy Trading Trajecctory")
        plt.grid()
        plt.show()
        return t_values, y_values

# ToDo: wirte an exampel wheere we create several stratiges, encode them
#   arrange into a list, then calculate optimal response and cost for this list
#   and a probility-weighted cost
#   Also show how I can set up and print a database of "fingerprints" for clients
