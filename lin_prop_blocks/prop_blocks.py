""" OW propagator model with blocks
    V. Ragulin - 10/19/2024
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize
from typing import Callable

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cost_function')))
import fourier as fr

SCIPY, QP = range(2)


class SinesBlocks:
    def __init__(self, N: int, blocks: tuple[float, float] = (0.0, 0.0),
                 coeff: np.ndarray | None = None, lambd: float = 1.0):
        """ Initialize the SinesBlocks strategy
            Blocks, coeffs are defined for the unit strategy.  To get the total size
            we multiply the coefficients by lambd
        """
        self.N = N
        self.blocks = blocks
        self.coeff = np.zeros(N) if coeff is None else coeff
        self.mu = 1.0 - sum(self.blocks)
        self.lambd = lambd

    def __str__(self):
        return (f"SinesBlocks: N={self.N}, blocks:({self.blocks[0]:.4g}, {self.blocks[1]:.4g})\n"
                f"\tcoeffs={np.array2string(self.coeff, precision=4, separator=', ', suppress_small=True)}, "
                f"lambd={self.lambd:.4g}")

    def calc(self, t: float) -> float:
        """ Calculate the value of the function a(t) at time t """
        match t:
            case 0:
                return 0
            case 1:
                return 1
            case _:
                return fr.reconstruct_from_sin(t, self.coeff) + self.blocks[0] + t * self.mu

    def deriv(self, t: float, order: int = 1) -> float:
        """ Calculate the derivate of 'order' at time t
            Strictly speaking, the derivative is not defined at t=0 and t=1, but
            we can use the limit to calculate the derivative at these points
        """
        return fr.reconstruct_deriv_from_sin(t, self.coeff, order=order) + self.mu

    @classmethod
    def from_func(cls, func: Callable, **kwargs) -> "SinesBlocks":
        """ Generate a SinesBlocks strategy from a function
            It is assumed that the function is defined on the interval [0, 1]
            If a function takes parameters, they can be passed as kwargs with 'params' key
        """
        N = kwargs.get('N', 10)
        lambd = kwargs.get('lambd', 1.0)
        params = kwargs.get('params', {})
        blocks = (func(0, **params), 1 - func(1, **params))

        # Generate the sine coefficients
        # The standard function assumes that the continuous function goes from (0,0) to (1,1)
        # Rescale our function to fit this template
        mu = 1.0 - blocks[0] - blocks[1]

        def func_rescaled(t: float) -> float:
            return func(t, **params) - (blocks[0] + mu * t)

        coeff = fr.sin_coeff(func_rescaled, N)
        return cls(N, blocks=blocks, coeff=coeff, lambd=lambd)


class CostModel:
    A, B = range(2)

    def __init__(self, strats: list[SinesBlocks | None], rho: float):
        self._strats = strats
        self.rho = rho

        self._mkt = None
        self._d_n = None
        self._Krl = None
        self._N = None

    @property
    def strats(self) -> list[SinesBlocks | None]:
        return self._strats

    @strats.setter
    def strats(self, value: list[SinesBlocks | None]) -> None:
        self._strats = value
        self._mkt = None
        self._d_n = None
        self._Krl = None

    @property
    def mkt(self) -> SinesBlocks:
        if self._mkt is None:
            blocks = (0.0, 0.0)
            coeffs = np.zeros(self.strats[0].N)
            lambd = 0.0
            for t in self.strats:
                blocks = (blocks[0] + t.blocks[0] * t.lambd, blocks[1] + t.blocks[1] * t.lambd)
                coeffs += t.coeff * t.lambd
                lambd += t.lambd
            blocks = (blocks[0] / lambd, blocks[1] / lambd)
            coeffs /= lambd
            self._mkt = SinesBlocks(self.strats[0].N, blocks=blocks, coeff=coeffs, lambd=lambd)
        return self._mkt

    def __str__(self):
        s = f"CostFunction: rho={self.rho}\n"
        for i, t in enumerate(["A", "B"]):
            s += f"Trader {t}: {str(self.strats[i])}\n"
        s += f"Market: {str(self.mkt)}"
        return s

    @property
    def N(self) -> int:
        if self._N is None:
            self._N = len(self.mkt.coeff)
        return self._N

    @property
    def d_n(self) -> np.ndarray:
        if self._d_n is None:
            n = np.arange(1, self.N + 1)
            self._d_n = (self.mkt.coeff * n * np.pi) / (self.rho ** 2 + (n * np.pi) ** 2)
        return self._d_n

    @property
    def Krl(self) -> float:
        if self._Krl is None:
            exp, pi, rho, m = np.exp, np.pi, self.rho, self.mkt
            self._Krl = m.blocks[0] - m.mu / rho - rho * np.sum(self.d_n)
        return self._Krl

    def update_strat(self, **kwargs) -> None:
        """ Update the strategy of a trader """
        A, B = self.A, self.B
        trader = kwargs.get('trader', A)
        blocks = kwargs.get('blocks', self.strats[trader].blocks)
        coeff = kwargs.get('coeff', self.strats[trader].coeff)
        lambd = kwargs.get('lambd', self.strats[trader].lambd)
        new_strat = SinesBlocks(self.N, blocks=blocks, coeff=coeff, lambd=lambd)
        if trader == A:
            new_strats = [new_strat, self.strats[B]]
        else:
            new_strats = [self.strats[A], new_strat]
        self.strats = new_strats

    def displacement_blocks(self, t: float) -> float:
        """ The displacement of the blocks only.  Used for testing"""
        exp, m, rho = np.exp, self.mkt, self.rho
        dp = (m.blocks[0] * exp(-rho * t) + m.mu * (1 - exp(-rho * t)) / rho) * m.lambd
        return dp

    def price(self, t: float) -> float:
        """ Calculate the price impact at time t """

        exp, pi, rho = np.exp, np.pi, self.rho
        m = self.mkt

        Crl = m.mu / rho + exp(-rho * t) * self.Krl
        n = np.arange(1, self.N + 1)
        P_unit = Crl + self.d_n @ (rho * np.cos(n * pi * t) + n * pi * np.sin(n * pi * t))
        return P_unit * self.mkt.lambd

    def cost_trader(self, trader: int = A, verbose: bool = False, **kwargs) -> float:
        """ Execution cost for trader A"""

        pi, exp, mkt, rho = np.pi, np.exp, self.mkt, self.rho
        N, d_n = self.N, self.d_n
        trd = self.strats[trader]

        if trd.lambd == 0:  # If the trader is not active, return 0
            return 0

        # Constants (later can be precomputed)
        n = np.arange(1, N + 1)
        n_sq = n ** 2
        n_odd = n % 2
        neg_one_to_n = np.ones(n.shape) - 2 * n_odd
        m_p_n_odd = (n[:, None] + n[None, :]) % 2
        msq_nsq = n_sq[:, None] - n_sq[None, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.where(m_p_n_odd, n_sq[:, None] / msq_nsq, 0)

        # Cost of blocks
        L_blocks = 0.5 * mkt.blocks[0] * trd.blocks[0] \
                   + (self.price(1) / mkt.lambd + 0.5 * mkt.blocks[1]) * trd.blocks[1]
        # Term 1
        t1 = trd.mu * (1 - exp(-rho)) / rho * self.Krl

        # Term 2
        t2 = trd.mu * 2 * np.sum(d_n * n_odd)

        # Term 3
        t3 = trd.mu * mkt.mu / rho

        # Term 4
        t4 = self.Krl * pi * rho * np.sum((trd.coeff * n) / (rho ** 2 + (n * pi) ** 2)
                                          * (1 - neg_one_to_n * exp(-rho)))

        # Term 5
        t5_1 = 0.5 * pi * rho * np.sum(trd.coeff * n * d_n)
        t5_2 = 2 * pi * np.sum(d_n[:, None] * (trd.coeff * n)[None, :] * M)
        t5 = t5_1 + t5_2

        # Term 6 = 0

        L_unit = (L_blocks + t1 + t2 + t3 + t4 + t5)
        L = L_unit * trd.lambd * mkt.lambd

        if verbose:
            print(f"Blocks:\t{L_blocks}")
            print(f"Term 1:\t{t1}")
            print(f"Term 2:\t{t2}")
            print(f"Term 3:\t{t3}")

            print(f"Term 4:\t{t4}")
            print(f"Term 5:\t{t5},\tterms 5_1:\t{t5_1},\tterms 5_2:\t{t5_2}")
            print(f"Unit Cost:\t{L_unit}")
            print(f"Total Cost:\t{L}")
        return L

    def cost_trader_integral(self, trader: int =A, verbose: bool = False, **kwargs) -> float:
        """ Calculate the cost of a trader.  Sams as cost-trader, but evaluate the integral using
            quad
        """
        from scipy.integrate import quad

        trd, mkt = self.strats[trader], self.mkt

        # Direct cost of the blocks
        L_block = 0.5 * (trd.blocks[0] * mkt.blocks[0] + trd.blocks[1] * mkt.blocks[1]
                         ) * mkt.lambd + self.price(1) * trd.blocks[1]

        def integrand(t):
            return self.price(t) * trd.deriv(t)

        L_int, _ = quad(integrand, 0, 1)

        L = (L_block + L_int) * trd.lambd

        return L

    def solve_min_cost(self, trader: int = A, **kwargs) -> tuple:
        """  Solve for the strategy that minimizes the cost of a trader
        """
        solver = kwargs.get('solver', SCIPY)
        if solver == SCIPY:
            print("Using SCIPY solver")
            x = np.zeros(self.N + 2)
            x[:2] = np.array(self.strats[trader].blocks)
            x[2:] = self.strats[trader].coeff

            def obj_func(x):
                blocks = tuple(x[:2])
                coeff = x[2:]
                self.update_strat(trader=trader, blocks=blocks, coeff=coeff)
                return self.cost_trader(trader)

            res = minimize(obj_func, x)
            opt_blocks = tuple(res.x[:2])
            opt_coeffs = res.x[2:]
            opt_strat = SinesBlocks(self.N, blocks=opt_blocks, coeff=opt_coeffs,
                                    lambd=self.strats[trader].lambd)
            return opt_strat, res
        else:
            raise NotImplementedError("Only SCIPY solver is implemented")