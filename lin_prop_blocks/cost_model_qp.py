""" Estimate coefficient matrices needed to represent the Loss function L(a_bar, b_bar) as a
    quadratic form
    V. Ragulin, 10/23/2024
"""
import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.abspath(os.path.join(CURRENT_DIR, '..', 'matrix_utils'))
])
from prop_blocks import SinesBlocks, CostModel
from est_quad_coeffs import find_coefficients

# Aliases
A, B = CostModel.A, CostModel.B  # Traders


class CostModelQP(CostModel):

    def __init__(self, strats: list[SinesBlocks | None], rho: float, qp_coeffs: list[dict] | None = None,
                 reg_params: dict | None = None):
        super().__init__(strats, rho)
        self._qp_coeffs = None if qp_coeffs is None else qp_coeffs
        self.wiggle = 0 if reg_params is None else reg_params.get('wiggle', 0.0)
        self.wiggle_exp = 4 if reg_params is None else reg_params.get('wiggle_exp', 4)

    @property
    def qp_coeffs(self) -> list[dict]:
        if self._qp_coeffs is None:
            self._qp_coeffs = []
            for trader, s in enumerate(self.strats):
                self._qp_coeffs.append(self.est_trader_coeffs(trader=trader)) if s is not None else {}
        return self._qp_coeffs

    @qp_coeffs.setter
    def qp_coeffs(self, value: list[dict]):
        if not isinstance(value, list):
            raise ValueError("qp_coeffs must be a list of dictionaries")
        self._qp_coeffs = value

    def est_trader_coeffs(self, trader: int = A) -> dict:
        """ Estimate the trader cost function as bilinear form
        """
        # Find the quadratic coefficients
        n_stacked = (self.N + 2) * len(self.strats)

        def Loss(x: np.ndarray) -> float:
            """ The vector x contains all the coefficents for both traders in the orders:
            a_s, a_e, {a_n}, b_s, b_e, {b_n}
            """
            N, A, B = self.N, self.A, self.B
            blocks_a = (float(x[0]), float(x[1]))
            coeff_a = np.array(x[2:2 + N])
            blocks_b = (float(x[2 + N]), float(x[3 + N]))
            coeff_b = np.array(x[4 + N:4 + 2 * N])

            strat_a = SinesBlocks(N=N, blocks=blocks_a, coeff=coeff_a, lambd=self.strats[A].lambd)
            strat_b = SinesBlocks(N=N, blocks=blocks_b, coeff=coeff_b, lambd=self.strats[B].lambd)
            c = CostModel(strats=[strat_a, strat_b], rho=self.rho)
            return c.cost_trader(trader=trader)

        H, f, C = find_coefficients(Loss, n_stacked)

        # Add regularization term to the Hessian
        n = np.arange(1, self.N + 1)
        wiggle_vec_half = np.concatenate([np.zeros(2), n ** self.wiggle_exp])
        wiggle_vec = np.concatenate([wiggle_vec_half * self.strats[A].lambd ** 2,
                                     wiggle_vec_half * self.strats[B].lambd ** 2])
        smooth_adj = self.wiggle * np.diag(wiggle_vec)  # roll the pi **4 multiplier into 'wiggle'

        return {"H": H + smooth_adj, "f": f, "C": C, "W": smooth_adj}

    def cost_trader_matrix(self, trader: int = A, verbose: bool = False) -> float:
        """ Compute the cost function for the trader using the matrix form
        """
        c = np.zeros(2 * self.N + 4)
        c[0:2] = self.strats[A].blocks
        c[2:2 + self.N] = self.strats[A].coeff
        c[self.N + 2:self.N + 4] = self.strats[B].blocks
        c[4 + self.N:4 + 2 * self.N] = self.strats[B].coeff

        q = self.qp_coeffs[trader]
        H, f, C = q["H"], q["f"], q["C"]
        if verbose:
            print(q)
        return 0.5 * c @ H @ c + f @ c + C

    def _trader_FOC_matrices(self, trader: int = A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Extract the rows of the Hessian and the linear coefficients corresponding to the trader
        """
        N = self.N
        q = self.qp_coeffs[trader]
        H, f = q["H"], q["f"]
        trader_mask = np.zeros(2 * N + 4, dtype=bool)
        trader_mask[trader * (2 + N):(trader + 1) * (2 + N)] = True
        H_t = H[trader_mask][:, trader_mask]
        G = H[trader_mask][:, ~trader_mask]
        f_t = f[trader_mask]
        return H_t, G, f_t

    def solve_min_cost_qp(self, trader: int = A) -> tuple:
        """ Solve the quadratic program for the trader
            Math summary:  is s = (a,b) the stacked parameter vector, and H, f, C the quadratic
            coefficients.
            Here is how we solve for trader a (and for B is similar)
            Represent the H matrix as:
            H = [[H_a, G],
                 [G, 0]]
            likewise, the f = (f_a, f_b)
            The First Order conditions say that all gradient components of a must be zero
            So, we have:
            H_a @ a + G @ b + f_a = 0, so
            a_opt = -H_a^{-1} @ (G @ b + f_a)
        """

        H_t, G, f_t = self._trader_FOC_matrices(trader=trader)

        # Extract strategy params used by the other trader
        other = 1 - trader
        s_other = np.zeros(self.N + 2)
        s_other[:2] = self.strats[other].blocks
        s_other[2:] = self.strats[other].coeff

        # Solve for the optimal strategy
        s_trader = -np.linalg.solve(H_t, G @ s_other + f_t)

        # Prepare the output
        opt_strat = SinesBlocks(N=self.N, blocks=tuple(s_trader[:2]), coeff=s_trader[2:],
                                lambd=self.strats[trader].lambd)
        res = {'s_trader': s_trader, 'H_t': H_t, 'G': G, 'f_t': f_t}
        return opt_strat, res

    def wiggle_penalty(self, trader):
        W = self.qp_coeffs[trader]["W"]
        W_trader = W[2:2 + self.N, 2:2 + self.N]
        b_n = self.strats[trader].coeff
        return 0.5 * b_n @ W_trader @ b_n

    @classmethod
    def solve_equilibrium(cls, N: int, lambd: float, rho: float,
                          reg_params: dict | None = None) -> tuple:
        """ Solve the equilibrium for the cost function
        For each trader represent his cost function matrices:
        for function La:
        Ha = [[Ha_a, Ga],
             [Ga, 0]]
        fa = (fa_a, fa_b)

        Likewise for trader b.
        Then FOC for trader A and B resp. are:
        Ha @ a + Ga @ b + fa_a = 0
        Gb @ a + Hb @ b + fa_b = 0

        We can stack the matrices to form the combined linear system:
        H = [[Ha, Ga],
             [Gb, Hb]]
        f = [fa_a, fa_b]
        The solution is x = -H^{-1} @ f
        """

        # Set up the environment
        strats = [SinesBlocks(N=N), SinesBlocks(N=N, lambd=lambd)]
        c = cls(strats, rho=rho, reg_params=reg_params)

        # Calculate the first order condition terms
        H_a, G_a, f_a = c._trader_FOC_matrices(trader=A)
        H_b, G_b, f_b = c._trader_FOC_matrices(trader=B)

        # Stack the matrices to form the combined linear system
        H = np.block([[H_a, G_a],
                      [G_b, H_b]])
        f = np.block([f_a, f_b])

        # Solve for the stacked coefficients vector
        x = np.linalg.solve(H, -f)

        strats = []
        for trd, i in enumerate([A, B]):
            params = x[i * (N + 2):(i + 1) * (N + 2)]
            trd_size = 1 if trd == A else lambd
            strats.append(SinesBlocks(N=N, blocks=tuple(params[:2]), coeff=params[2:], lambd=trd_size))

        return strats, {'H_a': H_a, 'G_a': G_a, 'f_a': f_a, 'H_b': H_b, 'G_b': G_b, 'f_b': f_b, 'x': x}

    @classmethod
    def solve_equilibrium_blocks(cls, N: int, lambd: float, rho: float,
                                 reg_params: dict | None = None) -> tuple:
        """ Solve the equilibrium for the cost function assuming we only trade blocks
        For each trader represent his cost function matrices:
        for function La:
        Ha = [[Ha_a, Ga],
             [Ga, 0]]
        fa = (fa_a, fa_b)

        Likewise for trader b.
        Then FOC for trader A and B resp. are:
        Ha @ a + Ga @ b + fa_a = 0
        Gb @ a + Hb @ b + fa_b = 0

        From all matrices only take the top 2 rows. Then proceed as with the fulll solve,
        but we only have 4 variables to solve for.

        We can stack the matrices to form the combined linear system:
        H = [[Ha, Ga],
             [Gb, Hb]]
        f = [fa_a, fa_b]
        The solution is x = -H^{-1} @ f

        Then pad the solution for the blocks with zeros for the coefficients
        """

        # Set up the environment
        strats = [SinesBlocks(N=N), SinesBlocks(N=N, lambd=lambd)]
        c = cls(strats, rho=rho, reg_params=reg_params)

        # Calculate the first order condition terms
        H_a, G_a, f_a = c._trader_FOC_matrices(trader=A)
        H_b, G_b, f_b = c._trader_FOC_matrices(trader=B)

        # Stack the matrices to form the combined linear system
        H = np.block([[H_a[:2, :2], G_a[:2, :2]],
                      [G_b[:2, :2], H_b[:2, :2]]])
        f = np.block([f_a[:2], f_b[:2]])

        # Solve for the stacked coefficients vector
        x = np.linalg.solve(H, -f)

        strats = []
        for trd, i in enumerate([A, B]):
            params = x[i * 2:(i + 1) * 2]
            trd_size = 1 if trd == A else lambd
            strats.append(SinesBlocks(N=N, blocks=tuple(params), lambd=trd_size))

        return strats, {'H_a': H_a, 'G_a': G_a, 'f_a': f_a, 'H_b': H_b, 'G_b': G_b, 'f_b': f_b, 'x': x}
