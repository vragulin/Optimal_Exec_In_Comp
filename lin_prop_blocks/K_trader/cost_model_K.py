""" Class to model N-Trader cost functions
    V. Ragulin - 11/01/2024
"""
import os
import sys
from base64 import encode

import numpy as np
from scipy.optimize import minimize
from typing import Callable
from copy import deepcopy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.abspath(os.path.join(CURRENT_DIR, '..')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'cost_function')),
    os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'matrix_utils'))
])
import fourier as fr
from prop_blocks import SinesBlocks, CostModel
from est_quad_coeffs import find_coefficients

# Constants
SCIPY, QP = range(2)


class Group:
    def __init__(self, name: str | int | None,
                 strat: SinesBlocks | None, ntraders: int = 1, **kwargs):
        self.name = name
        self.strat = strat
        self.ntraders = ntraders

    def __str__(self):
        return (f"Group {self.name}: ntraders={self.ntraders}\n"
                f"strat={str(self.strat)}")


class CostModelK:

    def __init__(self, groups: list[Group | None], rho: float, **kwargs):
        self._groups = groups
        self.rho = rho

        # Variables used for cost calculation
        self._mkt = None
        self._d_n = None
        self._Krl = None
        self._N = None
        self._K = None

        # Variables used for optimization
        self.groups_idx = None  # Link a variation model to its source model
        self._var_model = None

    @property
    def K(self) -> int:
        if self._K is None:
            self._K = len(self.groups)
        return self._K

    @property
    def N(self) -> int:
        if self._N is None:
            for g in self.groups:
                if g is not None and g.strat is not None:
                    self._N = g.strat.N
                    break
        return self._N

    @property
    def groups(self) -> list[Group | None]:
        return self._groups

    @groups.setter
    def groups(self, value: list[Group | None]) -> None:
        self._groups = value
        self._reset_intermediate_params()

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

    @property
    def var_model(self) -> "CostModelK":
        """ Return the variation model """
        if self._var_model is None:
            # strats = [SinesBlocks(self.N, blocks=g.strat.blocks, coeff=g.strat.coeff,
            #                       lambd=g.strat.lambd) for g in self.groups]
            # var_groups = [Group(g.name, s, 1) for g, s in zip(self.groups, strats)]
            # self._var_model = self.update_subset_of_groups(var_groups)
            var_groups = deepcopy(self.groups)
            for i, g in enumerate(var_groups):
                if g is not None:
                    g.ntraders = 1

            self._var_model = self.update_subset_of_groups(var_groups)
        return self._var_model

    def _reset_intermediate_params(self) -> None:
        self._mkt = None
        self._d_n = None
        self._Krl = None
        self._K = None
        self._var_model = None

    def __str__(self):
        s = f"CostModel: rho={self.rho}\n"
        for i, t in enumerate(self.groups):
            s += f"{str(t)}\n"
        s += f"Market: {str(self.mkt)}"
        return s

    @property
    def mkt(self) -> SinesBlocks:
        if self._mkt is None:
            blocks_arr = np.zeros(2)
            coeffs = np.zeros(self.N)
            lambd = 0.0
            for i, g in enumerate(self.groups):
                s, n = g.strat, g.ntraders
                blocks_arr += np.array(s.blocks) * n * s.lambd
                coeffs += s.coeff * n * s.lambd
                lambd += n * s.lambd
            blocks = tuple(blocks_arr / lambd)

            coeffs /= lambd
            self._mkt = SinesBlocks(self.N, blocks=blocks, coeff=coeffs, lambd=lambd)
        return self._mkt

    def update_group(self, group: int, **kwargs) -> None:
        """ Update the strategy of group of traders, specified by its position on the list """

        # Get the parameters about the new strategy
        g, s = self.groups[group], self.groups[group].strat  # aliases
        blocks = kwargs.get('blocks', s.blocks)
        coeff = kwargs.get('coeff', s.coeff)
        lambd = kwargs.get('lambd', s.lambd)
        name = kwargs.get('name', None)
        ntraders = kwargs.get('ntraders', g.ntraders)

        new_strat = SinesBlocks(self.N, blocks=blocks, coeff=coeff, lambd=lambd)
        new_group = Group(name, new_strat, ntraders)
        self.groups[group] = new_group
        self._reset_intermediate_params()

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

    def cost_trader(self, group: int = 0, verbose: bool = False, **kwargs) -> float:
        """ Execution cost for trader A"""

        pi, exp, mkt, rho = np.pi, np.exp, self.mkt, self.rho
        N, d_n = self.N, self.d_n
        trd = self.groups[group].strat

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

    def cost_trader_integral(self, group: int = 0, verbose: bool = False, **kwargs) -> float:
        """ Calculate the cost of a trader.  Sams as cost-trader, but evaluate the integral using
            quad
        """
        from scipy.integrate import quad

        trd, mkt = self.groups[group].strat, self.mkt

        # Direct cost of the blocks
        L_block = 0.5 * (trd.blocks[0] * mkt.blocks[0] + trd.blocks[1] * mkt.blocks[1]
                         ) * mkt.lambd + self.price(1) * trd.blocks[1]

        def integrand(t):
            return self.price(t) * trd.deriv(t)

        L_int, _ = quad(integrand, 0, 1)

        L = (L_block + L_int) * trd.lambd

        return L

    @property
    def qp_coeffs(self) -> dict:
        if self._qp_coeffs is None:
            self._qp_coeffs = {}
            for i, t in enumerate(self.strats):
                self._qp_coeffs[i] = self.est_trader_coeffs(i) if t is not None else {}
        return self._qp_coeffs

    @qp_coeffs.setter
    def qp_coeffs(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("qp_coeffs must be a dictionary of dictionaries")
        self._qp_coeffs = value

    def update_subset_of_groups(self, var_groups: list[Group]) -> "CostModelK":
        """
        Generate a new CostModel instance assuming that only some of the traders in the group
        (e.g. just one) update their strategy.  The variance strategy in var_groups replaces the base
        (current) strategy for some of the group' traders (e.g. 1 or 2) specified in the var_groups[i] variable
        All others remain unchanged.  Note that the length of var_groups must be the same as self.groups.
        For groups where there are no changes, the respective var_groups[i] should be None
        """

        # Buils a new cost model from the base model and the variation
        # Maybe also add a dict showing how to map the variation strats to the old ones
        new_groups = []
        groups_idx = {}
        counter = 0
        for i, (g, v) in enumerate(zip(self.groups, var_groups)):
            if v is None:  # No change
                new_groups.append(g)
                groups_idx[i] = counter
                counter += 1
            else:
                # Split the group into the base and the variation
                # Variation strategy applied to a subset of group members
                new_groups.append(v)
                groups_idx[i] = {'variation': counter}
                counter += 1
                if g.ntraders > v.ntraders:
                    # Base strategy
                    base_group = Group(name=g.name, strat=g.strat, ntraders=g.ntraders - v.ntraders)
                    new_groups.append(base_group)
                    groups_idx[i]['base'] = counter
                    counter += 1
                elif g.ntraders < v.ntraders:
                    raise ValueError(f"The number of traders in the variation group: {v.ntraders} "
                                     "must be less or equal to the base group: {g.ntraders}")

        # Create the new cost model
        new_model = CostModelK(groups=new_groups, rho=self.rho)
        new_model.groups_idx = groups_idx

        return new_model

    def est_trader_coeffs(self, group: int = 0, use_blocks: bool = True, use_sines: bool = True) -> dict:
        """ Estimate the trader cost function as bilinear form
            This version allows for having multiple traders running the stame strategy
        """

        # Create a variation model which has both unchanged and changed members for each group
        var_model = self.var_model
        var_idx = var_model.groups_idx[group]['variation']

        # Find the quadratic coefficients
        encode_length = var_model.N * use_sines + 2 * use_blocks  # Length of vector to encode a unit strategy + blocks
        n_stacked = encode_length * len(var_model.groups)

        def Loss(x: np.ndarray) -> float:
            """
            The vector x contains all the coefficients for both traders in the orders:
            a_s, a_e, {a_n}, b_s, b_e, {b_n}
            Assume that only one trader of the class is being updated
            """
            # Unpack the variation strategy from the x-vector
            for i, _ in enumerate(var_model.groups):

                encode_start = i * encode_length

                if use_blocks:
                    blocks = (float(x[encode_start]), float(x[encode_start + 1]))
                    var_model.update_group(group=i, blocks=blocks)
                if use_sines:
                    coeff = np.array(x[encode_start + 2 * use_blocks:encode_start + encode_length])
                    var_model.update_group(group=i, coeff=coeff)

            return var_model.cost_trader(group=var_idx)

        H, f, C = find_coefficients(Loss, n_stacked)
        return {'H': H, 'f': f, 'C': C}

    def solve_equilibrium(self, strat_type: dict | None = None, reg_params: dict | None = None,
                          **kwargs) -> dict:
        """ Solve the equilibrium for the K-trader cost model """

        # Model specification
        use_blocks = strat_type.get('blocks', True) if strat_type is not None else True
        use_sines = strat_type.get('sines', True) if strat_type is not None else True

        # Regularization parameters
        if reg_params is not None:
            raise ValueError("Regularization is not implemented yet")

        # Solve for the coefficients for all trader groups
        qp_coeffs = {}
        for i, g in enumerate(self.groups):
            qp_coeffs[i] = self.est_trader_coeffs(group=i, use_blocks=use_blocks, use_sines=use_sines)

        # Calculate first order conditions
        H_list, f_list = [], []
        var_model = self.var_model
        for i, g in enumerate(self.groups):
            # Determine the rows mask that correspond to the FOC of this trader
            encode_length = var_model.N * use_sines + 2 * use_blocks
            var_idx = var_model.groups_idx[i]['variation']
            encode_start = var_idx * encode_length
            H_var = qp_coeffs[i]['H'][encode_start:encode_start + encode_length]
            f_var = qp_coeffs[i]['f'][encode_start:encode_start + encode_length]
            H_list.append(H_var)
            f_list.append(f_var)
            if 'base' in var_model.groups_idx[i]:
                # Add equations to ensure that variance parameters are equal to the base parameters
                zero_block1 = np.zeros((encode_length, encode_start))
                identity_block = np.eye(encode_length)
                zero_block2 = np.zeros((encode_length,
                                        len(qp_coeffs[i]['H']) - encode_start - 2 * encode_length))
                H_base = np.hstack((zero_block1, identity_block, -identity_block, zero_block2))
                H_list.append(H_base)
                f_list.append(np.zeros(encode_length))

        H, f = np.vstack(H_list), np.hstack(f_list)

        # Solve the system of equations
        x = np.linalg.solve(H, -f)

        # Unpack the coefficients and update the strategies
        for i, g in enumerate(self.groups):
            encode_length = var_model.N * use_sines + 2 * use_blocks
            var_idx = var_model.groups_idx[i]['variation']
            encode_start = var_idx * encode_length
            if use_blocks:
                blocks = (x[encode_start], x[encode_start + 1])
                self.update_group(group=i, blocks=blocks)
            if use_sines:
                coeff = np.array(x[encode_start + 2 * use_blocks:encode_start + encode_length])
                self.update_group(group=i, coeff=coeff)

        return {'H': H, 'f': f, 'x': x}

    def solve_min_cost(self, group: int = 0, **kwargs) -> tuple[SinesBlocks, dict]:
        """ Solve the minimum cost for a trader """
        solver = kwargs.get('solver', SCIPY)
        strat_type = kwargs.get('strat_type', {'blocks': True, 'sines': True})
        abs_tol = kwargs.get('abs_tol', 1e-6)

        if solver == SCIPY:
            print("Using SCIPY solver")

            # Extract initial guess
            use_blocks, use_sines = strat_type['blocks'], strat_type['sines']
            encode_length = 2 * use_blocks + self.N * use_sines
            x = np.zeros(encode_length)
            if use_blocks:
                x[:2] = np.array(self.groups[group].strat.blocks)
            if strat_type['sines']:
                x[2 * use_blocks:] = self.groups[group].strat.coeff

            # Set up a variation model
            var_groups = []
            for i, g in enumerate(self.groups):
                var_groups.append(Group(name='variation', strat=g.strat, ntraders=1)
                                  if i == group else None)
            var_model = self.update_subset_of_groups(var_groups)
            var_trader = var_model.groups_idx[group]['variation']

            def obj_func(x: np.ndarray) -> float:
                if use_blocks:
                    var_model.update_group(group=var_trader, blocks=tuple(x[:2]))
                if use_sines:
                    var_model.update_group(group=var_trader, coeff=x[2 * use_blocks:])
                return var_model.cost_trader(group=var_trader)

            res = minimize(fun= obj_func, x0=x, tol=abs_tol)
            opt_blocks = tuple(res.x[:2]) if use_blocks else (0, 0)
            opt_coeffs = res.x[2 * use_blocks:] if use_sines else np.zeros(self.N)
            opt_strat = SinesBlocks(self.N, blocks=opt_blocks, coeff=opt_coeffs,
                                    lambd=self.groups[group].strat.lambd)
        else:
            raise NotImplementedError("Only SCIPY solver is implemented")

        var_model.update_group(group=var_trader, blocks=opt_blocks, coeff=opt_coeffs)
        res_dict = {'scipy_output': res, 'var_model': var_model, 'var_trader_idx': var_trader}
        return opt_strat, res_dict
