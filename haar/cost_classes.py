""" Class to model N-Trader cost functions using Haar Wavelets
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
    os.path.abspath(os.path.join(CURRENT_DIR, '..', 'matrix_utils'))
])
import haar_funcs_fast as hf
from est_quad_coeffs import find_coefficients

# Constants
SCIPY, QP = range(2)
SOLVER_METHOD = 'SLSQP'  # SLSQP, trust-constr
# Global
iteration_count = 0  # Initialize the iteration count


class HaarStrat:
    def __init__(self, level: int, coeff: np.ndarray | None = None, lambd: float = 1.0,
                 **kwargs):
        """ Initialize the Haar strategy
        """
        self.level = level
        self.coeff = np.zeros(N) if coeff is None else coeff
        self.lambd = lambd

        self._lmum = kwargs.get('lmum', None)
        self._N = 2 ** level

    @property
    def lmum(self):
        if self._lmum is None:
            self._lmum = hf.calc_lmum(self.level)
        return self._lmum

    @property
    def N(self) -> int:
        if self._N is None:
            self._N = 2 ** self.level
        return self._N

    def __str__(self):
        return (f"HaarStrat: level={self.level}, "
                f"\tcoeff={np.array2string(self.coeff, precision=4, separator=', ', suppress_small=True)}, "
                f"lambd={self.lambd:.4g}")

    def calc(self, t: float) -> float:
        """ Calculate the value of the function a(t) at time t """
        return hf.integrate_haar(self.coeff, 0, t, self.lmum)

    def deriv(self, t: float, order: int = 1) -> float:
        """ Calculate the derivate of  at time t (i.e. the trading intensity)
            It is encoded with the Haar wavelet coefficients
        """
        return hf.reconstruct_from_haar(t, self.coeff, self.lmum)

    @classmethod
    def from_deriv(cls, func: Callable, **kwargs) -> "HaarStrat":
        """ Generate a HaarStrat instance from a trading intensitfy function
            (i.e. the derivative of the trading trajectory)
            It is recommended (but not enforced) that the function integrates to 1
            over the interval [0, 1]
            If a function takes parameters, they can be passed as a tuple with the func_args key
        """
        level = kwargs.get('level', 6)
        lambd = kwargs.get('lambd', 1.0)
        func_args = kwargs.get('func_args', ())

        # Generate the Haar wavelet coefficients
        coeff = hf.haar_coeff(func, level, func_args=func_args)
        return cls(level, coeff=coeff, lambd=lambd)

    @classmethod
    def from_func(cls, func: Callable, **kwargs) -> "HaarStrat":
        """ Generate a HaarStrat instance from the trading trajectory
            It assumes that the function is defined on the interval [0, 1]
            It is recommended (but not enforced) that f(0)=0, f(1)=1.
            If a function takes parameters, they can be passed as a tuple with the func_args key
        """
        level = kwargs.get('level', 6)
        lambd = kwargs.get('lambd', 1.0)
        func_args = kwargs.get('func_args', ())

        step = 1e-6

        def deriv(t: float) -> float:
            return (func(t + step, *func_args) - func(t - step, *func_args)) / (2 * step)

        # Generate the Haar wavelet coefficients
        coeff = hf.haar_coeff(deriv, level)
        return cls(level, coeff=coeff, lambd=lambd)


class Group:
    def __init__(self, name: str | int | None,
                 strat: HaarStrat | None, ntraders: int = 1, **kwargs):
        self.name = name
        self.strat = strat
        self.ntraders = ntraders

    def __str__(self):
        return (f"Group {self.name}: ntraders={self.ntraders}\n"
                f"strat={str(self.strat)}")


class CostHaarK:

    def __init__(self, groups: list[Group | None], rho: float, **kwargs):
        self._groups = groups
        self.rho = rho

        # Variables used for cost calculation
        self._mkt = None
        self._level = None
        self._N = None
        self._K = None
        self._lmum = None

        # Variables used for optimization
        self.groups_idx = None  # Link a variation model to its source model
        self._var_model = None

    @property
    def K(self) -> int:
        if self._K is None:
            self._K = len(self.groups)
        return self._K

    @property
    def level(self) -> int:
        if self._level is None:
            for g in self.groups:
                if g is not None and g.strat is not None:
                    self._level = g.strat.level
                    self._lmum = None
                    break
        return self._level

    @property
    def N(self) -> int:
        if self._N is None:
            if self.level is not None:
                self._N = 2 ** self.level
        return self._N

    @property
    def lmum(self):
        if self._lmum is None:
            self._lmum = hf.calc_lmum(self.level)
        return self._lmum

    @property
    def groups(self) -> list[Group | None]:
        return self._groups

    @groups.setter
    def groups(self, value: list[Group | None]) -> None:
        self._groups = value
        self._reset_intermediate_params()

    @property
    def var_model(self) -> "CostHaarK":
        """ Return the variation model """
        if self._var_model is None:
            var_groups = deepcopy(self.groups)
            for i, g in enumerate(var_groups):
                if g is not None:
                    g.ntraders = 1

            self._var_model = self.update_subset_of_groups(var_groups)
        return self._var_model

    def _reset_intermediate_params(self) -> None:
        self._mkt = None
        self._K = None
        self._var_model = None

    def __str__(self):
        s = f"Cost Model (Haar) K-Trader: rho={self.rho}\n"
        for i, t in enumerate(self.groups):
            s += f"{str(t)}\n"
        s += f"Market: {str(self.mkt)}"
        return s

    @property
    def mkt(self) -> HaarStrat:
        """ Return the market trading strategy """
        if self._mkt is None:
            lambd_list = [g.ntraders * g.strat.lambd for g in self.groups if g is not None]
            coeff_list = [g.strat.coeff for g in self.groups if g is not None]
            lambd = sum(lambd_list)
            if lambd == 0:
                mcoeff = np.zeros(self.N)
            else:
                lamd_scaled_list = [l / lambd for l in lambd_list]
                mcoeff = hf.add_haar(coeff_list, lamd_scaled_list)
            self._mkt = HaarStrat(self.level, coeff=mcoeff, lambd=lambd, lmum=self.lmum)
        return self._mkt

    def update_group(self, group: int, **kwargs) -> None:
        """ Update the strategy of group of traders, specified by its position on the list """

        # Get the parameters about the new strategy
        g, s = self.groups[group], self.groups[group].strat  # aliases
        coeff = kwargs.get('coeff', s.coeff)
        lambd = kwargs.get('lambd', s.lambd)
        name = kwargs.get('name', None)
        ntraders = kwargs.get('ntraders', g.ntraders)

        new_strat = HaarStrat(self.level, coeff=coeff, lambd=lambd, lmum=self.lmum)
        new_group = Group(name, new_strat, ntraders)
        self.groups[group] = new_group
        self._reset_intermediate_params()

    def price(self, t: float, **kwargs) -> float:
        """ Calculate the price impact at time t """

        p_unit = hf.price_haar(t, haar_coef=self.mkt.coeff, rho=self.rho,
                               lmum=self.lmum)

        return p_unit * self.mkt.lambd

    def cost_trader(self, group: int = 0, verbose: bool = False, **kwargs) -> float:
        """ Execution cost for o trader from group 'group'"""

        trd = self.groups[group].strat
        mkt = self.mkt
        scale_factor = trd.lambd * mkt.lambd
        if scale_factor == 0:  # If the trader is not active, return 0
            return 0

        # Calculate the cost
        l_unit = hf.cost_quad(trd.coeff, mkt.coeff, rho=self.rho, lmum=self.lmum)
        return l_unit * scale_factor

    @property
    def qp_coeffs(self) -> dict:
        if self._qp_coeffs is None:
            self._qp_coeffs = {}
            for i, s in enumerate(self.strats):
                self._qp_coeffs[i] = self.est_trader_coeffs(i) if s is not None else {}
        return self._qp_coeffs

    @qp_coeffs.setter
    def qp_coeffs(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("qp_coeffs must be a dictionary of dictionaries")
        self._qp_coeffs = value

    def update_subset_of_groups(self, var_groups: list[Group]) -> "CostHaarK":
        """
        Generate a new CostHaarK instance assuming that only some of the traders in the group
        (e.g. just one) update their strategy.  The variance strategy in var_groups replaces the base
        (current) strategy for some of the group' traders (e.g. 1 or 2) specified in the var_groups[i] variable
        All others remain unchanged.  Note that the length of var_groups must be the same as self.groups.
        For groups where there are no changes, the respective var_groups[i] should be None
        """

        # Builds a new cost model from the base model and the variation
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
        new_model = CostHaarK(groups=new_groups, rho=self.rho, lmum=self.lmum)
        new_model.groups_idx = groups_idx

        return new_model

    def est_trader_coeffs(self, group: int = 0) -> dict:
        """ Estimate the trader cost function as bilinear form
            This version allows for having multiple traders running the stame strategy
        """

        # Create a variation model which has both unchanged and changed members for each group
        var_model = self.var_model
        var_idx = var_model.groups_idx[group]['variation']

        # Find the quadratic coefficients
        encode_length = var_model.N  # Length of vector to encode a unit strategy
        n_stacked = encode_length * len(var_model.groups)

        def Loss(x: np.ndarray) -> float:
            """
            The vector x contains stacked coefficients for all traders in the market:
            When the gradient is calculated, assume that only one trader of the group is being updated
            """
            # Unpack the variation strategy from the x-vector
            for i, _ in enumerate(var_model.groups):
                encode_start = i * encode_length
                coeff = np.array(x[encode_start:encode_start + encode_length])
                var_model.update_group(group=i, coeff=coeff)

            return var_model.cost_trader(group=var_idx)

        H, f, C = find_coefficients(Loss, n_stacked)
        return {'H': H, 'f': f, 'C': C}

    def solve_equilibrium(self, **kwargs) -> dict:
        """ Solve the equilibrium for the K-trader cost model """

        # Regularization parameters
        if reg_params is not None:
            raise ValueError("Regularization is not implemented yet")

        # Solve for the coefficients for all trader groups
        qp_coeffs = {}
        for i, g in enumerate(self.groups):
            qp_coeffs[i] = self.est_trader_coeffs(group=i)

        # Calculate first order conditions
        H_list, f_list = [], []
        var_model = self.var_model
        for i, g in enumerate(self.groups):
            # Determine the rows mask that correspond to the FOC of this trader
            encode_length = var_model.N
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
            encode_length = var_model.N
            var_idx = var_model.groups_idx[i]['variation']
            encode_start = var_idx * encode_length
            coeff = np.array(x[encode_start:encode_start + encode_length])
            self.update_group(group=i, coeff=coeff)

        return {'H': H, 'f': f, 'x': x}

    def solve_min_cost(self, group: int = 0, **kwargs) -> tuple[HaarStrat, dict]:
        """ Solve the minimum cost for a trader """
        solver = kwargs.get('solver', SCIPY)
        abs_tol = kwargs.get('abs_tol', 1e-6)
        gstrat = self.groups[group].strat

        if solver == SCIPY:
            print("Using SCIPY solver")

            # Define the callback function
            def progress_callback(x):
                global iteration_count
                iteration_count += 1
                if iteration_count % 10 == 0:
                    obj_value = obj_func(x)
                    print(f"Iteration {iteration_count}: Parameters = {x},\n"
                          f" Objective Function Value = {obj_value}")

            # Extract initial guess
            x = gstrat.coeff[1:].copy()

            # Set up a variation model
            var_groups = []
            for i, g in enumerate(self.groups):
                var_groups.append(Group(name='variation', strat=g.strat, ntraders=1)
                                  if i == group else None)
            var_model = self.update_subset_of_groups(var_groups)
            var_trader = var_model.groups_idx[group]['variation']

            def obj_func(x: np.ndarray) -> float:
                coeff = np.zeros(self.N)
                coeff[0] = gstrat.coeff[0]
                coeff[1:] = x
                var_model.update_group(group=var_trader, coeff=coeff)
                return var_model.cost_trader(group=var_trader)

            res = minimize(fun=obj_func, x0=x, tol=abs_tol, method=SOLVER_METHOD,
                           callback=progress_callback)
            opt_coeffs = gstrat.coeff.copy()
            opt_coeffs[1:] = res.x
            opt_strat = HaarStrat(self.level, coeff=opt_coeffs,
                                  lambd=self.groups[group].strat.lambd, lmum=self.lmum)
        else:
            raise NotImplementedError("Only SCIPY solver is implemented")

        var_model.update_group(group=var_trader, coeff=opt_coeffs)
        res_dict = {'scipy_output': res, 'var_model': var_model, 'var_trader_idx': var_trader}
        return opt_strat, res_dict
