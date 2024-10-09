""" Test the computation of the first order correction terms for the OW Cost Function
    V. Ragulin - 10/09/2024
"""
import os
import sys
import pytest as pt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'cost_function')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'lin_propagator_qp')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'lin_propagator_qp/equilibrium')))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../../..', 'matrix_utils')))
from prop_cost_to_QP import cost_to_QP_numerical, cost_QP_params
from analytic_solution import foc_terms_a, foc_terms_b
from propagator import cost_fn_prop_b_approx
from est_quad_coeffs import find_coefficients

DEBUG = True
np.random.seed(0)


@pt.mark.parametrize("n", [2, 4, 6, 8, 10, 20, 100])
def test_foc_terms_a(n):
    lambd = 1
    rho = 10

    # Calculate the first order correction terms
    terms = foc_terms_a(n, lambd, rho)
    H = terms['H']
    G = terms['G']
    m = terms['m']

    # Check against the cost_to_QP_calculation assuming random b_n
    a_n = np.random.randn(n)
    b_n = np.random.randn(n)

    H0, f0, C0 = cost_to_QP_numerical(a_n, b_n, lambd, rho)
    f = G @ b_n + m

    # Check that the matrices are correct
    assert np.allclose(H, H0), "Hessian matrix is incorrect"
    assert np.allclose(f, f0), "Linear coefficients vector is incorrect"


@pt.mark.parametrize("n", [2, 4, 6, 8, 10, 20, 100])
def test_foc_terms_b(n):
    lambd = 1
    rho = 10

    # Calculate the first order correction terms
    terms = foc_terms_b(n, lambd, rho)
    H = terms['H']
    G = terms['G']
    m = terms['m']

    # Check against the cost_to_QP_calculation assuming random a_n
    a_n = np.random.randn(n)
    b_n = np.random.randn(n)
    f = G @ a_n + m

    # Fit the Hessian and gradient from the cost function
    S = lambda x: cost_fn_prop_b_approx(a_n, x, lambd, rho)
    H0, f0, C0 = find_coefficients(S, n)

    # Check that the matrices are correct
    assert np.allclose(H, H0), "Hessian matrix is incorrect"
    assert np.allclose(f, f0), "Linear coefficients vector is incorrect"
