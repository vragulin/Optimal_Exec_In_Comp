import os
import sys
import pytest as pt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '../../../..',
                                             'lin_propagator_qp/equilibrium/ow_exact')))
import ow_cost as ow


def test_cost_zero_ab():
    lambd = 2
    rho = 10
    m = ow.OW(rho, lambd)
    exp_a = 0.27
    exp_b = 0.54
    assert np.isclose(m.cost_a, exp_a), "Cost A is incorrect"
    assert np.isclose(m.cost_b, exp_b), "Cost B is incorrect"


@pt.mark.parametrize("a,b, exp_a, exp_b", [
    ((0.2, 0.4), (0.3, 0.5), 0.452812, 1.08083),
    ((0.0, 0.0), (0.0, 0.0), 0.27, 0.54),
    ((0.1, 0.2), (0.2, 0.1), 0.274302, 0.516601),
])
def test_cost_ab(a, b, exp_a, exp_b):
    lambd = 2
    rho = 10
    m = ow.OW(rho, lambd, a=a, b=b)
    m.a = np.array(a)
    m.b = np.array(b)
    assert np.isclose(m.cost_a, exp_a, atol=1e-6), "Cost A is incorrect"
    assert np.isclose(m.cost_b, exp_b, atol=1e-6), "Cost B is incorrect"


def test_grad_a_zero():
    lambd = 2
    rho = 10
    m = ow.OW(rho, lambd)
    grad = m.grad_a()
    exp_grad = np.array([-0.260006, -0.0600154])
    assert np.allclose(grad, exp_grad, atol=1e-5), "Gradient A is incorrect"


@pt.mark.parametrize("a,b, exp_grad", [
    ((0.0, 0.0), (0.0, 0.0), (-0.260006, -0.0600154)),
    ((0.2, 0.4), (0.3, 0.5), (0.312021, 0.752038)),
    ((0.1, 0.2), (0.2, 0.1), (0.0480078, 0.188015)),
])
def test_grad_a(a,b, exp_grad):
    lambd = 2
    rho = 10
    m = ow.OW(rho, lambd, a=a, b=b)
    grad = m.grad_a()
    assert np.allclose(grad, exp_grad, atol=1e-5), "Gradient A is incorrect"


@pt.mark.parametrize("a,b, exp_grad", [
    ((0.0, 0.0), (0.0, 0.0), (-0.500023, -0.300032)),
    ((0.2, 0.0), (0.0, 0.0), (-0.304021, -0.34401)),
    ((0.1, 0.2), (0.2, 0.3), (0.394053, 1.03404)),
])
def test_grad_b(a, b, exp_grad):
    lambd = 2
    rho = 10
    m = ow.OW(rho, lambd, a=a, b=b)
    grad = m.grad_b()
    assert np.allclose(grad, exp_grad, atol=1e-6), "Gradient A is incorrect"
