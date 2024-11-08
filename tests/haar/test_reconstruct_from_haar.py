import pytest as pt
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../haar')))
import haar_funcs as hf

DEBUG = True


@pt.mark.parametrize("c", [0, 1, 3])
@pt.mark.parametrize("level", [1, 2, 5])
def test_reconstruct_const(c, level):
    def func(t):
        return c

    c0, coeffs = hf.haar_coeff(func, (0, 1), level)
    recon = hf.reconstruct_from_haar((c0, coeffs), 0.33)

    if DEBUG:
        print("recon:", recon)
    assert pt.approx(recon, abs=1e-6) == c


@pt.mark.parametrize("c", [0, 3])
@pt.mark.parametrize("level", [1, 2, 3, 4, 5, 6])
def test_reconstruct_linear(c, level):
    plot = False
    def func(t):
        return c * t
    t = 0.33
    c0, coeffs = hf.haar_coeff(func, (0, 1), level)
    recon = hf.reconstruct_from_haar((c0, coeffs), t)
    exp = c * t

    # Estimate the diffence between the expected and reconstructed values
    t_values0 = np.linspace(0, 1, 101)
    t_values = t_values0[:-1]
    exp_values = c * t_values
    recon_values = [hf.reconstruct_from_haar((c0, coeffs), t) for t in t_values]
    diff = np.array(exp_values) - np.array(recon_values)
    mean_error = np.linalg.norm(diff) / np.sqrt(len(diff))
    assert mean_error < 1 / 2 ** level
    if DEBUG:
        print("mean_error:", mean_error)
        print("recon:", recon, "exp:", exp)
        if plot:
            plt.plot(t_values, exp_values, label="Expected")
            plt.plot(t_values, recon_values, label="Reconstructed")
            plt.title(f"Reconstruction of {c} * t, level={level}")
            plt.legend()
            plt.show()
    assert pt.approx(recon, abs=1/level**2) == exp


@pt.mark.parametrize("level", [1, 2, 3, 4, 5, 6])
@pt.mark.parametrize("c", [0, 1])
def test_reconstruct_square(level, c):
    plot = True
    def func(t):
        return c * t**2
    t = 0.33
    c0, coeffs = hf.haar_coeff(func, (0, 1), level)
    recon = hf.reconstruct_from_haar((c0, coeffs), t)
    exp = func(t)

    # Estimate the diffence between the expected and reconstructed values
    t_values0 = np.linspace(0, 1, 101)
    t_values = t_values0[:-1]
    exp_values = [func(t) for t in t_values]
    recon_values = [hf.reconstruct_from_haar((c0, coeffs), t) for t in t_values]
    diff = np.array(exp_values) - np.array(recon_values)
    mean_error = np.linalg.norm(diff) / np.sqrt(len(diff))
    assert mean_error < 1 / 2 ** level
    if DEBUG:
        print("mean_error:", mean_error)
        print("recon:", recon, "exp:", exp)
        if plot:
            plt.plot(t_values, exp_values, label="Expected")
            plt.plot(t_values, recon_values, label="Reconstructed")
            plt.title(f"Reconstruction of {c} * t, level={level}")
            plt.legend()
            plt.show()
    assert pt.approx(recon, abs=1/level**2) == exp


@pt.mark.parametrize("level", [1, 2, 3, 4, 5, 6])
@pt.mark.parametrize("c", [1, 2, 5, 10])
def test_reconstruct_sin(level, c):
    plot = True
    def func(t):
        return np.sin(np.pi * c * t)

    t = 0.33
    c0, coeffs = hf.haar_coeff(func, (0, 1), level)
    recon = hf.reconstruct_from_haar((c0, coeffs), t)
    exp = func(t)

    # Estimate the diffence between the expected and reconstructed values
    t_values0 = np.linspace(0, 1, 101)
    t_values = t_values0[:-1]
    exp_values = [func(t) for t in t_values]
    recon_values = [hf.reconstruct_from_haar((c0, coeffs), t) for t in t_values]
    diff = np.array(exp_values) - np.array(recon_values)
    mean_error = np.linalg.norm(diff) / np.sqrt(len(diff))
    assert mean_error < c / 2 ** level
    if DEBUG:
        print("mean_error:", mean_error)
        print("recon:", recon, "exp:", exp)
        if plot:
            plt.plot(t_values, exp_values, label="Expected")
            plt.plot(t_values, recon_values, label="Reconstructed")
            plt.title(f"Reconstruction of {c} * t, level={level}")
            plt.legend()
            plt.show()
    assert pt.approx(recon, abs=c/level**2) == exp
