""" Haar wavelet class implementation.
V. Ragulin - 11/08/2024
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize
from typing import Callable

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class HaarWavelet:
    def __init__(self, level: int, ):
        pass