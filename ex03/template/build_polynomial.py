# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi_x = np.ones((len(x), 1))
        
    for j in range(1, degree + 1):
        x_power_j = np.power(x, j)
        phi_x = np.c_[phi_x, x_power_j]

    return phi_x
