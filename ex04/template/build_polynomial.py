# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np



def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    # problem ici avec 1 et sum
    phi_x = []
    
    for j in range(0, degree + 1):
        if(j == 0):
            phi_x = np.power(x, j)
        else:
            x_power_j = np.power(x, j)
            phi_x = np.c_[phi_x, x_power_j]

    return phi_x
