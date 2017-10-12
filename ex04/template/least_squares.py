# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_loss(y, tx, w):
    e = y - tx @ w
    loss = np.sum(e*e)/(2*len(e))
    return loss;

def least_squares(y, tx):
    """calculate the least squares solution."""
    
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss;
