# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.
        
        You can calculate the loss using mse or mae.
        """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    e = y - tx @ w;
    double_length = 2*len(e);
    mse = np.sum(e * e) / double_length;
    
    return mse;


def least_squares(y, tx):
    """calculate the least squares solution."""
    
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_loss(y, tx, w)
    return w, mse
