# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def compute_loss(y, tx, w):
    e = y - tx @ w
    loss = np.sum(e*e)/(2*len(e))
    return loss;

def ridge_regression(y, tx, lambda_):

    lambda_prime = 2*len(tx)*lambda_
    first_term = tx.T@tx
    #sum ici les x*w
    
    w = np.linalg.solve(first_term + lambda_prime *np.identity(len(first_term)), tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss;
