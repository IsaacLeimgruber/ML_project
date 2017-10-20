import numpy as np
from sigmoid import *

def compute_gradient(y, tx, w):
    e = y - tx @ w
    grad = -np.transpose(tx)@e/len(e)
    return grad;


def compute_stoch_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)
    return grad/len(e);


def calculate_gradient(y, tx, w):
    
    return  tx.T @(sigmoid(tx @ w) - y)
