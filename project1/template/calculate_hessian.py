import numpy as np
from sigmoid import *

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # calculate hessian
    S = sigmoid(tx @ w) * (1- sigmoid(tx @ w))
    
    i = np.identity(len(S))
    S = S.T * i
    
    Sx = S @ tx
    
    H = tx.T @ Sx
    
    return H
