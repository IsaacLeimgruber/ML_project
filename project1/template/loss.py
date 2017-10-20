import numpy as np

def compute_loss(y, tx, w):
    e = y - tx @ w
    return np.mean(e**2)/2.


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    loss = np.log(1 + np.exp(tx @ w)) - (y * (tx @ w))
    
    sum_loss = np.sum(loss)
    
    return sum_loss
