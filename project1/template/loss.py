import numpy as np

def compute_loss(y, tx, w):
    e = y - tx @ w
    return np.mean(e**2)/2.


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    tx_w = tx @ w
    
    ex = np.exp(tx_w)
    
    loss = np.log(1 + ex) - (y * (tx_w))
    
    sum_loss = np.sum(loss)
    
    return sum_loss
