# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx @ w
    mse = np.sum(e*e)/(2*len(e))
    return mse