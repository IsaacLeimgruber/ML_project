import numpy as np
from penalized_logistic_regression import *

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
        Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
        """
    # return loss, gradient
    loss, grad, H = penalized_logistic_regression(y, tx, w, lambda_)
    # update w
    
    w = w - gamma * np.linalg.solve(H, grad);
    
    
    return loss, w
