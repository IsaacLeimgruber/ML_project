from loss import calculate_loss
from gradient import calculate_gradient
from calculate_hessian import *

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian
    loss = calculate_loss(y, tx, w) + (w.T@w*lambda_ / 2.0)
    
    grad = calculate_gradient(y, tx, w) + (w*lambda_)
    
    H = calculate_hessian(y, tx, w) + (w.T@w*lambda_ / 2.0)
    
    
    return loss, grad, H
