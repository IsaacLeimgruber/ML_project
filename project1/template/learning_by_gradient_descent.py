from loss import *
from gradient import *

def learning_by_gradient_descent(y, tx, w, gamma):
    # compute the cost
    loss = calculate_loss(y, tx, w)
    # compute the gradient
    grad = calculate_gradient(y, tx, w)
    # update w
    w = w - gamma*grad
    
    return loss, w
